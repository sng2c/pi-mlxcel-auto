/**
 * mlxcel-auto
 *
 * Auto-discovers a running `mlxcel-server` (OpenAI-compatible) and registers
 * its served models with pi, reading each model's real context window from
 * config.json so you never type ctx-size manually. Supports http and https,
 * local and remote servers.
 *
 * Metadata source is remote-first (Hugging Face), falling back to the local
 * model store when HF is unreachable, the model is gated/private, or the id is
 * a local path. The local search covers both the mlxcel store and the Hugging
 * Face hub cache, so mlx-lm models are found as well. Results are cached.
 *
 * Beyond context, also detects:
 *   - reasoning/thinking: from chat_template tokens (enable_thinking /
 *     reasoning_content / clear_thinking / think). Sets `reasoning: true` and
 *     `compat.thinkingFormat: "qwen-chat-template"` so pi drives MLX's
 *     `chat_template_kwargs.enable_thinking` (off disables, others enable).
 *   - vision: from `vision_config` or tokenizer image/video tokens. Sets
 *     `input: ["text","image"]`.
 *   - tools: from chat template tool-call markers or `tool_parser_type`.
 *     Informational only (pi has no per-model tool toggle); cached as metadata.
 *
 * Config (env vars):
 *   MLXCEL_AUTO_BASEURLS       comma-separated server base URLs (default: "http://127.0.0.1:8080")
 *                              e.g. "http://127.0.0.1:8080,https://remote.example.com:8443"
 *                              Scheme is optional (defaults to http).
 *   MLXCEL_AUTO_APIKEY         api key sent to the server (default: "not-needed")
 *   MLXCEL_AUTO_MAXOUT         cap on maxTokens (default: 32768)
 *   MLXCEL_AUTO_FALLBACK_CTX   context window when detection fails (default: 32768)
 *   MLXCEL_AUTO_NO_REASONING   "1" disables automatic reasoning detection
 *   MLXCEL_AUTO_NO_CACHE        "1" disables the on-disk config cache
 *   MLXCEL_DEFAULT_ORG         org for bare model names, tried first before HF search (default: "mlx-community")
 *   MLXCEL_MODELS_DIR           override mlxcel model-store root (default: unset)
 *   MLXCEL_CACHE_DIR            override mlxcel cache root (default: "~/.cache/mlxcel")
 *   HF_HUB_CACHE / HF_HOME      Hugging Face hub cache location (used for mlx-lm models)
 *
 * Registers a provider per base URL; the provider id is "mlxcel-auto" for the
 * default (http on 127.0.0.1:8080) or "mlxcel-auto-{host}-{port}" otherwise.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { tokenize } from "@huggingface/jinja";
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const CACHE_PATH = `${homedir()}/.pi/agent/extensions-data/mlxcel-auto-cache.json`;
const PROBE_TIMEOUT_MS = 1500;
const HTTP_TIMEOUT_MS = 4000;

const CTX_KEYS = [
  "max_position_embeddings",
  "context_length",
  "max_seq_len",
  "seq_length",
  "n_positions",
  "model_max_length",
] as const;

type Cfg = Record<string, any> | null;

// Trailing stop tokens some MLX chat templates leak into assistant text. These
// are stripped from the finalized assistant message via a `message_end` hook.
// Per-model string eos tokens from config/tokenizer_config are added at reg.
const STOP_TOKENS = [
  "<|im_end|>",
  "<|endofsentext|>",
  "",
  "<end_of_turn>",
  "<start_of_turn>",
];
const modelStopTokens = new Map<string, string[]>();

interface CacheEntry {
  modelMaxCtx: number;
  vision: boolean;
  reasoning: boolean;
  tools: boolean;
  source: "local" | "hf" | "fallback";
  ts: number;
  // Informational metadata from HF config docs (not used in registration).
  modelType?: string;
  architectures?: string[];
  eosToken?: string | number;
  quantization?: string;
  genMaxNewTokens?: number;
  // TODO(future): split modality flags once pi supports more than text/image.
  //   audio?: boolean;  // tokenizer_config.audio_token / chat_template audio ids
  //   video?: boolean;  // tokenizer_config.video_token / chat_template video ids
  // Detect via VISION/audio/video token keys + template ids; pi `input` only
  // accepts "text" | "image" today, so these would be metadata-only until pi
  // extends the model `input` union.
}

interface ModelMeta {
  cfg: Cfg;
  tokCfg: Cfg;        // tokenizer_config.json (may be null)
  genCfg: Cfg;        // generation_config.json (may be null)
  template: string;   // chat template text (jinja file or embedded string)
  source: CacheEntry["source"];
}
type Cache = Record<string, CacheEntry>;

function env(key: string, def: string): string {
  const v = process.env[key];
  return v && v.trim() ? v.trim() : def;
}
function envInt(key: string, def: number): number {
  const v = process.env[key];
  if (!v) return def;
  const n = Number(v);
  return Number.isFinite(n) ? n : def;
}

async function fetchJson(url: string, timeoutMs: number): Promise<any | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

// --- local model dir resolution --------------------------------------------

// Resolve a model id (as returned by /v1/models) to an `owner/name` repo id.
// - Full repo id (contains "/"): returned as-is.
// - Local path (starts with "/", "./", "../"): null (not a HF repo).
// - Bare name: resolved as MLXCEL_DEFAULT_ORG/<name> (default "mlx-community").
//   If that HF URL returns 404, the caller falls back to the HF search API
//   to find the correct org automatically.
function resolveRepoId(modelId: string): string | null {
  if (!modelId) return null;
  if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")) {
    return null;
  }
  if (modelId.includes("/")) return modelId; // owner/name already
  const org = env("MLXCEL_DEFAULT_ORG", "mlx-community");
  return `${org}/${modelId}`;
}

// Search the HF Hub API for a model matching the given bare name.
// Returns the repo id of the most downloaded match, or null.
async function searchHfModel(bareName: string): Promise<string | null> {
  const data = await fetchJson(
    `https://huggingface.co/api/models?search=${encodeURIComponent(bareName)}&sort=downloads&direction=-1&limit=5`,
    HTTP_TIMEOUT_MS,
  );
  if (!data || !Array.isArray(data)) return null;
  for (const m of data) {
    const id: string = m?.id ?? "";
    // Prefer exact match on the name part (after "/")
    if (id && id.split("/").pop() === bareName) return id;
  }
  // No exact name match; return the first result if any
  if (data.length > 0 && typeof data[0]?.id === "string") return data[0].id;
  return null;
}

// Find a local model directory by scanning for the model name, org-agnostic.
// Searches: 1) explicit MLXCEL_MODELS_DIR, 2) mlxcel store, 3) HF hub cache.
// For stores with an org/ directory structure, scans all orgs for a matching name.
// For the HF hub cache, scans models--*--<name> patterns.
function localModelDir(modelId: string): string | null {
  // Local directory path passed as -m?
  if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")) {
    const maybeDir = modelId.startsWith("/") ? modelId : join(process.cwd(), modelId);
    if (existsSync(join(maybeDir, "config.json"))) return maybeDir;
  }

  // Determine name (after "/") and a candidate owner for exact-match paths.
  const slashIdx = modelId.indexOf("/");
  const name = slashIdx >= 0 ? modelId.slice(slashIdx + 1) : modelId;
  const knownOwner = slashIdx >= 0 ? modelId.slice(0, slashIdx) : null;

  const candidates: string[] = [];

  // 1) Explicit override (uses known owner or MLXCEL_DEFAULT_ORG)
  const md = process.env.MLXCEL_MODELS_DIR;
  if (md) {
    if (knownOwner) {
      candidates.push(join(md, knownOwner, name));
    } else {
      const org = env("MLXCEL_DEFAULT_ORG", "mlx-community");
      candidates.push(join(md, org, name));
    }
  }

  // 2) mlxcel store: scan all org dirs for a matching model name
  const cacheDir = process.env.MLXCEL_CACHE_DIR || `${homedir()}/.cache/mlxcel`;
  const mlxcelModels = join(cacheDir, "models");
  if (existsSync(mlxcelModels)) {
    try {
      for (const entry of readdirSync(mlxcelModels)) {
        const candidate = join(mlxcelModels, entry, name);
        if (existsSync(join(candidate, "config.json"))) {
          candidates.push(candidate);
        }
      }
    } catch {}
  }

  // 3) HF hub cache: scan models--*--<name> patterns
  const hfCacheDir = process.env.HF_HUB_CACHE || process.env.HF_HOME
    ? join(process.env.HF_HUB_CACHE || process.env.HF_HOME!, "hub")
    : `${homedir()}/.cache/huggingface/hub`;
  const hfModelsPrefix = `models--`;
  const hfNamePattern = `--${name}`;
  if (existsSync(hfCacheDir)) {
    try {
      for (const entry of readdirSync(hfCacheDir)) {
        if (entry.startsWith(hfModelsPrefix) && entry.endsWith(hfNamePattern)) {
          const snapshotsDir = join(hfCacheDir, entry, "snapshots");
          try {
            const commits = readdirSync(snapshotsDir).sort();
            const latest = commits[commits.length - 1];
            if (latest && existsSync(join(snapshotsDir, latest, "config.json"))) {
              candidates.push(join(snapshotsDir, latest));
            }
          } catch {}
        }
      }
    } catch {}
  }

  // Return the first candidate with config.json (prioritize exact match)
  for (const d of candidates) {
    if (existsSync(join(d, "config.json"))) return d;
  }
  return null;
}

function readJsonFile(path: string): any | null {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

function readTextFile(path: string): string | null {
  try {
    return readFileSync(path, "utf8");
  } catch {
    return null;
  }
}

// Extract the chat template text from tokenizer_config.json (string or
// list-of-templates) or from a sibling chat_template.jinja/.json file.
function extractTemplate(tokCfg: Cfg, dir: string | null): string {
  if (tokCfg) {
    const ct = (tokCfg as any).chat_template;
    if (typeof ct === "string") return ct;
    if (Array.isArray(ct)) {
      const parts = ct
        .map((x) => (x && typeof x === "object" ? (x as any).template : x))
        .filter((x) => typeof x === "string");
      if (parts.length) return parts.join("\n");
    }
  }
  if (dir) {
    for (const f of ["chat_template.jinja", "chat_template.json"]) {
      const t = readTextFile(join(dir, f));
      if (t) return t;
    }
  }
  return "";
}

async function fetchText(url: string, timeoutMs: number): Promise<string | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    if (!r.ok) return null;
    return await r.text();
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

async function fetchHfMeta(repoId: string): Promise<ModelMeta | null> {
  const cfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/config.json`, HTTP_TIMEOUT_MS);
  if (!cfg || typeof cfg !== "object") return null;
  const tokCfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/tokenizer_config.json`, HTTP_TIMEOUT_MS);
  const genCfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/generation_config.json`, HTTP_TIMEOUT_MS);
  let template = extractTemplate(tokCfg, null);
  if (!template) {
    const jinja = await fetchText(`https://huggingface.co/${repoId}/raw/main/chat_template.jinja`, HTTP_TIMEOUT_MS);
    if (jinja) template = jinja;
  }
  return { cfg, tokCfg, genCfg, template, source: "hf" };
}

async function loadModelMeta(modelId: string): Promise<ModelMeta | null> {
  // 1) Hugging Face remote-first.
  //    - Full repo id (owner/name): try directly.
  //    - Bare name: try MLXCEL_DEFAULT_ORG/name first, then HF search API.
  if (modelId.includes("/")) {
    // owner/name already — try directly
    const meta = await fetchHfMeta(modelId);
    if (meta) return meta;
  } else if (!modelId.startsWith("/") && !modelId.startsWith("./") && !modelId.startsWith("../")) {
    // Bare name — try default org first
    const repoId = resolveRepoId(modelId);
    if (repoId) {
      const meta = await fetchHfMeta(repoId);
      if (meta) return meta;
    }
    // Default org didn't match — search HF for the correct org
    const found = await searchHfModel(modelId);
    if (found) {
      const meta = await fetchHfMeta(found);
      if (meta) return meta;
    }
  }
  // 2) Local failover: offline, gated/private, or local-path model ids.
  //    Org-agnostic search — scans all owners in mlxcel store and HF hub cache.
  const dir = localModelDir(modelId);
  if (dir) {
    const cfg = readJsonFile(join(dir, "config.json"));
    if (cfg && typeof cfg === "object") {
      const tokCfg = readJsonFile(join(dir, "tokenizer_config.json"));
      const genCfg = readJsonFile(join(dir, "generation_config.json"));
      const template = extractTemplate(tokCfg, dir);
      return { cfg, tokCfg, genCfg, template, source: "local" };
    }
  }
  return null;
}

// --- detection -------------------------------------------------------------

// Lex the Jinja chat template and collect identifier-like tokens. HF's own
// @huggingface/jinja tokenizer is more robust than regex: it distinguishes
// real identifiers from string literals/comments, and exposes template
// variables (enable_thinking, tools, tool_call, image, ...). Returns null if
// the lexer cannot handle the template (caller falls back to regex).
function templateIdentifiers(template: string): Set<string> | null {
  if (!template) return null;
  try {
    const ids = new Set<string>();
    for (const tk of tokenize(template)) {
      const v = tk?.value;
      if (typeof v === "string" && /^[A-Za-z_][A-Za-z0-9_]*$/.test(v)) ids.add(v);
    }
    return ids;
  } catch {
    return null;
  }
}

function extractContext(cfg: Cfg): number | undefined {
  if (!cfg) return undefined;
  for (const k of CTX_KEYS) if (Number.isFinite(cfg[k])) return cfg[k];
  const tc = cfg.text_config;
  if (tc && typeof tc === "object") {
    for (const k of CTX_KEYS) if (Number.isFinite(tc[k])) return tc[k];
  }
  return undefined;
}

const VISION_TOK_KEYS = ["image_token", "video_token", "boi_token", "eoi_token", "vision_bos_token", "vision_eos_token"];
const VISION_TEMPLATE_IDS = ["image", "image_url", "image_count", "video", "video_count", "vision", "add_vision_id", "do_vision_count"];

function isVision(cfg: Cfg, tokCfg: Cfg, ids: Set<string> | null): boolean {
  if (cfg) {
    const v = cfg.vision_config;
    if (v && typeof v === "object" && Object.keys(v).length > 0) return true;
  }
  if (tokCfg) {
    for (const k of VISION_TOK_KEYS) if (tokCfg[k] != null) return true;
  }
  if (ids) {
    for (const k of VISION_TEMPLATE_IDS) if (ids.has(k)) return true;
  }
  return false;
}

const REASONING_IDS = ["enable_thinking", "reasoning_content", "clear_thinking", "think", "reasoning"];

function detectReasoning(ids: Set<string> | null, template: string): boolean {
  if (env("MLXCEL_AUTO_NO_REASONING", "") === "1") return false;
  if (ids) {
    for (const k of REASONING_IDS) if (ids.has(k)) return true;
    return false;
  }
  return /enable_thinking|reasoning_content|clear_thinking|<think|<\/think/.test(template || "");
}

function detectTools(tokCfg: Cfg, ids: Set<string> | null, template: string): boolean {
  if (tokCfg && tokCfg.tool_parser_type) return true;
  if (ids) {
    if (ids.has("tools")) {
      for (const k of ["tool_call", "tool_calls", "tool", "function_call", "function"]) {
        if (ids.has(k)) return true;
      }
    }
    return false;
  }
  return /tool_call|function_call/.test(template || "") && /tools/.test(template || "");
}

// --- cache ------------------------------------------------------------------

function loadCache(): Cache {
  if (env("MLXCEL_AUTO_NO_CACHE", "") === "1") return {};
  try {
    if (existsSync(CACHE_PATH)) return JSON.parse(readFileSync(CACHE_PATH, "utf8")) as Cache;
  } catch {}
  return {};
}

function saveCache(c: Cache) {
  if (env("MLXCEL_AUTO_NO_CACHE", "") === "1") return;
  try {
    writeFileSync(CACHE_PATH, JSON.stringify(c, null, 2));
  } catch {}
}

// --- discovery + registration ----------------------------------------------

interface ServerUrl {
  origin: string;   // e.g. "http://127.0.0.1:8080"
  baseUrl: string;  // e.g. "http://127.0.0.1:8080/v1"
  host: string;
  port: number;
}

function parseBaseUrls(raw: string): ServerUrl[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      // tolerate missing scheme
      const url = s.includes("://") ? s : `http://${s}`;
      const u = new URL(url.endsWith("/") ? url : url + "/");
      const origin = u.origin;
      const baseUrl = `${origin}/v1`;
      const host = u.hostname;
      const port = u.port ? Number(u.port) : (u.protocol === "https:" ? 443 : 80);
      return { origin, baseUrl, host, port };
    });
}

function providerIdFrom(url: ServerUrl): string {
  const isDefault =
    (url.origin === "http://127.0.0.1:8080" || url.origin === "http://localhost:8080");
  if (isDefault) return "mlxcel-auto";
  return `mlxcel-auto-${url.host}-${url.port}`;
}

async function probeServer(origin: string): Promise<any[] | null> {
  const data = await fetchJson(`${origin}/v1/models`, PROBE_TIMEOUT_MS);
  if (!data || !Array.isArray(data.data)) return null;
  return data.data as any[];
}

async function resolveModel(modelId: string, cache: Cache): Promise<CacheEntry> {
  const cached = cache[modelId];
  if (cached) {
    // Migrate pre-0.3.0 entries that used `contextWindow`.
    if (cached.modelMaxCtx == null && (cached as any).contextWindow != null) {
      cached.modelMaxCtx = (cached as any).contextWindow;
    }
    return cached;
  }

  const meta = await loadModelMeta(modelId);
  const fbCtx = envInt("MLXCEL_AUTO_FALLBACK_CTX", 32768);
  if (!meta) {
    const entry: CacheEntry = {
      modelMaxCtx: fbCtx, vision: false, reasoning: false, tools: false,
      source: "fallback", ts: Date.now(),
    };
    cache[modelId] = entry;
    return entry;
  }
  const ids = templateIdentifiers(meta.template);
  const ctx =
    extractContext(meta.cfg) ??
    (meta.tokCfg && Number.isFinite(meta.tokCfg.model_max_length) ? meta.tokCfg.model_max_length : undefined) ??
    fbCtx;
  const modelType = meta.cfg?.model_type ?? meta.cfg?.text_config?.model_type;
  const architectures = Array.isArray(meta.cfg?.architectures) ? meta.cfg.architectures : undefined;
  const eosToken = meta.cfg?.eos_token_id ?? meta.tokCfg?.eos_token;
  const quantBits = meta.cfg?.quantization?.bits;
  const quantization = quantBits != null ? `${quantBits}-bit` : (meta.cfg?.quantization?.group_size != null ? "quantized" : undefined);
  const genMaxNewTokens = meta.genCfg?.max_new_tokens ?? meta.genCfg?.max_length;
  const entry: CacheEntry = {
    modelMaxCtx: ctx,
    vision: isVision(meta.cfg, meta.tokCfg, ids),
    reasoning: detectReasoning(ids, meta.template),
    tools: detectTools(meta.tokCfg, ids, meta.template),
    source: meta.source,
    ts: Date.now(),
    modelType,
    architectures,
    eosToken,
    quantization,
    genMaxNewTokens,
  };
  cache[modelId] = entry;
  return entry;
}

// Query the server's effective per-slot context window. mlxcel-server reports
// `context_size` on /health and /slots. With `--ctx-size 0` (model default) it
// reports 0, meaning unbounded up to model max, so the caller keeps the model
// max. With an explicit `--ctx-size C --parallel N`, it reports floor(C / N),
// the real per-slot budget, which should override.
async function fetchEffectiveCtx(origin: string): Promise<number> {
  const h = await fetchJson(`${origin}/health`, PROBE_TIMEOUT_MS);
  if (h && Number.isFinite(h.context_size) && h.context_size > 0) return h.context_size;
  const slots = await fetchJson(`${origin}/slots`, PROBE_TIMEOUT_MS);
  if (Array.isArray(slots)) {
    for (const s of slots) {
      const cs = s?.context_size;
      if (Number.isFinite(cs) && cs > 0) return cs;
    }
  }
  return 0;
}

async function discoverAndRegister(pi: ExtensionAPI) {
  const servers = parseBaseUrls(env("MLXCEL_AUTO_BASEURLS", "http://127.0.0.1:8080"));
  const apiKey = env("MLXCEL_AUTO_APIKEY", "not-needed");
  const maxOut = envInt("MLXCEL_AUTO_MAXOUT", 32768);

  const cache = loadCache();
  let registered = 0;
  const failures: string[] = [];

  for (const srv of servers) {
    const models = await probeServer(srv.origin);
    if (models === null) {
      failures.push(`${srv.origin} not reachable`);
      continue;
    }
    const providerId = providerIdFrom(srv);
    const effectiveCtx = await fetchEffectiveCtx(srv.origin);
    const regModels = [];
    for (const m of models) {
      const id: string = m.id ?? m.model ?? "";
      if (!id) continue;
      const meta = await resolveModel(id, cache);
      const ctx = effectiveCtx > 0 ? effectiveCtx : meta.modelMaxCtx;
      const maxTokens = Math.min(ctx, maxOut);
      const regId = resolveRepoId(id) ?? id;
      // Collect this model's stop tokens for the message_end trim hook.
      const stops = new Set<string>(STOP_TOKENS);
      if (typeof meta.eosToken === "string") stops.add(meta.eosToken);
      modelStopTokens.set(regId, [...stops]);
      const baseCompat = {
        supportsDeveloperRole: false as const,
        supportsReasoningEffort: false as const,
        supportsStore: false as const,
        supportsStrictMode: false as const,
        maxTokensField: "max_tokens" as const,
      };
      regModels.push({
        id: regId,
        name: id.split("/").pop() ?? id,
        reasoning: meta.reasoning,
        input: meta.vision ? (["text", "image"] as const) : (["text"] as const),
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: ctx,
        maxTokens,
        compat: meta.reasoning
          ? { ...baseCompat, thinkingFormat: "qwen-chat-template" as const }
          : baseCompat,
        // MLX treats enable_thinking as a boolean toggle, so all non-off pi
        // thinking levels are identical. Hide the intermediate levels and keep
        // only off (disable) + high (enable) for an honest selector.
        ...(meta.reasoning
          ? { thinkingLevelMap: { minimal: null, low: null, medium: null, xhigh: null } as const }
          : {}),
      });
      registered++;
    }
    if (regModels.length === 0) {
      failures.push(`${srv.origin} returned no models`);
      continue;
    }
    pi.registerProvider(providerId, {
      baseUrl: srv.baseUrl,
      apiKey,
      api: "openai-completions",
      models: regModels,
    });
  }

  saveCache(cache);
  return { registered, failures, cache };
}

// --- entry point ------------------------------------------------------------

export default async function (pi: ExtensionAPI) {
  await discoverAndRegister(pi);

  // Trim trailing leaked stop tokens from finalized assistant messages for our
  // models only. mlxcel-server normally stops at EOS, but some quantized/small
  // models emit the template's stop token at the end of the content.
  pi.on("message_end", async (event, ctx) => {
    const msg = event.message;
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) return;
    const provider = (msg as any).provider ?? ctx.model?.provider;
    if (!provider || !String(provider).startsWith("mlxcel-auto")) return;
    const stops = modelStopTokens.get((msg as any).model);
    if (!stops || stops.length === 0) return;
    let changed = false;
    const content = msg.content.map((block: any) => {
      if (block?.type !== "text" || typeof block.text !== "string") return block;
      let text = block.text;
      let prev: string;
      do {
        prev = text;
        text = text.replace(/\s+$/, "");
        for (const s of stops) {
          if (!s) continue;
          if (text.endsWith(s)) {
            text = text.slice(0, -s.length);
            break;
          }
        }
      } while (text !== prev);
      if (text === block.text) return block;
      changed = true;
      return { ...block, text };
    });
    return changed ? { message: { ...msg, content } } : undefined;
  });

  pi.on("session_start", async (_e, ctx) => {
    if (ctx.hasUI) {
      const r = await discoverAndRegister(pi);
      if (r.failures.length && r.registered === 0) {
        ctx.ui.notify(`mlxcel-auto: ${r.failures.join("; ")}`, "warn");
      } else if (r.registered > 0) {
        ctx.ui.notify(`mlxcel-auto: registered ${r.registered} model(s)`, "info");
      }
    } else {
      await discoverAndRegister(pi);
    }
  });

  pi.registerCommand("mlxcel-auto", {
    description: "Re-probe mlxcel-server and re-register discovered models",
    handler: async (_args, ctx) => {
      const r = await discoverAndRegister(pi);
      const lines = [
        `Registered: ${r.registered}`,
        r.failures.length ? `Failures: ${r.failures.join("; ")}` : "Failures: none",
        `Cached models: ${Object.keys(r.cache).length}`,
      ];
      ctx.ui.notify(lines.join(" | "), r.failures.length && r.registered === 0 ? "warn" : "info");
    },
  });

  pi.registerCommand("mlxcel-auto-info", {
    description: "Show detected metadata for cached mlxcel-auto models (optional substring filter)",
    handler: async (args, ctx) => {
      const cache = loadCache();
      const ids = Object.keys(cache);
      if (!ids.length) {
        ctx.ui.notify("mlxcel-auto: no cached models. Run /mlxcel-auto first.", "warn");
        return;
      }
      const arg = (args || "").trim();
      const lines: string[] = [];
      for (const id of ids) {
        if (arg && !id.includes(arg)) continue;
        const e = cache[id];
        lines.push(
          `${id}: ctx ${e.modelMaxCtx} | vision ${e.vision ? "yes" : "no"} | reasoning ${e.reasoning ? "yes" : "no"} | tools ${e.tools ? "yes" : "no"} | ${e.source}`,
        );
        const meta = [
          e.modelType ? `model_type=${e.modelType}` : null,
          e.architectures ? `arch=${e.architectures.join("|")}` : null,
          e.quantization ? `quant=${e.quantization}` : null,
          e.eosToken != null ? `eos=${e.eosToken}` : null,
          e.genMaxNewTokens != null ? `genMaxNewTokens=${e.genMaxNewTokens}` : null,
        ].filter(Boolean);
        if (meta.length) lines.push(`  ${meta.join("  ")}`);
      }
      if (!lines.length) {
        ctx.ui.notify(`mlxcel-auto: no cached model matching "${arg}"`, "warn");
        return;
      }
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}