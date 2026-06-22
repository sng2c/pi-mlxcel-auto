/**
 * mlxcel-auto
 *
 * Auto-discovers a running `mlxcel-server` (OpenAI-compatible) and registers
 * its served models with pi, reading each model's real context window from
 * Hugging Face metadata so you never type ctx-size manually. Supports http
 * and https, local and remote servers. Works with any OpenAI-compatible MLX
 * server (mlxcel, mlx-lm, etc.).
 *
 * Metadata is fetched from Hugging Face (remote-only). Bare model names are
 * resolved by trying MLXCEL_DEFAULT_ORG first, then the HF search API.
 * Results are cached permanently — model metadata doesn't change.
 *
 * Beyond context, also detects:
 *   - reasoning/thinking: from chat_template tokens (enable_thinking /
 *     reasoning_content / clear_thinking / think). Sets `reasoning: true` and
 *     `compat.thinkingFormat: "qwen-chat-template"` so pi drives MLX's
 *     `chat_template_kwargs.enable_thinking` (off disables, others enable).
 *   - vision: from `vision_config` or tokenizer image/video tokens. Sets
 *     `input: ["text","image"]`.
 *
 * Config (env vars):
 *   MLXCEL_AUTO_BASEURLS       comma-separated server base URLs (default: "http://127.0.0.1:8080")
 *                              e.g. "http://127.0.0.1:8080,https://remote.example.com:8443"
 *                              Scheme is optional (defaults to http).
 *   MLXCEL_AUTO_APIKEY         api key sent to the server (default: "not-needed")
 *   MLXCEL_AUTO_MAXOUT         cap on maxTokens, also used as fallback context (default: 32768)
 *   MLXCEL_AUTO_NO_REASONING   "1" disables automatic reasoning detection
 *   MLXCEL_AUTO_NO_CACHE       "1" disables the on-disk config cache
 *   MLXCEL_DEFAULT_ORG         org for bare model names (default: "mlx-community")
 *
 * Registers a provider per base URL; the provider id is "mlxcel-auto" for the
 * default (http on 127.0.0.1:8080) or "mlxcel-auto-{host}-{port}" otherwise.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { tokenize } from "@huggingface/jinja";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
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
  eosToken?: string | number;
}

interface ModelMeta {
  cfg: Cfg;
  tokCfg: Cfg;        // tokenizer_config.json (may be null)
  template: string;   // chat template text (jinja file or embedded string)
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

// --- repo id resolution -----------------------------------------------------

// Resolve a model id (as returned by /v1/models) to an `owner/name` repo id.
// - Full repo id (contains "/"): returned as-is.
// - Bare name: resolved as MLXCEL_DEFAULT_ORG/<name> (default "mlx-community").
//   If that HF URL returns 404, the caller uses fallback context — no search.
function resolveRepoId(modelId: string): string | null {
  if (!modelId) return null;
  if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")) {
    return null;
  }
  if (modelId.includes("/")) return modelId; // owner/name already
  const org = env("MLXCEL_DEFAULT_ORG", "mlx-community");
  return `${org}/${modelId}`;
}

// Extract the chat template text from tokenizer_config.json (string or
// list-of-templates) or from a sibling chat_template.jinja file.
function extractTemplate(tokCfg: Cfg, repoId: string | null): string {
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
  if (repoId) {
    // chat_template.jinja is a standalone file on some repos
    // (fetched synchronously here since it's only reached when the embedded
    // template is missing; caller handles the await)
  }
  return "";
}

async function fetchHfMeta(repoId: string): Promise<ModelMeta | null> {
  const cfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/config.json`, HTTP_TIMEOUT_MS);
  if (!cfg || typeof cfg !== "object") return null;
  const tokCfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/tokenizer_config.json`, HTTP_TIMEOUT_MS);
  let template = extractTemplate(tokCfg, repoId);
  if (!template) {
    const jinja = await fetchText(`https://huggingface.co/${repoId}/raw/main/chat_template.jinja`, HTTP_TIMEOUT_MS);
    if (jinja) template = jinja;
  }
  return { cfg, tokCfg, template };
}

// Load model metadata from Hugging Face (remote-only).
// - Full repo id (owner/name): try directly.
// - Bare name: try MLXCEL_DEFAULT_ORG/name. 404 → fallback (no search).
async function loadModelMeta(modelId: string): Promise<ModelMeta | null> {
  if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")) {
    return null; // local path — no HF metadata
  }
  const repoId = resolveRepoId(modelId);
  if (!repoId) return null;
  return await fetchHfMeta(repoId);
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
  const fallbackCtx = envInt("MLXCEL_AUTO_MAXOUT", 32768);
  if (!meta) {
    const entry: CacheEntry = {
      modelMaxCtx: fallbackCtx, vision: false, reasoning: false,
    };
    cache[modelId] = entry;
    return entry;
  }
  const ids = templateIdentifiers(meta.template);
  const ctx =
    extractContext(meta.cfg) ??
    (meta.tokCfg && Number.isFinite(meta.tokCfg.model_max_length) ? meta.tokCfg.model_max_length : undefined) ??
    fallbackCtx;
  const eosToken = meta.cfg?.eos_token_id ?? meta.tokCfg?.eos_token;
  const entry: CacheEntry = {
    modelMaxCtx: ctx,
    vision: isVision(meta.cfg, meta.tokCfg, ids),
    reasoning: detectReasoning(ids, meta.template),
    eosToken,
  };
  cache[modelId] = entry;
  return entry;
}

// Query the server's effective per-slot context window. mlxcel-server reports
// `context_size` on /health and /slots. With `--ctx-size 0` (model default) it
// reports 0, meaning unbounded up to model max, so the caller keeps the model
// max. With an explicit `--ctx-size C --parallel N`, it reports floor(C / N),
// the real per-slot budget, which should override. Other servers (e.g. mlx-lm)
// don't provide these endpoints — they return 0 and the model max is used.
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
          `${id}: ctx ${e.modelMaxCtx} | vision ${e.vision ? "yes" : "no"} | reasoning ${e.reasoning ? "yes" : "no"}`,
        );
        if (typeof e.eosToken === "string") lines.push(`  eos=${e.eosToken}`);
      }
      if (!lines.length) {
        ctx.ui.notify(`mlxcel-auto: no cached model matching "${arg}"`, "warn");
        return;
      }
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}