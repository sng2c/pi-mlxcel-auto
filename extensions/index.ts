/**
 * mlxcel-auto
 *
 * Auto-discovers a running `mlxcel-server` (OpenAI-compatible) on localhost,
 * reads each model's real max context window from the local MLX store's
 * config.json (no manual ctx-size entry), and registers it with pi.
 *
 * Local-first: reads `<store>/<owner>/<name>/config.json` from the mlxcel
 * model store. Falls back to fetching Hugging Face config.json if the model
 * is not in the local store. Results are cached to avoid repeated work.
 *
 * Beyond context, it also detects:
 *   - reasoning/thinking: from chat_template tokens (enable_thinking /
 *     reasoning_content / <think>...</think>). Sets `reasoning: true` and
 *     `compat.thinkingFormat: "qwen-chat-template"` so pi drives MLX's
 *     `chat_template_kwargs.enable_thinking` (off disables, others enable).
 *   - vision: from `vision_config` or tokenizer image/video tokens. Sets
 *     `input: ["text","image"]`.
 *   - tools: from chat_template tool_call markers or `tool_parser_type`.
 *     Informational only (pi has no per-model tool toggle); cached as metadata.
 *
 * Config (env vars):
 *   MLXCEL_AUTO_PORTS  comma-separated ports to probe (default: "8080")
 *   MLXCEL_AUTO_HOST   host to probe (default: "127.0.0.1")
 *   MLXCEL_AUTO_APIKEY api key sent to the server (default: "not-needed")
 *   MLXCEL_AUTO_MAXOUT cap on maxTokens (default: 32768)
 *   MLXCEL_AUTO_FALLBACK_CTX  context window used when detection fails (default: 32768)
 *   MLXCEL_AUTO_NO_REASONING  "1" disables automatic reasoning detection
 *   MLXCEL_AUTO_NO_CACHE  "1" disables the on-disk config cache
 *
 * No changes to pi core, models.json, or other extensions. Registers a
 * provider per port as "mlxcel-auto" (port 8080) or "mlxcel-auto-<port>".
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
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

interface CacheEntry {
  contextWindow: number;
  vision: boolean;
  reasoning: boolean;
  tools: boolean;
  source: "local" | "hf" | "fallback";
  ts: number;
}

interface ModelMeta {
  cfg: Cfg;
  tokCfg: Cfg;        // tokenizer_config.json (may be null)
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
// mlxcel resolves a bare name (no slash) as `${MLXCEL_DEFAULT_ORG}/<name>`.
// Returns null for local filesystem paths and opaque aliases that cannot be
// mapped to a Hugging Face repo id.
function resolveRepoId(modelId: string): string | null {
  if (!modelId) return null;
  if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")) {
    return null;
  }
  if (modelId.includes("/")) return modelId; // owner/name already
  const org = env("MLXCEL_DEFAULT_ORG", "mlx-community");
  return `${org}/${modelId}`;
}

function localModelDir(modelId: string): string | null {
  // Local directory path passed as -m?
  if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")) {
    const maybeDir = modelId.startsWith("/") ? modelId : join(process.cwd(), modelId);
    if (existsSync(join(maybeDir, "config.json"))) return maybeDir;
  }
  const repoId = resolveRepoId(modelId);
  if (!repoId) return null;
  const parts = repoId.split("/");
  const owner = parts[0];
  const name = parts.slice(1).join("/");
  const dirs: string[] = [];
  const md = process.env.MLXCEL_MODELS_DIR;
  if (md) dirs.push(join(md, owner, name));
  const cacheDir = process.env.MLXCEL_CACHE_DIR || `${homedir()}/.cache/mlxcel`;
  dirs.push(join(cacheDir, "models", owner, name));
  for (const d of dirs) {
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

async function loadModelMeta(modelId: string): Promise<ModelMeta | null> {
  // 1) local store
  const dir = localModelDir(modelId);
  if (dir) {
    const cfg = readJsonFile(join(dir, "config.json"));
    if (cfg && typeof cfg === "object") {
      const tokCfg = readJsonFile(join(dir, "tokenizer_config.json"));
      const template = extractTemplate(tokCfg, dir);
      return { cfg, tokCfg, template, source: "local" };
    }
  }
  // 2) Hugging Face config + tokenizer_config + chat_template.jinja
  const repoId = resolveRepoId(modelId);
  if (!repoId) return null;
  const cfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/config.json`, HTTP_TIMEOUT_MS);
  if (!cfg || typeof cfg !== "object") return null;
  const tokCfg = await fetchJson(`https://huggingface.co/${repoId}/raw/main/tokenizer_config.json`, HTTP_TIMEOUT_MS);
  let template = extractTemplate(tokCfg, null);
  if (!template) {
    const jinja = await fetchText(`https://huggingface.co/${repoId}/raw/main/chat_template.jinja`, HTTP_TIMEOUT_MS);
    if (jinja) template = jinja;
  }
  return { cfg, tokCfg, template, source: "hf" };
}

// --- detection -------------------------------------------------------------

function extractContext(cfg: Cfg): number | undefined {
  if (!cfg) return undefined;
  for (const k of CTX_KEYS) if (Number.isFinite(cfg[k])) return cfg[k];
  const tc = cfg.text_config;
  if (tc && typeof tc === "object") {
    for (const k of CTX_KEYS) if (Number.isFinite(tc[k])) return tc[k];
  }
  return undefined;
}

function isVision(cfg: Cfg, tokCfg: Cfg): boolean {
  if (cfg) {
    const v = cfg.vision_config;
    if (v && typeof v === "object" && Object.keys(v).length > 0) return true;
  }
  if (tokCfg) {
    const keys = ["image_token", "video_token", "boi_token", "eoi_token", "vision_bos_token", "vision_eos_token"];
    for (const k of keys) if (tokCfg[k] != null) return true;
  }
  return false;
}

function detectReasoning(_cfg: Cfg, _tokCfg: Cfg, template: string): boolean {
  if (env("MLXCEL_AUTO_NO_REASONING", "") === "1") return false;
  return /enable_thinking|reasoning_content|<think|<\/think/.test(template || "");
}

function detectTools(_cfg: Cfg, tokCfg: Cfg, template: string): boolean {
  if (tokCfg && tokCfg.tool_parser_type) return true;
  const t = template || "";
  return /tool_call|function_call/.test(t) && /tools/.test(t);
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

async function probePort(host: string, port: number): Promise<any[] | null> {
  const data = await fetchJson(`http://${host}:${port}/v1/models`, PROBE_TIMEOUT_MS);
  if (!data || !Array.isArray(data.data)) return null;
  return data.data as any[];
}

async function resolveModel(modelId: string, cache: Cache): Promise<CacheEntry> {
  const cached = cache[modelId];
  if (cached) return cached;

  const meta = await loadModelMeta(modelId);
  const fbCtx = envInt("MLXCEL_AUTO_FALLBACK_CTX", 32768);
  if (!meta) {
    const entry: CacheEntry = {
      contextWindow: fbCtx, vision: false, reasoning: false, tools: false,
      source: "fallback", ts: Date.now(),
    };
    cache[modelId] = entry;
    return entry;
  }
  const entry: CacheEntry = {
    contextWindow: extractContext(meta.cfg) ?? fbCtx,
    vision: isVision(meta.cfg, meta.tokCfg),
    reasoning: detectReasoning(meta.cfg, meta.tokCfg, meta.template),
    tools: detectTools(meta.cfg, meta.tokCfg, meta.template),
    source: meta.source,
    ts: Date.now(),
  };
  cache[modelId] = entry;
  return entry;
}

async function discoverAndRegister(pi: ExtensionAPI) {
  const host = env("MLXCEL_AUTO_HOST", "127.0.0.1");
  const ports = env("MLXCEL_AUTO_PORTS", "8080")
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  const apiKey = env("MLXCEL_AUTO_APIKEY", "not-needed");
  const maxOut = envInt("MLXCEL_AUTO_MAXOUT", 32768);

  const cache = loadCache();
  let registered = 0;
  const failures: string[] = [];

  for (const port of ports) {
    const models = await probePort(host, port);
    if (models === null) {
      failures.push(`${host}:${port} not reachable`);
      continue;
    }
    const providerId = port === 8080 ? "mlxcel-auto" : `mlxcel-auto-${port}`;
    const regModels = [];
    for (const m of models) {
      const id: string = m.id ?? m.model ?? "";
      if (!id) continue;
      const meta = await resolveModel(id, cache);
      const ctx = meta.contextWindow;
      const maxTokens = Math.min(ctx, maxOut);
      regModels.push({
        id,
        name: id.split("/").pop() ?? id,
        reasoning: meta.reasoning,
        input: meta.vision ? (["text", "image"] as const) : (["text"] as const),
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: ctx,
        maxTokens,
        ...(meta.reasoning ? { compat: { thinkingFormat: "qwen-chat-template" as const } } : {}),
      });
      registered++;
    }
    if (regModels.length === 0) {
      failures.push(`${host}:${port} returned no models`);
      continue;
    }
    pi.registerProvider(providerId, {
      baseUrl: `http://${host}:${port}/v1`,
      apiKey,
      api: "openai-completions",
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        maxTokensField: "max_tokens",
      },
      models: regModels,
    });
  }

  saveCache(cache);
  return { registered, failures, cache };
}

// --- entry point ------------------------------------------------------------

export default async function (pi: ExtensionAPI) {
  await discoverAndRegister(pi);

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
}