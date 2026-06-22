# pi-mlxcel-auto

A [pi](https://pi.dev) extension that auto-discovers a running **`mlxcel-server`** (the Rust MLX runner from [lablup/mlxcel](https://github.com/lablup/mlxcel), OpenAI-compatible) and registers its served models with pi — **reading the real context window from the model's metadata so you never type `ctx-size` manually.**

Supports both `http` and `https` endpoints, local and remote servers.

## What it does

1. Probes each configured server URL (`/v1/models`), default `http://127.0.0.1:8080`.
2. For each model id:
   - Fetches metadata from Hugging Face first (`config.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`). Falls back to the local mlxcel model store if HF is unreachable, the model is gated/private, or the id is a local path. Results are cached.
   - Bare model names (no slash) are resolved as `${MLXCEL_DEFAULT_ORG}/<name>` (default `mlx-community`), matching mlxcel's own resolver. If the org guess is wrong, HF returns 404 and local fallback applies — never fatal.
   - Extracts the context window from `max_position_embeddings` (top-level, then `text_config.*`).
   - Detects vision from `vision_config` or tokenizer image/video tokens.
   - Detects reasoning from chat template tokens (`enable_thinking`, `reasoning_content`, `think`, etc.).
   - Detects tools from chat template tool-call markers or `tool_parser_type` (informational only).
3. Registers a provider `mlxcel-auto` (default URL) or `mlxcel-auto-<host>-<port>` via `pi.registerProvider()` with `openai-completions` and safe `compat` flags.
4. Caches results under `~/.pi/agent/extensions-data/mlxcel-auto-cache.json`.
5. Re-probes on `session_start` so a freshly started server is picked up automatically.

## Install

```bash
pi install git:github.com/sng2c/pi-mlxcel-auto
```

Then in pi:

```text
/reload
/model
```

## Usage

### Local server (default)

```bash
mlxcel serve --model mlx-community/gemma-3-4b-it-qat-4bit
# or: mlxcel-server -m mlx-community/Qwen3.5-0.8B-4bit --port 8080
```

### Remote server (HTTPS)

```bash
MLXCEL_AUTO_BASEURLS=https://ml.my-server.com:8443
```

### Multiple servers

```bash
MLXCEL_AUTO_BASEURLS=http://127.0.0.1:8080,https://ml.my-server.com:8443
```

Run pi (or `/reload` if already running). Re-probe manually with `/mlxcel-auto`.

Select `mlxcel-auto/<model>` in `/model`. The context window is auto-detected — no manual entry.

## Configuration (env vars)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLXCEL_AUTO_BASEURLS` | `http://127.0.0.1:8080` | Comma-separated server base URLs. Supports `http` and `https`. Scheme is optional (defaults to `http`). |
| `MLXCEL_AUTO_APIKEY` | `not-needed` | API key sent to the server |
| `MLXCEL_AUTO_MAXOUT` | `32768` | Cap on `maxTokens` |
| `MLXCEL_AUTO_FALLBACK_CTX` | `32768` | Context window used when detection fails |
| `MLXCEL_AUTO_NO_REASONING` | (off) | `1` disables automatic reasoning/thinking detection |
| `MLXCEL_AUTO_NO_CACHE` | (off) | `1` disables the on-disk config cache |
| `MLXCEL_DEFAULT_ORG` | `mlx-community` | Org used to resolve bare model names when the server returns a name without an owner prefix |
| `MLXCEL_MODELS_DIR` | (unset) | Override mlxcel model-store root |
| `MLXCEL_CACHE_DIR` | `~/.cache/mlxcel` | Override mlxcel cache root |

## Notes / limitations

- **reasoning**: detected from chat template tokens (`enable_thinking` / `reasoning_content` / `clear_thinking` / `think`). When detected, `reasoning: true`, `compat.thinkingFormat: "qwen-chat-template"`, and `thinkingLevelMap` hides intermediate levels (`minimal`/`low`/`medium`/`xhigh` → `null`) keeping only `off` (disable) + `high` (enable), since MLX treats `enable_thinking` as a boolean toggle. Set `MLXCEL_AUTO_NO_REASONING=1` to opt out.
- **context window**: resolved from the server's effective per-slot `context_size` (`/health` or `/slots`) when `--ctx-size` is set explicitly; otherwise falls back to the model's max from `config.json` (`max_position_embeddings`). This prevents pi from assuming a larger window than the server actually allocated.
- **vision**: detected from `vision_config` or tokenizer image/video tokens. Maps to `input: ["text","image"]`. Video/audio are not expressible in pi's model `input` field (text/image only).
- **tools**: detected from chat template tool-call markers or `tool_parser_type`. Informational only — pi has no per-model tool toggle, so it is cached as metadata and not reflected in registration.
- **stop-token trim**: a `message_end` hook strips trailing leaked stop tokens (e.g. Qwen `<|im_end|>`, Gemma `<end_of_turn>`, GLM ``) from finalized assistant messages for `mlxcel-auto` models only. Per-model string `eos_token` from config/tokenizer_config is also used when available.
- **metadata source**: remote-first. The extension fetches `config.json` / `tokenizer_config.json` / `generation_config.json` / `chat_template.jinja` from Hugging Face first, then falls back to the local mlxcel store / model dir if HF is unreachable (offline), the model is gated/private, or the id is a local path. Results are cached. This makes remote mlxcel servers work with no extra config.
- **metadata cached**: `model_type`, `architectures`, `eos_token`, `quantization`, `gen_max_new_tokens` (from `generation_config.json`). Audio/video modalities are not yet split out (pi `input` only accepts text/image); see the TODO in the source for future wiring.
- **default org**: `mlxcel-server` reports the bare snapshot directory name in `/v1/models` even when launched with a full `owner/name` repo id. Bare names are resolved as `MLXCEL_DEFAULT_ORG/<name>` (default `mlx-community`). If the org guess is wrong, HF returns 404 and local fallback applies — never fatal. Override with `MLXCEL_DEFAULT_ORG`.
- `reasoning` defaults to `false` when not detected. Add a `mlxcel` provider in `~/.pi/agent/models.json` to override per model — this extension uses the `mlxcel-auto` provider id so there is no collision.
- If you launch with `--alias <custom>` the id is not a repo id and cannot be resolved; fall back to a manual `models.json` entry in that case.
- **HTTPS**: set `MLXCEL_AUTO_BASEURLS=https://...` to connect to a remote or TLS-terminated mlxcel-server. Both `http` and `https` are fully supported.

## Cache

`~/.pi/agent/extensions-data/mlxcel-auto-cache.json` maps each model id (as returned by `/v1/models`) to:

| field | description |
| --- | --- |
| `modelMaxCtx` | model max context from `config.json` (`max_position_embeddings`, top-level or `text_config.*`), else `tokenizer_config.model_max_length`, else `MLXCEL_AUTO_FALLBACK_CTX` |
| `vision` | from `vision_config`, tokenizer image/video tokens, or chat-template ids |
| `reasoning` | from chat-template ids (`enable_thinking` / `reasoning_content` / `clear_thinking` / `think`); disabled by `MLXCEL_AUTO_NO_REASONING=1` |
| `tools` | from `tool_parser_type` or chat-template `tools`+`tool_call*` ids; informational only |
| `source` | `local` (mlxcel store), `hf` (Hugging Face fetch), or `fallback` (detection failed) |
| `ts` | detection timestamp (ms) |
| `modelType` | `config.model_type` (or `text_config.model_type`) |
| `architectures` | `config.architectures` array |
| `eosToken` | `config.eos_token_id` or `tokenizer_config.eos_token` (string or id array) |
| `quantization` | e.g. `4-bit` from `config.quantization.bits` |
| `genMaxNewTokens` | `generation_config.max_new_tokens` / `max_length` when present |

The registered `contextWindow` is the server's effective per-slot `context_size` (`/health` or `/slots`) when `--ctx-size` is set explicitly, otherwise `modelMaxCtx`.

Inspect from pi with `/mlxcel-auto-info [substring]`.

## License

MIT