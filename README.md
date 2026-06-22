# pi-mlxcel-auto

A [pi](https://pi.dev) extension that auto-discovers a running **`mlxcel-server`** (the Rust MLX runner from [lablup/mlxcel](https://github.com/lablup/mlxcel), OpenAI-compatible) and registers its served models with pi — **reading the real context window from Hugging Face metadata so you never type `ctx-size` manually.**

Supports both `http` and `https` endpoints, local and remote servers. Works with any OpenAI-compatible MLX server (mlxcel, mlx-lm, etc.).

## What it does

1. Probes each configured server URL (`/v1/models`), default `http://127.0.0.1:8080`.
2. For each model id:
   - Fetches metadata from Hugging Face (`config.json`, `tokenizer_config.json`, `chat_template.jinja`). Results are cached permanently — model metadata doesn't change.
   - Bare model names (no slash) are resolved by trying `MLXCEL_DEFAULT_ORG/<name>` first (default `mlx-community`), then the HF search API if that 404s. This handles models from any org automatically.
   - Extracts the context window from `max_position_embeddings` (top-level, then `text_config.*`).
   - Detects vision from `vision_config` or tokenizer image/video tokens.
   - Detects reasoning from chat template tokens (`enable_thinking`, `reasoning_content`, `think`, etc.).
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
| `MLXCEL_AUTO_MAXOUT` | `32768` | Cap on `maxTokens`; also used as fallback context window when detection fails |
| `MLXCEL_AUTO_NO_REASONING` | (off) | `1` disables automatic reasoning/thinking detection |
| `MLXCEL_AUTO_NO_CACHE` | (off) | `1` disables the on-disk config cache |
| `MLXCEL_DEFAULT_ORG` | `mlx-community` | Org tried first for bare model names. Falls back to HF search API on 404 |

## Notes / limitations

- **reasoning**: detected from chat template tokens (`enable_thinking` / `reasoning_content` / `clear_thinking` / `think`). When detected, `reasoning: true`, `compat.thinkingFormat: "qwen-chat-template"`, and `thinkingLevelMap` hides intermediate levels (`minimal`/`low`/`medium`/`xhigh` → `null`) keeping only `off` (disable) + `high` (enable), since MLX treats `enable_thinking` as a boolean toggle. Set `MLXCEL_AUTO_NO_REASONING=1` to opt out.
- **context window**: resolved from the server's effective per-slot `context_size` (`/health` or `/slots`) when `--ctx-size` is set explicitly; otherwise falls back to the model's max from `config.json` (`max_position_embeddings`). This prevents pi from assuming a larger window than the server actually allocated.
- **vision**: detected from `vision_config` or tokenizer image/video tokens. Maps to `input: ["text","image"]`. Video/audio are not expressible in pi's model `input` field (text/image only).
- **stop-token trim**: a `message_end` hook strips trailing leaked stop tokens (e.g. Qwen `<|im_end|>`, Gemma `<end_of_turn>`, GLM ``) from finalized assistant messages for `mlxcel-auto` models only. Per-model string `eos_token` from config/tokenizer_config is also used when available.
- **metadata source**: remote-only (Hugging Face). Bare names are resolved by trying `MLXCEL_DEFAULT_ORG/<name>` first, then the HF search API. If both fail, the fallback context window (`MLXCEL_AUTO_MAXOUT`) is used. No local file reading — works identically for mlxcel, mlx-lm, or any OpenAI-compatible server.
- **default org**: `mlxcel-server` reports the bare snapshot directory name in `/v1/models` even when launched with a full `owner/name` repo id. Bare names are resolved by trying `MLXCEL_DEFAULT_ORG/<name>` first (default `mlx-community`), then the Hugging Face search API if that 404s. This handles models from any org automatically. Override the first-try org with `MLXCEL_DEFAULT_ORG`.
- `reasoning` defaults to `false` when not detected. Add a `mlxcel` provider in `~/.pi/agent/models.json` to override per model — this extension uses the `mlxcel-auto` provider id so there is no collision.
- If you launch with `--alias <custom>` the id is not a repo id and cannot be resolved; fall back to a manual `models.json` entry in that case.
- **HTTPS**: set `MLXCEL_AUTO_BASEURLS=https://...` to connect to a remote or TLS-terminated mlxcel-server. Both `http` and `https` are fully supported.

## Cache

`~/.pi/agent/extensions-data/mlxcel-auto-cache.json` maps each model id (as returned by `/v1/models`) to:

| field | description |
| --- | --- |
| `modelMaxCtx` | model max context from `config.json` (`max_position_embeddings`, top-level or `text_config.*`), else `tokenizer_config.model_max_length`, else `MLXCEL_AUTO_MAXOUT` |
| `vision` | from `vision_config`, tokenizer image/video tokens, or chat-template ids |
| `reasoning` | from chat-template ids (`enable_thinking` / `reasoning_content` / `clear_thinking` / `think`); disabled by `MLXCEL_AUTO_NO_REASONING=1` |
| `eosToken` | `config.eos_token_id` or `tokenizer_config.eos_token` (string or id array); used for stop-token trim |

The registered `contextWindow` is the server's effective per-slot `context_size` (`/health` or `/slots`) when `--ctx-size` is set explicitly, otherwise `modelMaxCtx`.

Inspect from pi with `/mlxcel-auto-info [substring]`.

## License

MIT