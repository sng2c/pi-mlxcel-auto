# pi-mlxcel-auto

A [pi](https://pi.dev) extension that auto-discovers a running **`mlxcel-server`** (the Rust MLX runner from [lablup/mlxcel](https://github.com/lablup/mlxcel), OpenAI-compatible) on localhost and registers its served model with pi — **reading the real context window from the model's local `config.json` so you never type `ctx-size` manually.**

## What it does

1. Probes `http://<host>:<port>/v1/models` on configured ports (default `8080`).
2. For each model id:
   - Reads `<mlxcel store>/<owner>/<name>/config.json` locally first (`MLXCEL_MODELS_DIR` → `~/.cache/mlxcel/models`). Bare names are resolved as `${MLXCEL_DEFAULT_ORG}/<name>` (default `mlx-community`), matching mlxcel's own resolver.
   - Falls back to fetching `https://huggingface.co/<repoId>/raw/main/config.json` if the snapshot is not local.
   - Extracts the context window from `max_position_embeddings` (top-level, then `text_config.*`).
   - Detects vision from a non-empty `vision_config`.
3. Registers a provider `mlxcel-auto` (port 8080) or `mlxcel-auto-<port>` via `pi.registerProvider()` with `openai-completions` and safe `compat` flags.
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

1. Start the server:

   ```bash
   mlxcel serve --model mlx-community/gemma-3-4b-it-qat-4bit
   # or: mlxcel-server -m mlx-community/Qwen3.5-0.8B-4bit --port 8080
   ```

2. Run pi (or `/reload` if already running). Re-probe manually with `/mlxcel-auto`.

3. Select `mlxcel-auto/<model>` in `/model`. The context window is auto-detected — no manual entry.

## Configuration (env vars)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLXCEL_AUTO_PORTS` | `8080` | Comma-separated ports to probe, e.g. `8080,8081` |
| `MLXCEL_AUTO_HOST` | `127.0.0.1` | Host to probe |
| `MLXCEL_AUTO_APIKEY` | `not-needed` | API key sent to the server |
| `MLXCEL_AUTO_MAXOUT` | `32768` | Cap on `maxTokens` |
| `MLXCEL_AUTO_FALLBACK_CTX` | `32768` | Context window used when detection fails |
| `MLXCEL_AUTO_NO_REASONING` | (off) | `1` disables automatic reasoning/thinking detection |
| `MLXCEL_AUTO_NO_CACHE` | (off) | `1` disables the on-disk config cache |
| `MLXCEL_DEFAULT_ORG` | `mlx-community` | Org used to resolve bare model names |
| `MLXCEL_MODELS_DIR` | (unset) | Override mlxcel model-store root |
| `MLXCEL_CACHE_DIR` | `~/.cache/mlxcel` | Override mlxcel cache root |

## Notes / limitations

- **reasoning**: detected from chat template tokens (`enable_thinking` / `reasoning_content` / `<think>...</think>`). When detected, `reasoning: true` and `compat.thinkingFormat: "qwen-chat-template"` are set so pi drives MLX's `chat_template_kwargs.enable_thinking` (`off` disables, others enable). Set `MLXCEL_AUTO_NO_REASONING=1` to opt out.
- **vision**: detected from `vision_config` or tokenizer image/video tokens. Maps to `input: ["text","image"]`. Video/audio are not expressible in pi's model `input` field (text/image only).
- **tools**: detected from chat template tool-call markers or `tool_parser_type`. Informational only — pi has no per-model tool toggle, so it is cached as metadata and not reflected in registration.
- `reasoning` defaults to `false` when not detected. Add a `mlxcel` provider in `~/.pi/agent/models.json` to override per model — this extension uses the `mlxcel-auto` provider id so there is no collision.
- If you launch with `--alias <custom>` the id is not a repo id and cannot be resolved; fall back to a manual `models.json` entry in that case.
- `mlxcel-server` reports the bare snapshot directory name in `/v1/models` even when launched with a full `owner/name` repo id; the bare-name → `mlx-community/<name>` mapping handles this.

## License

MIT