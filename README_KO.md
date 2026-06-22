# pi-mlxcel-auto

[pi](https://pi.dev) 확장 프로그램으로, 실행 중인 **`mlxcel-server`**([lablup/mlxcel](https://github.com/lablup/mlxcel)의 Rust MLX 러너, OpenAI 호환)를 자동 탐지하여 모델을 pi에 등록합니다. **Hugging Face 메타데이터에서 실제 컨텍스트 윈도우를 읽어오므로 `ctx-size`를 수동으로 입력할 필요가 없습니다.**

`http` 및 `https` 엔드포인트, 로컬 및 원격 서버를 모두 지원합니다. mlxcel, mlx-lm 등 모든 OpenAI 호환 MLX 서버에 작동합니다.

## 동작 방식

1. 설정된 각 서버 URL(`/v1/models`)을 프로브합니다. 기본값 `http://127.0.0.1:8080`.
2. 각 모델 id에 대해:
   - Hugging Face에서 메타데이터를 가져옵니다(`config.json`, `tokenizer_config.json`, `chat_template.jinja`). 결과는 영구 캐싱됩니다 — 모델 메타데이터는 변하지 않습니다.
   - 슬래시가 없는 모델 이름은 `${MLXCEL_DEFAULT_ORG}/<이름>`(기본값 `mlx-community`)을 먼저 시도하고, 404면 Hugging Face 검색 API로 올바른 조직을 자동으로 찾습니다. 어느 조직의 모델이든 자동으로 처리됩니다.
   - `max_position_embeddings`(최상위, 그 다음 `text_config.*`)에서 컨텍스트 윈도우를 추출합니다.
   - `vision_config` 또는 토크나이저 이미지/비디오 토큰에서 비전을 감지합니다.
   - 채팅 템플릿 토큰(`enable_thinking`, `reasoning_content`, `think` 등)에서 리소닝을 감지합니다.
3. `pi.registerProvider()`를 통해 `mlxcel-auto`(기본 URL) 또는 `mlxcel-auto-<host>-<port>` 프로바이더를 등록합니다. `openai-completions` API와 안전한 `compat` 플래그를 사용합니다.
4. 결과를 `~/.pi/agent/extensions-data/mlxcel-auto-cache.json`에 캐시합니다.
5. `session_start` 시 다시 프로브하여 새로 시작된 서버를 자동으로 감지합니다.

## 설치

```bash
pi install git:github.com/sng2c/pi-mlxcel-auto
```

pi에서:

```text
/reload
/model
```

## 사용법

### 로컬 서버 (기본)

```bash
mlxcel serve --model mlx-community/gemma-3-4b-it-qat-4bit
# 또는: mlxcel-server -m mlx-community/Qwen3.5-0.8B-4bit --port 8080
```

### 원격 서버 (HTTPS)

```bash
MLXCEL_AUTO_BASEURLS=https://ml.my-server.com:8443
```

### 여러 서버

```bash
MLXCEL_AUTO_BASEURLS=http://127.0.0.1:8080,https://ml.my-server.com:8443
```

pi를 실행하거나, 이미 실행 중이면 `/reload`합니다. 수동으로 다시 프로브하려면 `/mlxcel-auto`를 실행합니다.

`/model`에서 `mlxcel-auto/<model>`을 선택합니다. 컨텍스트 윈도우는 자동 감지됩니다.

## 설정 (환경변수)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `MLXCEL_AUTO_BASEURLS` | `http://127.0.0.1:8080` | 쉼표로 구분된 서버 기본 URL. `http` 및 `https` 지원. 스킴 생략 시 `http`로 간주. |
| `MLXCEL_AUTO_APIKEY` | `not-needed` | 서버에 전송할 API 키 |
| `MLXCEL_AUTO_MAXOUT` | `32768` | `maxTokens` 상한; 감지 실패 시 폴백 컨텍스트로도 사용 |
| `MLXCEL_AUTO_NO_REASONING` | (끄기) | `1`이면 자동 리소닝/사고 감지 비활성화 |
| `MLXCEL_AUTO_NO_CACHE` | (끄기) | `1`이면 디스크 설정 캐시 비활성화 |
| `MLXCEL_DEFAULT_ORG` | `mlx-community` | bare 모델 이름 앞에 붙일 조직. 404면 폴백 사용 |

## 참고 / 제한사항

- **리소닝**: 채팅 템플릿 토큰(`enable_thinking` / `reasoning_content` / `clear_thinking` / `think`)에서 감지됩니다. 감지되면 `reasoning: true`, `compat.thinkingFormat: "qwen-chat-template"`, `thinkingLevelMap`이 중간 수준(`minimal`/`low`/`medium`/`xhigh` → `null`)을 숨기고 `off`(비활성화) + `high`(활성화)만 유지합니다. MLX는 `enable_thinking`을 부울 토글로 처리하기 때문입니다. `MLXCEL_AUTO_NO_REASONING=1`로 비활성화할 수 있습니다.
- **컨텍스트 윈도우**: `--ctx-size`가 명시적으로 설정된 경우 서버의 실제 per-slot `context_size`(`/health` 또는 `/slots`)에서 해석됩니다. 그렇지 않으면 `config.json`의 `max_position_embeddings`에서 가져온 모델 최대값으로 폴백합니다. 서버가 실제로 할당한 것보다 큰 윈도우를 pi가 가정하지 못하게 합니다.
- **비전**: `vision_config` 또는 토크나이저 이미지/비디오 토큰에서 감지됩니다. `input: ["text","image"]`로 매핑됩니다. pi의 모델 `input` 필드는 텍스트/이미지만 지원하므로 비디오/오디오는 표현할 수 없습니다.
- **스탑 토큰 트림**: `message_end` 훅이 `mlxcel-auto` 모델에 대해서만 완성된 어시스턴트 메시지 끝에 누출된 스톱 토큰(예: Qwen `<|im_end|>`, Gemma `<end_of_turn>`, GLM ``)을 제거합니다. config/tokenizer_config의 모델별 문자열 `eos_token`도 사용 가능 시 적용됩니다.
- **메타데이터 소스**: 원격 전용(Hugging Face). bare 이름은 `MLXCEL_DEFAULT_ORG/<이름>`(기본값 `mlx-community`)으로 해석됩니다. HF에 없는 모델(404)은 **목록에서 제외**됩니다 — `/model`에 나타나지 않습니다. 로컬 파일 읽기 없음 — mlxcel, mlx-lm, 모든 OpenAI 호환 서버에 동일하게 작동합니다.
- **기본 조직**: `mlxcel-server`는 전체 `owner/name` 리포지토리 id로 시작해도 `/v1/models`에서 bare 스냅샷 디렉토리 이름만 보고합니다. bare 이름은 `MLXCEL_DEFAULT_ORG/<이름>`(기본값 `mlx-community`)으로 해석됩니다. 조직 추측이 틀려도(404) 폴백 컨텍스트가 사용됩니다. `MLXCEL_DEFAULT_ORG`로 오버라이드할 수 있습니다.
- 리소닝은 감지되지 않은 경우 기본값이 `false`입니다. 모델별로 오버라이드하려면 `~/.pi/agent/models.json`에 `mlxcel` 프로바이더를 추가하세요 — 이 확장은 `mlxcel-auto` 프로바이더 id를 사용하므로 충돌이 없습니다.
- `--alias <custom>`으로 시작하면 id가 리포지토리 id가 아니므로 해석할 수 없습니다. 이 경우 수동 `models.json` 항목으로 폴백하세요.
- **HTTPS**: `MLXCEL_AUTO_BASEURLS=https://...`를 설정하여 원격 또는 TLS 종단 mlxcel-server에 연결합니다. `http`와 `https` 모두 완전히 지원됩니다.

## 캐시

`~/.pi/agent/extensions-data/mlxcel-auto-cache.json`은 `/v1/models`가 반환한 각 모델 id를 다음 필드에 매핑합니다:

| 필드 | 설명 |
| --- | --- |
| `modelMaxCtx` | `config.json`의 모델 최대 컨텍스트(`max_position_embeddings`, 최상위 또는 `text_config.*`), 없으면 `tokenizer_config.model_max_length`, 없으면 `MLXCEL_AUTO_MAXOUT` |
| `vision` | `vision_config`, 토크나이저 이미지/비디오 토큰, 또는 채팅 템플릿 id |
| `reasoning` | 채팅 템플릿 id(`enable_thinking` / `reasoning_content` / `clear_thinking` / `think`); `MLXCEL_AUTO_NO_REASONING=1`으로 비활성화 가능 |
| `eosToken` | `config.eos_token_id` 또는 `tokenizer_config.eos_token`(문자열 또는 id 배열); 스탑 토큰 트림에 사용 |

등록된 `contextWindow`은 `--ctx-size`가 명시적으로 설정된 경우 서버의 실제 per-slot `context_size`(`/health` 또는 `/slots`), 그렇지 않으면 `modelMaxCtx`입니다.

### 업데이트 정책

- **캐시 히트**: 캐시된 메타데이터를 그대로 사용. Hugging Face를 재조회하지 않음 — 나중에 모델이 HF에서 사라져도 캐시 항목이 로컬 사용을 유지합니다.
- **캐시 미스 + HF 조회 성공**: 메타데이터를 가져와 영구 캐싱.
- **캐시 미스 + HF 404**: 모델을 **스킵** — `/model`에 나타나지 않습니다.
- **수동 갱신**: `~/.pi/agent/extensions-data/mlxcel-auto-cache.json`(또는 특정 항목)을 삭제하고 `/mlxcel-auto` 재실행.

pi에서 `/mlxcel-auto-info [부분문자열]`로 확인할 수 있습니다.

## 라이선스

MIT