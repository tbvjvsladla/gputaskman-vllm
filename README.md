# gputaskman-vllm

**An `htop` for what's inside your GPU — extended for vLLM GPU Memory Breakdown, optional `config.yaml` endpoint/log knobs, and 3-pane TUI layout cycling.**

> **Upstream Credit & Versioning Policy**: This repository is a vLLM-focused fork of [**shanselman/gputaskman**](https://github.com/shanselman/gputaskman) (`v0.4.5`, MIT License, Copyright Scott Hanselman).
> To respect upstream versioning without bumping upstream patch numbers, releases use PEP 440 local SemVer tags on top of the upstream base: **`0.4.5+vllm.<major>.<minor>.<patch>`** (currently `0.4.5+vllm.0.1.1`).
> Full upstream platform documentation (Windows WDDM/TCC, Linux DRM/Ascend, WSL2, macOS Metal, `diff`/`record`/`summary`/`replay`) is preserved in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

---

## 한국어 빠른 시작 (Korean Quick Start)

`gputaskman-vllm`은 기존 `gputaskman` v0.4.5의 모든 기능을 100% 보존하면서, **Native(venv) 및 Docker/원격 vLLM 추론 서버**의 GPU 메모리 구성을 실시간으로 시각화하기 위해 확장된 배포판입니다.

### 1) `uv`로 설치 (어느 폴더에서나 바로 실행)

```bash
# GitHub에서 uv tool로 전역 격리 설치 (권장)
uv tool install git+https://github.com/tbvjvsladla/gputaskman-vllm.git

# 또는 로컬 클론 후 설치
git clone https://github.com/tbvjvsladla/gputaskman-vllm.git
cd gputaskman-vllm
uv tool install .
```

설치 후 `gputaskman`, `gtm`, `gputaskman-vllm` 세 명령을 어느 디렉터리에서나 사용할 수 있습니다.

```bash
gputaskman --version
# gputaskman 0.4.5+vllm.0.1.1
```

### 2) 실행 방법 — 무설정 자동 감지 vs 임의 폴더 `--config <yaml>`

- **무설정 자동 감지 (Native venv vLLM)** — 로컬 `VLLM::EngineCore` → 부모 `vllm serve` 포트 및 `/proc/<pid>/fd/{1,2}` 로그를 자동 감지합니다:
  ```bash
  gputaskman
  ```
- **임의 폴더에서 작성한 `config.yaml` 지정 (Native · Docker · Sidecar · 원격 vLLM)**:
  ```bash
  cat <<'EOF' > /tmp/my_vllm.yaml
  vllm:
    auto_detect: true
    endpoints:
      - name: local-vllm
        url: http://127.0.0.1:8000
        log_path: /tmp/vllm_e2e.log
  EOF

  # 어떤 디렉터리에서 실행해도 동작합니다 (YAML 내부 상대 log_path는 YAML 파일 위치 기준 해석)
  gputaskman --config /tmp/my_vllm.yaml
  gputaskman --config /tmp/my_vllm.yaml --once
  gputaskman --config /tmp/my_vllm.yaml --json
  ```
  - Docker 컨테이너나 원격 서버처럼 로컬 `/proc`에 `VLLM::EngineCore` PID가 보이지 않는 환경에서도, `config.yaml`에 적은 엔드포인트가 전용 행(`C:1`)으로 자동 생성되어 동일한 **vLLM Memory Breakdown**을 보여줍니다.

### 3) 주요 TUI 단축키

| 키 | 기능 |
|---|---|
| **`l`** *(신규)* | **3-패널 수직 배치 4조합 순환**: `(1,2,3) GPU→PID→vLLM` → `(1,3,2) GPU→vLLM→PID` → `(3,1,2) vLLM→GPU→PID` → `(3,2,1) vLLM→PID→GPU` |
| **`↑` / `↓`** | 프로세스 / 설정 엔드포인트 선택 |
| **`m`** | Slim / Full 모니터링 모드 전환 |
| **`c`** | 시계열 스파크라인 차트 모드 순환 |
| **`s`** | 프로세스 테이블 정렬 기준 순환 (`VRAM`, `GPU%`, `Shared`, `RAM`, `CPU%` 등) |
| **`a`** | Top / All 프로세스 필터 토글 |
| **`p`** | 화면 갱신 일시정지 / 재개 |
| **`?`** / **`q`** | 도움말 모달 / 종료 |

---

## What's New in `gputaskman-vllm` (`0.4.5+vllm.0.1.1`)

### 1. Non-Invasive Strategy B vLLM Memory Breakdown (TUI + Additive `--json`/`--once`)

Combines read-only vLLM HTTP endpoints (`/v1/models`, `/metrics` including `vllm:cache_config_info`), bounded/mtime-cached vLLM startup logs (V0/V1 `INFO`, `DEBUG`, and JSON log lines), and NVML GPU telemetry without modifying vLLM source code:

```text
vLLM Memory Breakdown  local-vllm (http://127.0.0.1:8000)  [config]
Process VRAM           N/A  [unattributed]
Device VRAM            5.57 GB / 8.00 GB  [NVML device-wide]
VRAM ██████████████████████████████████████░░░░░░░░░░░░░░░░░░ 5.16 GB profile / 8.00 GB GPU  (device 5.57 GB)
     ■ Model Weights 952 MB  ■ Activation Peak Budget 72 MB  ■ KV Cache Unused 4.12 GB  ■ CUDA Graph 41 MB
  ■ Model Weights              952 MB  [Init profile]
  ■ Activation Peak             72 MB  [Init peak, NOT current]
  ■ KV Cache Used                0 MB  (0.0%)  [derived; includes prefix cache]
  ■ KV Cache Unused           4.12 GB  [derived]
    KV Cache Allocated        4.12 GB  [Init profile]
  ■ Other/Unattributed            N/A  [requires reconciliation]
    CUDA Graph                  41 MB  [Init profile]
    Activation Current            N/A  [requires instrumentation]
  Active Requests KV            N/A  [requires instrumentation]
  Running / Waiting           0 / 0
  KV config: blocks 5,120 · block_size 64 · capacity 327,680 tokens · dtype auto · prefix_cache on
Sources: NVML + Init Profile + Metrics
```

#### Strict Provenance Rules (Inherited from `AGENTS.md`)
- **Unknown is never `0`**: Unmeasurable values (`Activation Current`, `Active Requests KV`) remain `N/A` in TUI/text and `null` in JSON (`confidence: "unavailable"`).
- **No double-counting**: `KV Cache Used + KV Cache Unused == KV Cache Allocated`. `Activation Peak` is explicitly labeled `[Init peak, NOT current]` and never misrepresented as live current activation.
- **Per-field metadata**: Every `--json` vLLM process carries an additive `llm.vllm_memory` object with `data_sources`, `measurement_timestamps`, `measurement_confidence` (`measured` | `init_profile` | `derived` | `unavailable`), and `stale` flags. Existing `diff`, `summary`, `replay`, and `record` schemas remain 100% compatible.
- **vLLM V1 real-world fixes**:
  - Matches HuggingFace Hub cache snapshot `model_name` labels in `/metrics` against `--served-model-name` aliases from `/v1/models`.
  - Handles `VLLM::EngineCore` internal ephemeral ZMQ/RPC ports (`>= 32768`) without blocking parent `vllm serve` auto-detection.
  - O(1) `/proc/<pid>/fd/{1,2}` auto-discovery of redirected stdout/stderr vLLM logs on Linux (never calls slow `psutil.open_files()`).
  - Handles naive UTC container timestamps via UTC `file_mtime` fallback so Docker logs on non-UTC hosts are not falsely marked stale.

### 2. Optional `config.yaml` Knob (Works From Any Directory & Docker/Remote Topologies)

See [gputaskman.yaml.example](gputaskman.yaml.example) for a complete template.

**Config discovery priority** (when `--config` is not passed, absence of config files falls back silently to 100% zero-config auto-detection):
1. `--config <path>` (CLI option; also supported on `gputaskman record --config <path>`)
2. `GPUTASKMAN_CONFIG` environment variable
3. `./gputaskman.yaml` (current working directory)
4. `~/.config/gputaskman/config.yaml`

**Path & Topology stability**:
- Relative `log_path` / `log_paths` entries inside a YAML file are resolved relative to **that YAML file's directory**, so `gputaskman --config /any/folder/vllm.yaml` works regardless of your current working directory.
- **Docker / Sidecar / Remote vLLM**: When no local `/proc` PID matches a configured endpoint, `gputaskman-vllm` synthesizes a dedicated read-only row (`C:1`, `origin: "config"`, `vram_attributed: false`, never added to `Snapshot.attributed`) so you still get the full vLLM Memory Breakdown and live metrics.
- **Secrets safety**: Prefer `api_key_env: MY_VLLM_KEY` over inline `api_key`. API keys are never logged, serialized to JSON, or sent to `/metrics`.

### 3. 3-Pane Vertical Layout Cycling (`l`)

Press **`l`** in the TUI to cycle the 3 vertical panes (`1: GPU Dashboard`, `2: Process Table`, `3: Inspector / vLLM Breakdown`) through 4 layouts while preserving selection, focus, and footer position:

- `(1, 2, 3)` — GPU → PID → vLLM *(default)*
- `(1, 3, 2)` — GPU → vLLM → PID
- `(3, 1, 2)` — vLLM → GPU → PID
- `(3, 2, 1)` — vLLM → PID → GPU

---

## Installation & Development

### Install with `uv` (Recommended)

```bash
# Install globally in an isolated uv tool environment from GitHub
uv tool install git+https://github.com/tbvjvsladla/gputaskman-vllm.git

# Upgrade later
uv tool upgrade gputaskman-vllm

# Or run directly without installing
uvx --from git+https://github.com/tbvjvsladla/gputaskman-vllm.git gputaskman --once
```

### Clone & Run Locally

```bash
git clone https://github.com/tbvjvsladla/gputaskman-vllm.git
cd gputaskman-vllm
uv sync --locked
uv run --locked gputaskman
```

### Enable Full vLLM Startup Profiling Logs (Optional, Recommended)

Default vLLM `INFO` logs provide `Model Weights`, `KV Cache Allocated`, `CUDA Graph`, and `GPU KV cache size`. Setting `VLLM_LOGGING_LEVEL=DEBUG` additionally exposes `Activation Peak` (`torch peak memory increase`):

```bash
VLLM_LOGGING_LEVEL=DEBUG \
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --enable-prefix-caching \
  2>&1 | tee /tmp/vllm.log
```

### Verification & Tests

```bash
# Full hardware-independent unit & regression suite (835+ tests)
uv run --locked python -B -m unittest discover -s tests -p "test_*.py" -q

# Offline self-check of the live vLLM E2E harness
uv run --locked python -B tools/live_e2e_vllm.py --self-check

# Live E2E verification against a running vLLM server
VLLM_E2E_URL=http://127.0.0.1:8000 \
VLLM_E2E_LOG=/tmp/vllm.log \
VLLM_E2E_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
uv run --locked python -B tools/live_e2e_vllm.py --verify
```

### Live Hardware Validation

Validated live on **WSL2 Linux + NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB, Driver 616.92, CUDA 13.4)** against **Native vLLM 0.34.0** (`Qwen/Qwen2.5-0.5B-Instruct`, FP16, `max-model-len 2048`, `gpu-memory-utilization 0.65`, `--enable-prefix-caching`, `VLLM_LOGGING_LEVEL=DEBUG`) across Idle, Single Request, Concurrent + Prefix Cache, Zero-Config Auto-detect, Explicit `--config` YAML, Fault/503 Fallback, and `gputaskman diff`.

---

## License

MIT License — see [LICENSE](LICENSE).
Original project: Copyright (c) 2026 Scott Hanselman ([shanselman/gputaskman](https://github.com/shanselman/gputaskman)).
