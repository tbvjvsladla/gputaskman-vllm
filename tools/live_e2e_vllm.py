"""Live E2E verification harness for gputaskman Strategy B vLLM GPU Memory Breakdown.

Uses only the Python standard library (plus gputaskman modules in the active venv).
Supports:
  --self-check              Offline self-test with local HTTP stubs (prints LIVE_E2E_SELF_CHECK_OK)
  --capture-baseline PATH   Capture and save a pre-vLLM baseline snapshot JSON
  --verify                  Run the 6 live scenarios + gputaskman diff check (prints LIVE_E2E_VLLM_PASS)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime
import http.server
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Ensure local src/ is importable when invoked directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gputaskman.collector import Collector  # noqa: E402
from gputaskman.config import load_config  # noqa: E402
from gputaskman.llm import vllm_cache_config, vllm_metrics, vllm_models  # noqa: E402
from gputaskman.report import to_dict  # noqa: E402


# Bypass any ambient HTTP_PROXY when talking to local loopback endpoints.
_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_PROM_LINE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?[ \t]+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)(?:[ \t]+-?\d+)?[ \t]*$"
)
_HOME_PATH_RE = re.compile(r"/home/[^/\s\"']+")
_WIN_USER_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")

_SHARED_PREFIX_UNIT = (
    "Distributed inference engines manage GPU video memory across model weights, pre-allocated KV cache pools, "
    "runtime allocators, CUDA contexts, and transient activation buffers during prefill and decode steps. "
    "When prefix caching is enabled, identical prompt prefixes reuse cached key-value blocks across concurrent "
    "requests, keeping logical KV pool occupancy elevated even while active request counts fluctuate. "
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sanitize_value(obj: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    """Recursively redact secrets and personal home paths from evidence payloads."""
    home_str = str(Path.home())
    active_secrets = [s for s in secrets if s]
    if isinstance(obj, str):
        text = obj
        for secret in active_secrets:
            text = text.replace(secret, "***")
        if home_str and home_str not in ("/", "/root"):
            text = text.replace(home_str, "/home/<user>")
        text = _HOME_PATH_RE.sub("/home/<user>", text)
        text = _WIN_USER_RE.sub(r"C:\\Users\\<user>", text)
        return text
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key_s = str(k)
            if "api_key" in key_s.lower() and key_s.lower() not in ("api_key_env", "has_api_key"):
                out[key_s] = "***" if v else None
            else:
                out[key_s] = sanitize_value(v, secrets=secrets)
        return out
    if isinstance(obj, list):
        return [sanitize_value(item, secrets=secrets) for item in obj]
    if isinstance(obj, tuple):
        return [sanitize_value(item, secrets=secrets) for item in obj]
    return obj


def _http_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_s: float = 10.0,
) -> tuple[int, bytes]:
    headers: dict[str, str] = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.8"}
    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _HTTP_OPENER.open(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read(8 << 20)
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read(1 << 20)
        except Exception:
            pass
        return int(exc.code), body


def fetch_models(base_url: str, *, api_key: str | None = None, timeout_s: float = 5.0) -> list[str]:
    status, body = _http_request(
        base_url.rstrip("/") + "/v1/models", api_key=api_key, timeout_s=timeout_s
    )
    if status != 200:
        raise RuntimeError(f"GET /v1/models returned HTTP {status}")
    parsed = json.loads(body.decode("utf-8", "replace"))
    entries = vllm_models(parsed)
    if not entries:
        raise RuntimeError("GET /v1/models returned no valid model entries")
    return [m["name"] for m in entries]


def fetch_metrics_text(base_url: str, *, timeout_s: float = 5.0) -> str:
    # Keep /metrics keyless to match gputaskman's Strategy B contract.
    status, body = _http_request(base_url.rstrip("/") + "/metrics", timeout_s=timeout_s)
    if status != 200:
        raise RuntimeError(f"GET /metrics returned HTTP {status}")
    return body.decode("utf-8", "replace")


def parse_vllm_prometheus_counters(text: str) -> dict[str, float]:
    """Extract numeric sums of key vLLM counters/gauges (tokens + prefix cache) from Prometheus text."""
    wanted_exact = {
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hit_rate",
        "vllm:prefix_cache_hit_rate",
    }
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith("vllm:"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name not in wanted_exact and "prefix_cache" not in name:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


def send_completion(
    base_url: str,
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 128,
    api_key: str | None = None,
    timeout_s: float = 30.0,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    """Send a single POST /v1/completions request (falling back without ignore_eos if HTTP 400)."""
    url = base_url.rstrip("/") + "/v1/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    t0 = time.perf_counter()
    status, body = _http_request(url, method="POST", payload=payload, api_key=api_key, timeout_s=timeout_s)
    if status == 400 and ignore_eos:
        payload.pop("ignore_eos", None)
        status, body = _http_request(url, method="POST", payload=payload, api_key=api_key, timeout_s=timeout_s)
    elapsed = time.perf_counter() - t0
    usage: dict[str, Any] = {}
    output_chars = 0
    if status == 200:
        try:
            parsed = json.loads(body.decode("utf-8", "replace"))
            if isinstance(parsed.get("usage"), dict):
                usage = parsed["usage"]
            choices = parsed.get("choices") or []
            if choices and isinstance(choices[0], dict):
                output_chars = len(str(choices[0].get("text") or ""))
        except ValueError:
            pass
    return {
        "http_status": status,
        "elapsed_s": round(elapsed, 3),
        "usage": usage,
        "output_chars": output_chars,
    }


def build_config_yaml(
    *,
    auto_detect: bool,
    url: str,
    name: str,
    log_path: str | None = None,
    gpu_index: int | None = 0,
    api_key_env: str | None = None,
    timeout_s: float = 3.0,
) -> str:
    """Render a valid gputaskman config.yaml without embedding raw secrets."""
    lines = [
        "vllm:",
        f"  auto_detect: {'true' if auto_detect else 'false'}",
        "  endpoints:",
        f"    - url: {json.dumps(url)}",
        f"      name: {json.dumps(name)}",
        f"      timeout_s: {float(timeout_s):.2f}",
    ]
    if gpu_index is not None:
        lines.append(f"      gpu_index: {int(gpu_index)}")
    if log_path:
        lines.append(f"      log_path: {json.dumps(log_path)}")
    if api_key_env:
        lines.append(f"      api_key_env: {json.dumps(api_key_env)}")
    return "\n".join(lines) + "\n"


def _run_gputaskman(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> tuple[int, str, str, float]:
    cmd = [sys.executable, "-B", "-m", "gputaskman", *args]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - t0


def find_vllm_row(snapshot: dict[str, Any], *, expected_url: str | None = None) -> dict[str, Any] | None:
    """Locate the process row containing llm.vllm_memory (preferring an endpoint_url match)."""
    candidates = [
        proc
        for proc in snapshot.get("processes", [])
        if isinstance(proc, dict)
        and isinstance(proc.get("llm"), dict)
        and isinstance(proc["llm"].get("vllm_memory"), dict)
    ]
    if not candidates:
        return None
    if expected_url:
        norm = expected_url.rstrip("/")
        for proc in candidates:
            vm = proc["llm"]["vllm_memory"]
            if (vm.get("endpoint_url") or "").rstrip("/") == norm:
                return proc
    # Prefer a row with a live server or weights populated.
    candidates.sort(
        key=lambda p: (
            bool(p["llm"].get("server")),
            p["llm"]["vllm_memory"].get("kv_pool_usage_ratio") is not None,
            p["llm"]["vllm_memory"].get("weights_bytes") is not None,
        ),
        reverse=True,
    )
    return candidates[0]


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def _check_breakdown_common_invariants(
    checks: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    prefix: str,
    require_init_log: bool,
    require_live_metrics: bool,
) -> None:
    vm = row["llm"]["vllm_memory"]
    conf = vm.get("measurement_confidence") or {}
    srcs = vm.get("data_sources") or {}
    tss = vm.get("measurement_timestamps") or {}

    _check(
        checks,
        f"{prefix}.activation_current_is_none",
        vm.get("activation_current_bytes") is None and conf.get("activation_current_bytes") == "unavailable",
        f"val={vm.get('activation_current_bytes')}, conf={conf.get('activation_current_bytes')}",
    )
    _check(
        checks,
        f"{prefix}.kv_active_request_is_none",
        vm.get("kv_active_request_bytes") is None and conf.get("kv_active_request_bytes") == "unavailable",
        f"val={vm.get('kv_active_request_bytes')}, conf={conf.get('kv_active_request_bytes')}",
    )

    kv_alloc = vm.get("kv_allocated_bytes")
    kv_occ = vm.get("kv_pool_occupied_bytes")
    kv_avail = vm.get("kv_pool_available_bytes")
    kv_ratio = vm.get("kv_pool_usage_ratio")

    if require_init_log:
        _check(
            checks,
            f"{prefix}.init_log_not_stale",
            vm.get("stale") is False,
            f"stale={vm.get('stale')}, reason={vm.get('stale_reason')}",
        )
        _check(
            checks,
            f"{prefix}.weights_bytes_positive",
            isinstance(vm.get("weights_bytes"), int)
            and vm["weights_bytes"] > 0
            and conf.get("weights_bytes") == "init_profile",
            f"weights={vm.get('weights_bytes')}, conf={conf.get('weights_bytes')}, src={srcs.get('weights_bytes')}",
        )
        _check(
            checks,
            f"{prefix}.kv_allocated_positive",
            isinstance(kv_alloc, int) and kv_alloc > 0 and conf.get("kv_allocated_bytes") == "init_profile",
            f"kv_allocated={kv_alloc}, conf={conf.get('kv_allocated_bytes')}, src={srcs.get('kv_allocated_bytes')}",
        )

    if require_live_metrics:
        _check(
            checks,
            f"{prefix}.kv_ratio_in_bounds",
            isinstance(kv_ratio, (int, float)) and 0.0 <= float(kv_ratio) <= 1.0 and conf.get("kv_pool_usage_ratio") == "measured",
            f"ratio={kv_ratio}, conf={conf.get('kv_pool_usage_ratio')}",
        )
        if isinstance(kv_alloc, int) and kv_alloc > 0:
            _check(
                checks,
                f"{prefix}.kv_occupied_in_bounds",
                isinstance(kv_occ, int)
                and 0 <= kv_occ <= kv_alloc
                and isinstance(kv_avail, int)
                and kv_occ + kv_avail == kv_alloc
                and conf.get("kv_pool_occupied_bytes") == "derived",
                f"occ={kv_occ}, avail={kv_avail}, alloc={kv_alloc}, conf={conf.get('kv_pool_occupied_bytes')}",
            )

    attributed = bool(row.get("vram_attributed"))
    if not attributed:
        _check(
            checks,
            f"{prefix}.unattributed_no_guessed_other",
            vm.get("total_vram_used_bytes") is None
            and vm.get("other_runtime_bytes") is None
            and isinstance(vm.get("device_vram_used_bytes"), int)
            and vm["device_vram_used_bytes"] > 0
            and (row["llm"].get("segments") or []) == [],
            f"total={vm.get('total_vram_used_bytes')}, other={vm.get('other_runtime_bytes')}, "
            f"device_used={vm.get('device_vram_used_bytes')}, segments={len(row['llm'].get('segments') or [])}",
        )
    else:
        total_used = vm.get("total_vram_used_bytes")
        other = vm.get("other_runtime_bytes")
        w = vm.get("weights_bytes")
        no_double = True
        if isinstance(total_used, int) and isinstance(w, int) and isinstance(kv_alloc, int) and isinstance(other, int):
            no_double = other >= 0 and (w + kv_alloc + other == total_used)
        segs = row["llm"].get("segments") or []
        seg_sum_ok = (not segs) or (isinstance(total_used, int) and sum(int(s.get("bytes", 0)) for s in segs) == total_used)
        _check(
            checks,
            f"{prefix}.attributed_no_double_counting",
            isinstance(total_used, int) and total_used > 0 and no_double and seg_sum_ok,
            f"total={total_used}, weights={w}, kv_alloc={kv_alloc}, other={other}, segs={len(segs)}",
        )

    _check(
        checks,
        f"{prefix}.provenance_populated",
        bool(srcs) and bool(conf) and bool(tss),
        f"sources={len(srcs)}, conf={len(conf)}, timestamps={len(tss)}",
    )


def evaluate_s1_baseline(
    current_snap: dict[str, Any],
    baseline_snap: dict[str, Any] | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []
    dev = current_snap.get("device") or {}
    vram_total = current_snap.get("vram_total")
    vram_used = current_snap.get("vram_used")

    _check(checks, "nvml_source", dev.get("source") == "NVML", f"device.source={dev.get('source')}")
    _check(checks, "vram_total_positive", isinstance(vram_total, int) and vram_total > 0, f"vram_total={vram_total}")
    _check(checks, "vram_used_measured", isinstance(vram_used, int) and vram_used >= 0, f"vram_used={vram_used}")

    observed: dict[str, Any] = {
        "baseline_provided": baseline_snap is not None,
        "gpu": current_snap.get("gpu"),
        "backend": current_snap.get("backend"),
        "device_source": dev.get("source"),
        "vram_total_bytes": vram_total,
        "current_vram_used_bytes": vram_used,
    }
    if baseline_snap is not None:
        b_dev = baseline_snap.get("device") or {}
        b_total = baseline_snap.get("vram_total")
        b_used = baseline_snap.get("vram_used")
        _check(checks, "baseline_nvml_source", b_dev.get("source") == "NVML", f"baseline.device.source={b_dev.get('source')}")
        _check(checks, "baseline_vram_total_positive", isinstance(b_total, int) and b_total > 0, f"baseline.vram_total={b_total}")
        _check(
            checks,
            "vram_increased_after_vllm_load",
            isinstance(b_used, int) and isinstance(vram_used, int) and b_used < vram_used,
            f"before={b_used}, after={vram_used}",
        )
        observed["baseline_vram_used_bytes"] = b_used
        observed["vram_delta_bytes"] = (vram_used - b_used) if isinstance(b_used, int) and isinstance(vram_used, int) else None

    passed = all(c["ok"] for c in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "observed": observed,
        "duration_s": round(time.perf_counter() - t0, 3),
    }


def evaluate_s2_idle(
    idle_snap: dict[str, Any],
    idle_text: str,
    *,
    expected_url: str,
    model_id: str,
    has_log: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []
    row = find_vllm_row(idle_snap, expected_url=expected_url)
    if row is None:
        _check(
            checks,
            "vllm_row_present",
            False,
            f"No process row with llm.vllm_memory found in snapshot (backend={idle_snap.get('backend')}, "
            f"processes={len(idle_snap.get('processes') or [])}). If vLLM is running inside an isolated "
            "Docker PID namespace on WSL2, run vLLM natively in the same WSL2 distro for full live E2E.",
        )
        return {
            "status": "FAIL",
            "checks": checks,
            "observed": {
                "backend": idle_snap.get("backend"),
                "process_count": len(idle_snap.get("processes") or []),
            },
            "duration_s": round(time.perf_counter() - t0, 3),
        }

    vm = row["llm"]["vllm_memory"]
    _check(checks, "vllm_row_present", True, f"pid={row.get('pid')}, name={row.get('name')}")
    _check(
        checks,
        "model_reported",
        model_id in (vm.get("model_names") or []),
        f"expected={model_id}, reported={vm.get('model_names')}",
    )
    _check(
        checks,
        "idle_requests_zero",
        vm.get("running_requests") == 0 and vm.get("waiting_requests") == 0,
        f"running={vm.get('running_requests')}, waiting={vm.get('waiting_requests')}",
    )
    _check_breakdown_common_invariants(
        checks, row, prefix="idle", require_init_log=has_log, require_live_metrics=True
    )

    for marker in (
        "vLLM Memory Breakdown",
        "Weights:",
        "KV Cache Allocated:",
        "KV Pool Occupied:",
        "Activation Current: N/A",
        "Active Requests KV: N/A",
        "Running / Waiting: 0 / 0",
        "Sources:",
    ):
        _check(checks, f"once_text_has_{marker.split(':')[0].lower().replace(' ', '_')}", marker in idle_text, f"marker={marker!r}")

    observed = {
        "pid": row.get("pid"),
        "process_name": row.get("name"),
        "vram_attributed": row.get("vram_attributed"),
        "match_source": vm.get("match_source"),
        "weights_bytes": vm.get("weights_bytes"),
        "kv_allocated_bytes": vm.get("kv_allocated_bytes"),
        "kv_pool_usage_ratio": vm.get("kv_pool_usage_ratio"),
        "kv_pool_occupied_bytes": vm.get("kv_pool_occupied_bytes"),
        "kv_pool_available_bytes": vm.get("kv_pool_available_bytes"),
        "activation_peak_bytes": vm.get("activation_peak_bytes"),
        "activation_current_bytes": vm.get("activation_current_bytes"),
        "kv_active_request_bytes": vm.get("kv_active_request_bytes"),
        "cuda_graph_bytes": vm.get("cuda_graph_bytes"),
        "other_runtime_bytes": vm.get("other_runtime_bytes"),
        "device_vram_used_bytes": vm.get("device_vram_used_bytes"),
        "total_vram_capacity_bytes": vm.get("total_vram_capacity_bytes"),
        "kv_num_gpu_blocks": vm.get("kv_num_gpu_blocks"),
        "kv_block_size": vm.get("kv_block_size"),
        "kv_cache_size_tokens": vm.get("kv_cache_size_tokens"),
        "prefix_caching_enabled": vm.get("prefix_caching_enabled"),
        "data_sources": vm.get("data_sources"),
        "measurement_confidence": vm.get("measurement_confidence"),
        "measurement_timestamps": vm.get("measurement_timestamps"),
    }
    passed = all(c["ok"] for c in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "observed": observed,
        "duration_s": round(time.perf_counter() - t0, 3),
    }


def _poll_collector_samples(
    collector: Collector,
    stop_event: threading.Event,
    *,
    expected_url: str,
    interval_s: float = 0.25,
) -> list[dict[str, Any]]:
    """Sample via gputaskman Collector at interval_s, bypassing ServerProber TTL cache while preserving rate baselines."""
    samples: list[dict[str, Any]] = []
    while True:
        t0 = time.perf_counter()
        collector.prober._cache.clear()
        snap_dict = to_dict(collector.sample())
        samples.append(snap_dict)
        if stop_event.is_set():
            break
        remaining = interval_s - (time.perf_counter() - t0)
        if remaining > 0:
            stop_event.wait(remaining)
    # One final post-stop fresh reading.
    collector.prober._cache.clear()
    samples.append(to_dict(collector.sample()))
    return samples


def evaluate_s3_single_request(
    *,
    request_result: dict[str, Any],
    counters_before: dict[str, float],
    counters_after: dict[str, float],
    polled_snapshots: list[dict[str, Any]],
    idle_metrics_ts: str | None,
    expected_url: str,
    has_log: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        "http_200",
        request_result.get("http_status") == 200,
        f"status={request_result.get('http_status')}, elapsed_s={request_result.get('elapsed_s')}",
    )

    p_before = counters_before.get("vllm:prompt_tokens_total", 0.0)
    p_after = counters_after.get("vllm:prompt_tokens_total", 0.0)
    g_before = counters_before.get("vllm:generation_tokens_total", 0.0)
    g_after = counters_after.get("vllm:generation_tokens_total", 0.0)
    _check(checks, "prompt_tokens_increased", p_after > p_before, f"before={p_before}, after={p_after}")
    _check(checks, "generation_tokens_increased", g_after > g_before, f"before={g_before}, after={g_after}")

    vllm_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for snap in polled_snapshots:
        row = find_vllm_row(snap, expected_url=expected_url)
        if row is not None:
            vllm_rows.append((snap, row))

    _check(checks, "polled_vllm_samples_present", len(vllm_rows) >= 1, f"samples={len(vllm_rows)}")

    max_running = 0
    max_kv_occ = 0
    max_kv_ratio = 0.0
    observed_ts: list[str] = []
    invariants_ok = True
    best_snap: dict[str, Any] | None = None

    for snap, row in vllm_rows:
        vm = row["llm"]["vllm_memory"]
        running = vm.get("running_requests")
        occ = vm.get("kv_pool_occupied_bytes")
        ratio = vm.get("kv_pool_usage_ratio")
        alloc = vm.get("kv_allocated_bytes")
        ts = (vm.get("measurement_timestamps") or {}).get("kv_pool_usage_ratio")
        if isinstance(ts, str):
            observed_ts.append(ts)
        if isinstance(running, int) and running > max_running:
            max_running = running
            best_snap = snap
        if isinstance(occ, int) and occ > max_kv_occ:
            max_kv_occ = occ
            if best_snap is None:
                best_snap = snap
        if isinstance(ratio, (int, float)) and float(ratio) > max_kv_ratio:
            max_kv_ratio = float(ratio)
        if vm.get("activation_current_bytes") is not None or vm.get("kv_active_request_bytes") is not None:
            invariants_ok = False
        if has_log and isinstance(alloc, int) and isinstance(occ, int) and not (0 <= occ <= alloc):
            invariants_ok = False

    if best_snap is None and vllm_rows:
        best_snap = vllm_rows[-1][0]

    _check(
        checks,
        "observed_active_or_kv_occupancy",
        max_running >= 1 or max_kv_occ > 0 or max_kv_ratio > 0.0,
        f"max_running={max_running}, max_kv_occ={max_kv_occ}, max_kv_ratio={max_kv_ratio:.4f}",
    )
    unique_ts = sorted(set(observed_ts))
    ts_advanced = len(unique_ts) >= 2 or (bool(unique_ts) and idle_metrics_ts is not None and unique_ts[-1] != idle_metrics_ts)
    _check(
        checks,
        "measurement_timestamps_updated",
        ts_advanced,
        f"unique_metrics_timestamps={len(unique_ts)}, idle_ts={idle_metrics_ts}",
    )
    _check(
        checks,
        "unmeasurable_invariants_preserved",
        invariants_ok,
        f"checked_samples={len(vllm_rows)}",
    )

    passed = all(c["ok"] for c in checks)
    return (
        {
            "status": "PASS" if passed else "FAIL",
            "checks": checks,
            "observed": {
                "http_status": request_result.get("http_status"),
                "request_elapsed_s": request_result.get("elapsed_s"),
                "prompt_tokens_delta": round(p_after - p_before, 1),
                "generation_tokens_delta": round(g_after - g_before, 1),
                "polled_samples": len(vllm_rows),
                "max_running_requests": max_running,
                "max_kv_pool_occupied_bytes": max_kv_occ,
                "max_kv_pool_usage_ratio": round(max_kv_ratio, 6),
                "distinct_metrics_timestamps": len(unique_ts),
            },
            "duration_s": round(time.perf_counter() - t0, 3),
        },
        best_snap,
    )


def evaluate_s4_concurrent(
    *,
    wave_results: list[list[dict[str, Any]]],
    counters_before: dict[str, float],
    counters_after: dict[str, float],
    polled_snapshots: list[dict[str, Any]],
    expected_url: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []

    flat = [r for wave in wave_results for r in wave]
    all_200 = bool(flat) and all(r.get("http_status") == 200 for r in flat)
    _check(
        checks,
        "all_concurrent_requests_200",
        all_200,
        f"total={len(flat)}, ok={sum(1 for r in flat if r.get('http_status') == 200)}",
    )

    vllm_rows = [
        row
        for snap in polled_snapshots
        if (row := find_vllm_row(snap, expected_url=expected_url)) is not None
    ]
    _check(checks, "polled_vllm_samples_present", len(vllm_rows) >= 1, f"samples={len(vllm_rows)}")

    peak_kv_occ = 0
    peak_kv_ratio = 0.0
    peak_running = 0
    peak_waiting = 0
    kv_alloc_seen: int | None = None
    kv_bounds_ok = True
    active_kv_always_none = True
    act_curr_always_none = True
    prefix_enabled: bool | None = None

    for row in vllm_rows:
        vm = row["llm"]["vllm_memory"]
        alloc = vm.get("kv_allocated_bytes")
        occ = vm.get("kv_pool_occupied_bytes")
        ratio = vm.get("kv_pool_usage_ratio")
        running = vm.get("running_requests")
        waiting = vm.get("waiting_requests")
        if prefix_enabled is None and vm.get("prefix_caching_enabled") is not None:
            prefix_enabled = vm.get("prefix_caching_enabled")
        if isinstance(alloc, int) and alloc > 0:
            kv_alloc_seen = alloc
        if isinstance(occ, int):
            peak_kv_occ = max(peak_kv_occ, occ)
            if isinstance(alloc, int) and not (0 <= occ <= alloc):
                kv_bounds_ok = False
        if isinstance(ratio, (int, float)):
            peak_kv_ratio = max(peak_kv_ratio, float(ratio))
        if isinstance(running, int):
            peak_running = max(peak_running, running)
        if isinstance(waiting, int):
            peak_waiting = max(peak_waiting, waiting)
        if vm.get("kv_active_request_bytes") is not None:
            active_kv_always_none = False
        if vm.get("activation_current_bytes") is not None:
            act_curr_always_none = False

    _check(
        checks,
        "peak_kv_occupied_positive_and_bounded",
        peak_kv_occ > 0 and kv_bounds_ok and (kv_alloc_seen is None or peak_kv_occ <= kv_alloc_seen),
        f"peak_kv_occ={peak_kv_occ}, kv_alloc={kv_alloc_seen}, peak_ratio={peak_kv_ratio:.4f}",
    )
    _check(
        checks,
        "kv_active_request_always_none",
        active_kv_always_none,
        f"checked_samples={len(vllm_rows)}",
    )
    _check(
        checks,
        "activation_current_always_none",
        act_curr_always_none,
        f"checked_samples={len(vllm_rows)}",
    )

    prefix_deltas: dict[str, float] = {}
    for k in sorted(set(counters_before) | set(counters_after)):
        if "prefix_cache" in k:
            prefix_deltas[k] = round(counters_after.get(k, 0.0) - counters_before.get(k, 0.0), 4)

    passed = all(c["ok"] for c in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "observed": {
            "waves": len(wave_results),
            "requests_total": len(flat),
            "polled_samples": len(vllm_rows),
            "peak_running_requests": peak_running,
            "peak_waiting_requests": peak_waiting,
            "peak_kv_pool_occupied_bytes": peak_kv_occ,
            "peak_kv_pool_usage_ratio": round(peak_kv_ratio, 6),
            "kv_allocated_bytes": kv_alloc_seen,
            "prefix_caching_enabled": prefix_enabled,
            "prefix_cache_metrics_delta": prefix_deltas,
        },
        "duration_s": round(time.perf_counter() - t0, 3),
    }


def evaluate_s5_modes(
    *,
    config_snap: dict[str, Any],
    auto_rc: int,
    auto_snap: dict[str, Any] | None,
    expected_url: str,
    api_key_used: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []

    cfg_row = find_vllm_row(config_snap, expected_url=expected_url)
    cfg_vm = cfg_row["llm"]["vllm_memory"] if cfg_row else {}
    _check(
        checks,
        "config_yaml_mode_collected",
        cfg_row is not None
        and cfg_vm.get("match_source") == "config"
        and cfg_vm.get("kv_pool_usage_ratio") is not None,
        f"match_source={cfg_vm.get('match_source')}, ratio={cfg_vm.get('kv_pool_usage_ratio')}",
    )

    _check(checks, "auto_detect_exit_zero", auto_rc == 0 and auto_snap is not None, f"rc={auto_rc}")
    topology = "unknown"
    auto_row = find_vllm_row(auto_snap or {})
    if auto_snap is not None:
        if auto_row is not None:
            auto_vm = auto_row["llm"]["vllm_memory"]
            srv = auto_row["llm"].get("server")
            srv_src = auto_row["llm"].get("server_source")
            if srv and srv.get("kind") == "vllm" and srv_src in ("process", "parent") and auto_vm.get("kv_pool_usage_ratio") is not None:
                topology = "native_matched"
                _check(
                    checks,
                    "auto_detect_mode_result",
                    True,
                    f"topology={topology}, server_source={srv_src}, url={auto_vm.get('endpoint_url')}",
                )
            elif api_key_used and any("401" in n or "locked" in n.lower() for n in (auto_row["llm"].get("notes") or [])):
                topology = "native_api_key_locked_safe_fallback"
                _check(checks, "auto_detect_mode_result", True, f"topology={topology}")
            else:
                topology = "native_unmatched"
                _check(
                    checks,
                    "auto_detect_mode_result",
                    False,
                    f"Local vLLM process seen (pid={auto_row.get('pid')}, name={auto_row.get('name')}) "
                    f"but auto-detect did not match server (server_source={srv_src}, notes={auto_row['llm'].get('notes')})",
                )
        else:
            dev_ok = (auto_snap.get("device") or {}).get("source") == "NVML"
            topology = "container_isolated_safe_fallback"
            _check(
                checks,
                "auto_detect_mode_result",
                dev_ok,
                f"topology={topology}, processes={len(auto_snap.get('processes') or [])}, device.source={(auto_snap.get('device') or {}).get('source')}",
            )

    passed = all(c["ok"] for c in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "observed": {
            "config_match_source": cfg_vm.get("match_source"),
            "auto_detect_topology": topology,
            "auto_detect_server_source": (auto_row["llm"].get("server_source") if auto_row else None),
            "auto_detect_endpoint_url": (auto_row["llm"]["vllm_memory"].get("endpoint_url") if auto_row else None),
        },
        "duration_s": round(time.perf_counter() - t0, 3),
    }


def evaluate_s6_fault(
    *,
    case_name: str,
    rc: int,
    elapsed_s: float,
    snap: dict[str, Any] | None,
    expected_url: str,
    expect_unreachable_note: bool,
    has_log: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _check(checks, f"{case_name}.exit_zero_fast", rc == 0 and elapsed_s < 5.0, f"rc={rc}, elapsed_s={elapsed_s:.2f}")
    row = find_vllm_row(snap or {}, expected_url=expected_url) if snap else None
    if row is None:
        _check(checks, f"{case_name}.vllm_row_present", False, "No vLLM row found in fault snapshot")
        return checks
    vm = row["llm"]["vllm_memory"]
    conf = vm.get("measurement_confidence") or {}
    _check(
        checks,
        f"{case_name}.realtime_metrics_none_not_zero",
        vm.get("kv_pool_usage_ratio") is None
        and vm.get("kv_pool_occupied_bytes") is None
        and vm.get("running_requests") is None
        and vm.get("waiting_requests") is None,
        f"ratio={vm.get('kv_pool_usage_ratio')}, occ={vm.get('kv_pool_occupied_bytes')}, "
        f"run={vm.get('running_requests')}, wait={vm.get('waiting_requests')}",
    )
    _check(
        checks,
        f"{case_name}.confidence_unavailable",
        conf.get("kv_pool_usage_ratio") == "unavailable" and conf.get("kv_pool_occupied_bytes") == "unavailable",
        f"ratio_conf={conf.get('kv_pool_usage_ratio')}, occ_conf={conf.get('kv_pool_occupied_bytes')}",
    )
    if expect_unreachable_note:
        notes = list(row["llm"].get("notes") or []) + list(vm.get("notes") or [])
        _check(
            checks,
            f"{case_name}.unreachable_note_recorded",
            any("unreachable" in n.lower() for n in notes),
            f"notes_count={len(notes)}",
        )
    if has_log:
        _check(
            checks,
            f"{case_name}.init_log_fallback_preserved",
            isinstance(vm.get("weights_bytes"), int) and vm["weights_bytes"] > 0,
            f"weights={vm.get('weights_bytes')}",
        )
    return checks


def run_diff_check(before_snap: dict[str, Any], after_snap: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gputaskman_e2e_diff_") as td:
        before_path = Path(td) / "before.json"
        after_path = Path(td) / "after.json"
        before_path.write_text(json.dumps(before_snap), encoding="utf-8")
        after_path.write_text(json.dumps(after_snap), encoding="utf-8")

        rc_txt, out_txt, err_txt, _ = _run_gputaskman(["diff", str(before_path), str(after_path)])
        rc_js, out_js, err_js, _ = _run_gputaskman(["diff", "--json", str(before_path), str(after_path)])

    _check(checks, "diff_text_exit_zero", rc_txt == 0 and "gputaskman diff:" in out_txt, f"rc={rc_txt}, err={err_txt.strip()}")
    diff_obj: dict[str, Any] | None = None
    if rc_js == 0:
        try:
            diff_obj = json.loads(out_js)
        except ValueError:
            diff_obj = None
    _check(
        checks,
        "diff_json_valid",
        rc_js == 0 and isinstance(diff_obj, dict) and "device" in diff_obj and "processes" in diff_obj,
        f"rc={rc_js}, err={err_js.strip()}",
    )
    passed = all(c["ok"] for c in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "observed": {
            "vram_used_delta": (diff_obj or {}).get("device", {}).get("vram_used", {}).get("delta"),
            "matched_processes": len((diff_obj or {}).get("processes") or []),
        },
        "duration_s": round(time.perf_counter() - t0, 3),
    }


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _local_fault_stub(model_id: str):
    """Local HTTP stub returning valid /v1/models (200) and HTTP 503 malformed /metrics."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": model_id}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = b"# degraded metrics\nvllm:kv_cache_usage_perc broken_value\n"
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.02), daemon=True)
    th.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2.0)


@contextlib.contextmanager
def _local_healthy_stub(model_id: str):
    """Offline self-check stub simulating /v1/models, /metrics, and /v1/completions."""
    state = {"prompt_tokens": 100.0, "gen_tokens": 50.0, "prefix_hits": 0.0, "prefix_queries": 0.0, "running": 0.0}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": model_id}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/metrics":
                with lock:
                    metrics = (
                        f'vllm:num_requests_running{{engine="0",model_name="{model_id}"}} {state["running"]}\n'
                        f'vllm:num_requests_waiting{{engine="0",model_name="{model_id}"}} 0.0\n'
                        f'vllm:kv_cache_usage_perc{{engine="0",model_name="{model_id}"}} 0.15\n'
                        f'vllm:prompt_tokens_total{{engine="0",model_name="{model_id}"}} {state["prompt_tokens"]}\n'
                        f'vllm:generation_tokens_total{{engine="0",model_name="{model_id}"}} {state["gen_tokens"]}\n'
                        f'vllm:prefix_cache_hits_total{{engine="0",model_name="{model_id}"}} {state["prefix_hits"]}\n'
                        f'vllm:prefix_cache_queries_total{{engine="0",model_name="{model_id}"}} {state["prefix_queries"]}\n'
                        'vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",'
                        'engine="0",gpu_memory_utilization="0.65",kv_cache_memory_bytes="None",'
                        'kv_cache_size_tokens="32768",num_cpu_blocks="128",num_gpu_blocks="2048"} 1.0\n'
                    ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(metrics)))
                self.end_headers()
                self.wfile.write(metrics)
                return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8", "replace"))
            with lock:
                state["running"] += 1.0
            try:
                max_tok = int(req.get("max_tokens", 16))
                with lock:
                    state["prompt_tokens"] += 300.0
                    state["gen_tokens"] += float(max_tok)
                    state["prefix_queries"] += 300.0
                    state["prefix_hits"] += 200.0
                body = json.dumps({
                    "id": "cmpl-selfcheck",
                    "choices": [{"text": "synthetic token output"}],
                    "usage": {"prompt_tokens": 300, "completion_tokens": max_tok, "total_tokens": 300 + max_tok},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with lock:
                    state["running"] = max(0.0, state["running"] - 1.0)

        def log_message(self, *args: Any) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.02), daemon=True)
    th.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2.0)


def run_capture_baseline(out_path: str, *, gpu_index: int) -> int:
    rc, stdout, stderr, _ = _run_gputaskman(["--once", "--json", "--gpu", str(gpu_index)])
    if rc != 0:
        print(f"ERROR: gputaskman --once --json failed (rc={rc}): {stderr.strip()}", file=sys.stderr)
        return 1
    snap = sanitize_value(json.loads(stdout))
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    print(f"BASELINE_CAPTURED {dest} vram_used={snap.get('vram_used')} vram_total={snap.get('vram_total')}")
    return 0


def run_self_check() -> int:
    """Offline self-check exercising config generation, stubs, evaluators, sanitization, and diff."""
    checks: list[dict[str, Any]] = []
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. Sanitization check
    dirty = {
        "path": "/home/alice/secret/vllm.log",
        "api_key": "sk-top-secret-123",
        "note": "key=sk-top-secret-123 at /home/alice/work",
    }
    cleaned = sanitize_value(dirty, secrets=("sk-top-secret-123",))
    _check(
        checks,
        "sanitize_redacts_home_and_secrets",
        "/home/alice" not in json.dumps(cleaned) and "sk-top-secret-123" not in json.dumps(cleaned),
        json.dumps(cleaned),
    )

    # 2. Config YAML roundtrip with gputaskman.config.load_config
    with tempfile.TemporaryDirectory(prefix="gputaskman_selfcheck_") as td:
        cfg_path = Path(td) / "config.yaml"
        cfg_path.write_text(
            build_config_yaml(
                auto_detect=False,
                url="http://127.0.0.1:8000",
                name="live-e2e",
                log_path="/tmp/vllm_e2e.log",
                gpu_index=0,
                api_key_env="VLLM_E2E_API_KEY",
                timeout_s=3.0,
            ),
            encoding="utf-8",
        )
        loaded = load_config(cfg_path)
        _check(
            checks,
            "config_yaml_roundtrip",
            loaded.vllm.auto_detect is False
            and len(loaded.vllm.endpoints) == 1
            and loaded.vllm.endpoints[0].url == "http://127.0.0.1:8000"
            and loaded.vllm.endpoints[0].api_key_env == "VLLM_E2E_API_KEY",
            str(loaded.redacted_dict()),
        )

    # 3. Healthy + fault local HTTP stubs
    with _local_healthy_stub(model_id) as port, _local_fault_stub(model_id) as fault_port:
        base = f"http://127.0.0.1:{port}"
        models = fetch_models(base)
        before_m = fetch_metrics_text(base)
        c_before = parse_vllm_prometheus_counters(before_m)
        comp = send_completion(base, model_id, _SHARED_PREFIX_UNIT * 4, max_tokens=32, timeout_s=5.0)
        after_m = fetch_metrics_text(base)
        c_after = parse_vllm_prometheus_counters(after_m)
        gauges = vllm_metrics(after_m, models)
        cache_cfg = vllm_cache_config(after_m)

        _check(checks, "stub_models_and_metrics", models == [model_id] and gauges["kv_cache_usage"] == 0.15, f"gauges={gauges}")
        _check(checks, "stub_cache_config", bool(cache_cfg and cache_cfg.get("enable_prefix_caching") is True), f"cache_cfg={cache_cfg}")
        _check(
            checks,
            "stub_completion_advances_counters",
            comp["http_status"] == 200
            and c_after["vllm:prompt_tokens_total"] > c_before["vllm:prompt_tokens_total"]
            and c_after["vllm:generation_tokens_total"] > c_before["vllm:generation_tokens_total"],
            f"before={c_before}, after={c_after}",
        )

        fault_status, _ = _http_request(f"http://127.0.0.1:{fault_port}/metrics", timeout_s=2.0)
        _check(checks, "fault_stub_returns_503", fault_status == 503, f"status={fault_status}")

    # 4. Synthetic scenario evaluators + real CLI gputaskman diff
    def _synth_snap(
        *,
        taken: float,
        vram_used: int,
        running: int | None,
        waiting: int | None,
        ratio: float | None,
        occ: int | None,
        avail: int | None,
        metrics_ts: str | None,
        match_source: str = "config",
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "taken": taken,
            "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
            "driver": "616.92",
            "cuda": "13.4",
            "adapter": {"luid": 1, "name": "NVIDIA GeForce RTX 4070 Laptop GPU", "vendor_id": 0x10DE},
            "vram_total": 8585740288,
            "vram_used": vram_used,
            "shared_used": None,
            "attributed": None,
            "backend": "wsl",
            "device": {"source": "NVML", "mem_used": vram_used, "mem_total": 8585740288},
            "processes": [
                {
                    "pid": 1234,
                    "host": None,
                    "name": "VLLM::EngineCore",
                    "kind": "LLM",
                    "runtimes": ["CUDA"],
                    "started_at": taken - 30.0,
                    "vram_attributed": False,
                    "vram_inferred_bytes": None,
                    "vram_inferred_by": None,
                    "vram_bytes": None,
                    "shared_bytes": None,
                    "committed_bytes": None,
                    "paged_out_bytes": None,
                    "vram_resident_bytes": None,
                    "gtt_resident_bytes": None,
                    "graphics_swapped_bytes": None,
                    "ports": [],
                    "model_files": [],
                    "llm": {
                        "segments": [],
                        "n_ctx": None,
                        "server": {"kind": "vllm", "url": "http://127.0.0.1:8000", "models": [{"name": model_id}]}
                        if ratio is not None
                        else None,
                        "server_source": match_source if ratio is not None else None,
                        "notes": notes or [],
                        "vllm_memory": {
                            "endpoint_url": "http://127.0.0.1:8000",
                            "endpoint_name": "live-e2e",
                            "match_source": match_source,
                            "model_names": [model_id] if ratio is not None else [],
                            "weights_bytes": 1_000_000_000,
                            "kv_allocated_bytes": 3_000_000_000,
                            "other_runtime_bytes": None,
                            "activation_peak_bytes": 250_000_000,
                            "activation_current_bytes": None,
                            "cuda_graph_bytes": 100_000_000,
                            "kv_pool_usage_ratio": ratio,
                            "kv_pool_occupied_bytes": occ,
                            "kv_pool_available_bytes": avail,
                            "kv_active_request_bytes": None,
                            "total_vram_used_bytes": None,
                            "device_vram_used_bytes": vram_used,
                            "total_vram_capacity_bytes": 8585740288,
                            "running_requests": running,
                            "waiting_requests": waiting,
                            "prefix_caching_enabled": True,
                            "stale": False,
                            "stale_reason": None,
                            "data_sources": {
                                "weights_bytes": "init_log:/tmp/vllm_e2e.log",
                                "kv_allocated_bytes": "init_log:/tmp/vllm_e2e.log",
                                "kv_pool_usage_ratio": "vllm:/metrics:kv_cache_usage_perc" if ratio is not None else "unavailable",
                            },
                            "measurement_timestamps": {
                                "device_vram_used_bytes": "2026-11-24T12:00:00+00:00",
                                "kv_pool_usage_ratio": metrics_ts,
                            },
                            "measurement_confidence": {
                                "weights_bytes": "init_profile",
                                "kv_allocated_bytes": "init_profile",
                                "kv_pool_usage_ratio": "measured" if ratio is not None else "unavailable",
                                "kv_pool_occupied_bytes": "derived" if occ is not None else "unavailable",
                                "activation_current_bytes": "unavailable",
                                "kv_active_request_bytes": "unavailable",
                            },
                            "notes": notes or [],
                        },
                    },
                }
            ],
        }

    base_snap = _synth_snap(
        taken=1795435200.0, vram_used=1_000_000_000, running=0, waiting=0, ratio=0.0, occ=0, avail=3_000_000_000,
        metrics_ts="2026-11-24T12:00:00+00:00",
    )
    base_snap["processes"] = []
    idle_snap = _synth_snap(
        taken=1795435210.0, vram_used=5_500_000_000, running=0, waiting=0, ratio=0.0, occ=0, avail=3_000_000_000,
        metrics_ts="2026-11-24T12:00:10+00:00",
    )
    active_snap = _synth_snap(
        taken=1795435212.0, vram_used=5_600_000_000, running=2, waiting=0, ratio=0.2, occ=600_000_000,
        avail=2_400_000_000, metrics_ts="2026-11-24T12:00:12+00:00",
    )
    auto_snap = _synth_snap(
        taken=1795435215.0, vram_used=5_500_000_000, running=0, waiting=0, ratio=0.0, occ=0, avail=3_000_000_000,
        metrics_ts="2026-11-24T12:00:15+00:00", match_source="parent",
    )
    fault_snap = _synth_snap(
        taken=1795435220.0, vram_used=5_500_000_000, running=None, waiting=None, ratio=None, occ=None, avail=None,
        metrics_ts=None, notes=["Configured vLLM API at http://127.0.0.1:8000 is unreachable; falling back."],
    )

    idle_text = "\n".join([
        "   vLLM Memory Breakdown  live-e2e (http://127.0.0.1:8000)  [config]",
        "     Weights: 954 MB  [Init profile]",
        "     Activation Current: N/A  [requires instrumentation]",
        "     KV Cache Allocated: 2.79 GB  [Init profile]",
        "     KV Pool Occupied: 0 MB (0.0%)  [derived; includes prefix cache]",
        "     Active Requests KV: N/A  [requires instrumentation]",
        "     Running / Waiting: 0 / 0",
        "     Sources: NVML, Init Profile, Metrics",
    ])

    s1 = evaluate_s1_baseline(idle_snap, base_snap)
    s2 = evaluate_s2_idle(idle_snap, idle_text, expected_url="http://127.0.0.1:8000", model_id=model_id, has_log=True)
    s3, _ = evaluate_s3_single_request(
        request_result={"http_status": 200, "elapsed_s": 0.4},
        counters_before={"vllm:prompt_tokens_total": 0.0, "vllm:generation_tokens_total": 0.0},
        counters_after={"vllm:prompt_tokens_total": 300.0, "vllm:generation_tokens_total": 128.0},
        polled_snapshots=[idle_snap, active_snap],
        idle_metrics_ts="2026-11-24T12:00:10+00:00",
        expected_url="http://127.0.0.1:8000",
        has_log=True,
    )
    s4 = evaluate_s4_concurrent(
        wave_results=[[{"http_status": 200}] * 4, [{"http_status": 200}] * 4],
        counters_before={"vllm:prefix_cache_hits_total": 0.0},
        counters_after={"vllm:prefix_cache_hits_total": 2400.0},
        polled_snapshots=[active_snap],
        expected_url="http://127.0.0.1:8000",
    )
    s5 = evaluate_s5_modes(
        config_snap=idle_snap, auto_rc=0, auto_snap=auto_snap, expected_url="http://127.0.0.1:8000", api_key_used=False
    )
    s6_checks = evaluate_s6_fault(
        case_name="fault_down",
        rc=0,
        elapsed_s=1.2,
        snap=fault_snap,
        expected_url="http://127.0.0.1:8000",
        expect_unreachable_note=True,
        has_log=True,
    )
    s7 = run_diff_check(base_snap, idle_snap)

    _check(checks, "eval_s1_pass", s1["status"] == "PASS", json.dumps(s1["checks"]))
    _check(checks, "eval_s2_pass", s2["status"] == "PASS", json.dumps(s2["checks"]))
    _check(checks, "eval_s3_pass", s3["status"] == "PASS", json.dumps(s3["checks"]))
    _check(checks, "eval_s4_pass", s4["status"] == "PASS", json.dumps(s4["checks"]))
    _check(checks, "eval_s5_pass", s5["status"] == "PASS", json.dumps(s5["checks"]))
    _check(checks, "eval_s6_pass", all(c["ok"] for c in s6_checks), json.dumps(s6_checks))
    _check(checks, "eval_s7_diff_cli_pass", s7["status"] == "PASS", json.dumps(s7["checks"]))

    # 5. Verify stdout summary, stderr failures, and missing-log error all sanitize home paths & secrets
    dirty_secret = "sk-secret-e2e-999"
    dirty_ev = sanitize_value(
        {
            "provenance": "measured",
            "captured_at": "2026-11-24T12:00:00+00:00",
            "gpu": {"name": "RTX 4070"},
            "topology": {"backend": "wsl"},
            "model_id": model_id,
            "overall_status": "FAIL",
            "scenarios": {
                "S2_idle": {
                    "status": "FAIL",
                    "duration_s": 0.5,
                    "observed": {
                        "log_source": "init_log:/home/alice/vllm.log",
                        "debug_note": f"token={dirty_secret} path=/home/alice/vllm.log",
                    },
                    "checks": [
                        {
                            "name": "s2_init_log",
                            "ok": False,
                            "detail": f"failed reading /home/alice/vllm.log with key {dirty_secret}",
                        }
                    ],
                }
            },
        },
        secrets=(dirty_secret,),
    )
    summary_text = json.dumps(_build_verify_summary(dirty_ev), indent=2)
    failure_text = "\n".join(_format_verify_failures(dirty_ev))
    combined_out = summary_text + "\n" + failure_text
    _check(
        checks,
        "verify_summary_and_failures_sanitized",
        "/home/alice" not in combined_out
        and dirty_secret not in combined_out
        and "/home/<user>/vllm.log" in summary_text
        and "/home/<user>/vllm.log" in failure_text
        and "***" in summary_text
        and "***" in failure_text,
        combined_out,
    )

    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        missing_rc = run_verify(
            argparse.Namespace(
                url="http://127.0.0.1:8000",
                log="/home/alice/vllm.log",
                model=None,
                api_key=dirty_secret,
                baseline_json=None,
                evidence_out=None,
                gpu=0,
                timeout_s=5.0,
            )
        )
    missing_err = err_buf.getvalue()
    _check(
        checks,
        "verify_missing_log_error_sanitized",
        missing_rc == 1
        and "/home/alice" not in missing_err
        and dirty_secret not in missing_err
        and "/home/<user>/vllm.log" in missing_err,
        f"rc={missing_rc}, stderr={missing_err.strip()}",
    )

    passed = all(c["ok"] for c in checks)
    print(json.dumps({"self_check": "PASS" if passed else "FAIL", "checks_passed": sum(c["ok"] for c in checks), "checks_total": len(checks)}, indent=2))
    if passed:
        print("LIVE_E2E_SELF_CHECK_OK")
        return 0
    for c in checks:
        if not c["ok"]:
            print(f"FAIL {c['name']}: {c['detail']}", file=sys.stderr)
    return 1


def _build_verify_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    """Build the stdout summary strictly from the already-sanitized evidence dict."""
    return {
        "provenance": evidence["provenance"],
        "captured_at": evidence["captured_at"],
        "gpu": evidence["gpu"],
        "topology": evidence["topology"],
        "model_id": evidence["model_id"],
        "overall_status": evidence["overall_status"],
        "scenarios": {
            k: {
                "status": v["status"],
                "observed": v["observed"],
                "duration_s": v["duration_s"],
            }
            for k, v in evidence["scenarios"].items()
        },
    }


def _format_verify_failures(evidence: dict[str, Any]) -> list[str]:
    """Format failing check lines strictly from the already-sanitized evidence dict."""
    lines: list[str] = []
    for sc_name, sc in evidence["scenarios"].items():
        if sc["status"] != "PASS":
            for chk in sc["checks"]:
                if not chk["ok"]:
                    lines.append(f"FAIL {sc_name}.{chk['name']}: {chk['detail']}")
    return lines


def run_verify(args: argparse.Namespace) -> int:
    url = (args.url or os.environ.get("VLLM_E2E_URL") or "http://127.0.0.1:8000").rstrip("/")
    log_path = args.log or os.environ.get("VLLM_E2E_LOG") or None
    model_arg = args.model or os.environ.get("VLLM_E2E_MODEL") or None
    baseline_path = args.baseline_json or os.environ.get("VLLM_E2E_BASELINE_JSON") or None
    evidence_out = args.evidence_out or os.environ.get("VLLM_E2E_EVIDENCE_OUT") or None
    api_key = args.api_key or os.environ.get("VLLM_E2E_API_KEY") or None
    api_key_env = "VLLM_E2E_API_KEY" if api_key else None
    secrets = (api_key,) if api_key else ()

    if log_path and not Path(log_path).is_file():
        print(
            sanitize_value(f"ERROR: VLLM_E2E_LOG file does not exist: {log_path}", secrets=secrets),
            file=sys.stderr,
        )
        return 1

    baseline_snap: dict[str, Any] | None = None
    if baseline_path:
        baseline_snap = json.loads(Path(baseline_path).read_text(encoding="utf-8"))

    # Pre-flight: check /v1/models and /metrics reachability.
    try:
        served_models = fetch_models(url, api_key=api_key, timeout_s=min(10.0, args.timeout_s))
    except Exception as exc:
        print(
            sanitize_value(f"ERROR: Live vLLM endpoint {url} is unreachable (/v1/models): {exc}", secrets=secrets),
            file=sys.stderr,
        )
        return 1
    model_id = model_arg or served_models[0]
    if model_id not in served_models:
        print(f"ERROR: Requested model {model_id!r} not in /v1/models ({served_models})", file=sys.stderr)
        return 1
    try:
        _ = fetch_metrics_text(url, timeout_s=min(10.0, args.timeout_s))
    except Exception as exc:
        print(
            sanitize_value(f"ERROR: Live vLLM /metrics at {url} failed: {exc}", secrets=secrets),
            file=sys.stderr,
        )
        return 1

    env_with_key = dict(os.environ)
    if api_key:
        env_with_key["VLLM_E2E_API_KEY"] = api_key

    scenarios: dict[str, Any] = {}
    sanitized_snapshots: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="gputaskman_live_e2e_") as td:
        td_path = Path(td)
        cfg_path = td_path / "live_config.yaml"
        cfg_path.write_text(
            build_config_yaml(
                auto_detect=False,
                url=url,
                name="live-e2e",
                log_path=log_path,
                gpu_index=args.gpu,
                api_key_env=api_key_env,
                timeout_s=3.0,
            ),
            encoding="utf-8",
        )

        # S2 (and S1 current snapshot): CLI --once --json and --once text with config.yaml
        rc_idle_js, out_idle_js, err_idle_js, _ = _run_gputaskman(
            ["--once", "--json", "--gpu", str(args.gpu), "--config", str(cfg_path)],
            env=env_with_key,
            timeout_s=args.timeout_s,
        )
        if rc_idle_js != 0:
            print(
                sanitize_value(
                    f"ERROR: gputaskman --once --json --config failed (rc={rc_idle_js}): {err_idle_js.strip()}",
                    secrets=secrets,
                ),
                file=sys.stderr,
            )
            return 1
        idle_snap = json.loads(out_idle_js)

        rc_idle_txt, out_idle_txt, err_idle_txt, _ = _run_gputaskman(
            ["--once", "--gpu", str(args.gpu), "--config", str(cfg_path)],
            env=env_with_key,
            timeout_s=args.timeout_s,
        )
        if rc_idle_txt != 0:
            print(
                sanitize_value(
                    f"ERROR: gputaskman --once --config failed (rc={rc_idle_txt}): {err_idle_txt.strip()}",
                    secrets=secrets,
                ),
                file=sys.stderr,
            )
            return 1

        scenarios["S1_baseline"] = evaluate_s1_baseline(idle_snap, baseline_snap)
        scenarios["S2_idle"] = evaluate_s2_idle(
            idle_snap, out_idle_txt, expected_url=url, model_id=model_id, has_log=bool(log_path)
        )
        sanitized_snapshots["idle"] = sanitize_value(idle_snap, secrets=secrets)
        if baseline_snap is not None:
            sanitized_snapshots["baseline"] = sanitize_value(baseline_snap, secrets=secrets)

        idle_row = find_vllm_row(idle_snap, expected_url=url)
        idle_metrics_ts = (
            (idle_row["llm"]["vllm_memory"].get("measurement_timestamps") or {}).get("kv_pool_usage_ratio")
            if idle_row
            else None
        )

        # S3 & S4: use Collector with config.yaml for 0.25s active-inference polling
        app_cfg = load_config(cfg_path, env=env_with_key)
        prev_key = os.environ.get("VLLM_E2E_API_KEY")
        if api_key:
            os.environ["VLLM_E2E_API_KEY"] = api_key
        collector = Collector(gpu_index=args.gpu, scan_budget=None, host=False, config=app_cfg)
        try:
            # S3: Single request (~300 tokens prompt, 192 output tokens)
            s3_counters_before = parse_vllm_prometheus_counters(fetch_metrics_text(url, timeout_s=5.0))
            s3_prompt = (_SHARED_PREFIX_UNIT * 5) + "\nProvide a detailed step-by-step explanation of paged attention."
            s3_stop = threading.Event()
            s3_result_box: list[dict[str, Any]] = []

            def _s3_worker() -> None:
                try:
                    s3_result_box.append(
                        send_completion(
                            url, model_id, s3_prompt, max_tokens=192, api_key=api_key, timeout_s=args.timeout_s
                        )
                    )
                except Exception as exc:
                    s3_result_box.append({"http_status": 0, "error": str(exc), "elapsed_s": 0.0})
                finally:
                    s3_stop.set()

            th_s3 = threading.Thread(target=_s3_worker, daemon=True)
            th_s3.start()
            s3_samples = _poll_collector_samples(collector, s3_stop, expected_url=url, interval_s=0.25)
            th_s3.join(timeout=args.timeout_s)
            s3_counters_after = parse_vllm_prometheus_counters(fetch_metrics_text(url, timeout_s=5.0))
            s3_res = s3_result_box[0] if s3_result_box else {"http_status": 0, "error": "no result"}

            s3_eval, s3_best_snap = evaluate_s3_single_request(
                request_result=s3_res,
                counters_before=s3_counters_before,
                counters_after=s3_counters_after,
                polled_snapshots=s3_samples,
                idle_metrics_ts=idle_metrics_ts,
                expected_url=url,
                has_log=bool(log_path),
            )
            scenarios["S3_single_request"] = s3_eval
            if s3_best_snap is not None:
                sanitized_snapshots["single_request_peak"] = sanitize_value(s3_best_snap, secrets=secrets)

            # S4: 2 waves x 4 concurrent requests with ~700-token shared prefix
            s4_counters_before = parse_vllm_prometheus_counters(fetch_metrics_text(url, timeout_s=5.0))
            shared_prefix = _SHARED_PREFIX_UNIT * 12  # ~650-750 tokens
            s4_stop = threading.Event()
            wave_results: list[list[dict[str, Any]]] = []

            def _s4_worker() -> None:
                try:
                    for wave_idx in range(2):
                        prompts = [
                            f"{shared_prefix}\nWave {wave_idx + 1} Request {req_idx + 1}: "
                            f"Explain block allocation and KV cache reuse for batch slot {req_idx + 1}."
                            for req_idx in range(4)
                        ]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                            futs = [
                                pool.submit(
                                    send_completion,
                                    url,
                                    model_id,
                                    p,
                                    max_tokens=128,
                                    api_key=api_key,
                                    timeout_s=args.timeout_s,
                                )
                                for p in prompts
                            ]
                            wave_out: list[dict[str, Any]] = []
                            for fut in futs:
                                try:
                                    wave_out.append(fut.result())
                                except Exception as exc:
                                    wave_out.append({"http_status": 0, "error": str(exc)})
                            wave_results.append(wave_out)
                finally:
                    s4_stop.set()

            th_s4 = threading.Thread(target=_s4_worker, daemon=True)
            th_s4.start()
            s4_samples = _poll_collector_samples(collector, s4_stop, expected_url=url, interval_s=0.25)
            th_s4.join(timeout=args.timeout_s * 3)
            s4_counters_after = parse_vllm_prometheus_counters(fetch_metrics_text(url, timeout_s=5.0))

            scenarios["S4_concurrent_prefix_cache"] = evaluate_s4_concurrent(
                wave_results=wave_results,
                counters_before=s4_counters_before,
                counters_after=s4_counters_after,
                polled_snapshots=s4_samples,
                expected_url=url,
            )
            if s4_samples:
                peak_s4_snap = max(
                    s4_samples,
                    key=lambda s: (
                        ((find_vllm_row(s, expected_url=url) or {}).get("llm") or {})
                        .get("vllm_memory", {})
                        .get("kv_pool_occupied_bytes")
                        or -1
                    ),
                )
                sanitized_snapshots["concurrent_peak"] = sanitize_value(peak_s4_snap, secrets=secrets)
        finally:
            collector.close()
            if api_key:
                if prev_key is None:
                    os.environ.pop("VLLM_E2E_API_KEY", None)
                else:
                    os.environ["VLLM_E2E_API_KEY"] = prev_key

        # S5: Zero-config Auto-detect mode (empty cwd, no GPUTASKMAN_CONFIG, clean HOME)
        empty_cwd = td_path / "empty_cwd"
        empty_home = td_path / "empty_home"
        empty_cwd.mkdir()
        empty_home.mkdir()
        auto_env = dict(os.environ)
        auto_env.pop("GPUTASKMAN_CONFIG", None)
        auto_env["HOME"] = str(empty_home)
        rc_auto, out_auto, _, _ = _run_gputaskman(
            ["--once", "--json", "--gpu", str(args.gpu)],
            cwd=empty_cwd,
            env=auto_env,
            timeout_s=args.timeout_s,
        )
        auto_snap = json.loads(out_auto) if rc_auto == 0 else None
        scenarios["S5_modes"] = evaluate_s5_modes(
            config_snap=idle_snap,
            auto_rc=rc_auto,
            auto_snap=auto_snap,
            expected_url=url,
            api_key_used=bool(api_key),
        )
        if auto_snap is not None:
            sanitized_snapshots["auto_detect"] = sanitize_value(auto_snap, secrets=secrets)

        # S6: API outage / fault fallback (6a unused port + 6b HTTP 503 malformed /metrics)
        s6_t0 = time.perf_counter()
        s6_checks: list[dict[str, Any]] = []
        s6_observed: dict[str, Any] = {}

        unused_port = _unused_loopback_port()
        down_url = f"http://127.0.0.1:{unused_port}"
        fault_down_cfg = td_path / "fault_down.yaml"
        fault_down_cfg.write_text(
            build_config_yaml(
                auto_detect=False,
                url=down_url,
                name="fault-down",
                log_path=log_path,
                gpu_index=args.gpu,
                timeout_s=0.5,
            ),
            encoding="utf-8",
        )
        rc_down, out_down, _, elapsed_down = _run_gputaskman(
            ["--once", "--json", "--gpu", str(args.gpu), "--config", str(fault_down_cfg)],
            timeout_s=10.0,
        )
        snap_down = json.loads(out_down) if rc_down == 0 else None
        s6_checks.extend(
            evaluate_s6_fault(
                case_name="unused_port",
                rc=rc_down,
                elapsed_s=elapsed_down,
                snap=snap_down,
                expected_url=down_url,
                expect_unreachable_note=True,
                has_log=bool(log_path),
            )
        )
        s6_observed["unused_port_elapsed_s"] = round(elapsed_down, 3)

        with _local_fault_stub(model_id) as stub_port:
            stub_url = f"http://127.0.0.1:{stub_port}"
            fault_503_cfg = td_path / "fault_503.yaml"
            fault_503_cfg.write_text(
                build_config_yaml(
                    auto_detect=False,
                    url=stub_url,
                    name="fault-503",
                    log_path=log_path,
                    gpu_index=args.gpu,
                    timeout_s=1.0,
                ),
                encoding="utf-8",
            )
            rc_503, out_503, _, elapsed_503 = _run_gputaskman(
                ["--once", "--json", "--gpu", str(args.gpu), "--config", str(fault_503_cfg)],
                timeout_s=10.0,
            )
            snap_503 = json.loads(out_503) if rc_503 == 0 else None
        s6_checks.extend(
            evaluate_s6_fault(
                case_name="metrics_503",
                rc=rc_503,
                elapsed_s=elapsed_503,
                snap=snap_503,
                expected_url=stub_url,
                expect_unreachable_note=False,
                has_log=bool(log_path),
            )
        )
        s6_observed["metrics_503_elapsed_s"] = round(elapsed_503, 3)

        scenarios["S6_fault_fallback"] = {
            "status": "PASS" if all(c["ok"] for c in s6_checks) else "FAIL",
            "checks": s6_checks,
            "observed": s6_observed,
            "duration_s": round(time.perf_counter() - s6_t0, 3),
        }

        # S7: gputaskman diff compatibility across two real snapshots
        diff_before = baseline_snap if baseline_snap is not None else idle_snap
        diff_after = idle_snap if baseline_snap is not None else (s3_best_snap or idle_snap)
        scenarios["S7_diff_compat"] = run_diff_check(diff_before, diff_after)

    overall_pass = all(sc["status"] == "PASS" for sc in scenarios.values())
    evidence = sanitize_value(
        {
            "schema_version": 1,
            "provenance": "measured",
            "captured_at": _utc_now_iso(),
            "gpu": {
                "name": idle_snap.get("gpu"),
                "driver": idle_snap.get("driver"),
                "cuda": idle_snap.get("cuda"),
                "backend": idle_snap.get("backend"),
                "vram_total_bytes": idle_snap.get("vram_total"),
            },
            "topology": {
                "backend": idle_snap.get("backend"),
                "wsl_host_status": idle_snap.get("host_status"),
                "auto_detect_mode": scenarios["S5_modes"]["observed"].get("auto_detect_topology"),
            },
            "model_id": model_id,
            "overall_status": "PASS" if overall_pass else "FAIL",
            "scenarios": scenarios,
            "sanitized_snapshots": sanitized_snapshots,
        },
        secrets=secrets,
    )

    if evidence_out:
        out_file = Path(evidence_out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    summary = _build_verify_summary(evidence)
    print(json.dumps(summary, indent=2))
    if overall_pass:
        print("LIVE_E2E_VLLM_PASS")
        return 0
    for line in _format_verify_failures(evidence):
        print(line, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="live_e2e_vllm.py",
        description="Live E2E verification harness for gputaskman Strategy B vLLM GPU Memory Breakdown.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="Run the 6 live vLLM E2E scenarios + diff compatibility check.")
    mode.add_argument("--self-check", action="store_true", help="Run offline self-check with local HTTP stubs.")
    mode.add_argument("--capture-baseline", metavar="PATH", help="Capture a pre-vLLM baseline snapshot JSON to PATH.")

    parser.add_argument("--url", help="vLLM base URL (default: VLLM_E2E_URL or http://127.0.0.1:8000).")
    parser.add_argument("--log", help="Path to vLLM startup log file (default: VLLM_E2E_LOG).")
    parser.add_argument("--model", help="Served model ID (default: VLLM_E2E_MODEL or auto-queried from /v1/models).")
    parser.add_argument("--api-key", help="Optional Bearer API key (default: VLLM_E2E_API_KEY).")
    parser.add_argument("--baseline-json", help="Optional pre-vLLM baseline snapshot JSON path (default: VLLM_E2E_BASELINE_JSON).")
    parser.add_argument("--evidence-out", help="Optional path to write full sanitized evidence JSON (default: VLLM_E2E_EVIDENCE_OUT).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to verify (default: 0).")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP/subprocess timeout in seconds (default: 30.0).")
    args = parser.parse_args(argv)

    if args.self_check:
        return run_self_check()
    if args.capture_baseline:
        return run_capture_baseline(args.capture_baseline, gpu_index=args.gpu)
    return run_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
