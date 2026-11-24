"""Assembles one Snapshot from Windows counters, NVML or DRM, process inspection and LLM introspection."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from contextlib import ExitStack
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import psutil

if TYPE_CHECKING:
    from .config import AppConfig, VllmEndpointConfig

from .amdsmi import AmdSmi, amdsmi_status
from .adl import Adl
from .dxgi import (VENDOR_AMD, VENDOR_NVIDIA, VENDOR_HUAWEI, VENDOR_INTEL, VENDOR_QUALCOMM, Adapter,
                  adapter_driver_version)
from .handy import HandyEvidence, inspect_handy
from .llm import (LlmView, ServerInfo, ServerProber, VllmInitProfile, api_key, build_view, gb,
                  ollama_log_path, parse_vllm_init_log)
from . import nvml as nvml_mod
from .nvml import Nvml, NvDevice
from .pdh import ENGINE_UNAVAILABLE_REASON, ProcGpu
from .procinfo import AI_KINDS, ProcDetail, ProcInspector, classify, target_path, vllm_launch_info
from .tcc import TccAdapter, TccCounters, _pci_identity, tcc_adapters
from .telemetry import DeviceTelemetry

if sys.platform == "win32":
    from . import dxgi
    from .pdh import GpuCounters

    BACKEND = "wddm"

    def enumerate_adapters(*, strict: bool = False) -> list[Adapter]:
        """DXGI's adapters plus the TCC-mode NVIDIA GPUs it does not list."""
        listed = dxgi.enumerate_adapters(strict=strict)  # a strict DXGI failure raises before NVML is asked
        return listed + tcc_adapters(listed)
elif sys.platform == "darwin":
    from .macos import MacCounters as GpuCounters
    from .macos import enumerate_adapters, macos_version

    BACKEND = "metal"
else:
    from .linux import LinuxCounters as GpuCounters
    from .linux import enumerate_adapters, is_wsl

    BACKEND = "wsl" if is_wsl() else "nvml"

from .wslhost import ContextTracker, HostStream, interop_enabled, match_adapter

MiB = 1 << 20


@dataclass
class ProcRow:
    pid: int
    name: str
    kind: str
    gpu: ProcGpu
    detail: ProcDetail
    llm: LlmView | None = None
    handy: HandyEvidence | None = None
    host: bool = False  # pid is from another PID namespace, never a local one (see origin)
    origin: str = ""  # with host: "windows" | "outside" (legacy) | "nvml" (host PID, local owner unresolved)
    gpu_index: int | None = None
    ram: int | None = None  # system RAM (working set) from host.py; None = unknown, not zero
    cpu: float | None = None  # % of the whole machine from host.py; None until the second sample

    @property
    def key(self) -> str:
        """Unique row id: PIDs from other namespaces and local PIDs can collide."""
        pid = f"{self.origin[0]}{self.pid}" if self.host else str(self.pid)
        return f"{self.gpu_index}:{pid}" if self.gpu_index is not None else pid

    @property
    def pid_label(self) -> str:
        if self.origin == "config":
            return f"C:{self.pid}"
        return f"{'W' if self.origin == 'windows' else 'H'}:{self.pid}" if self.host else str(self.pid)

    @property
    def runtime(self) -> str:
        return "+".join(self.detail.runtimes[:3])

    @property
    def inside(self) -> str:
        """One-line 'what's in there' summary for the process table."""
        if self.handy:
            if self.handy.last_model:
                name = self.handy.last_model.replace("\\", "/").rsplit("/", 1)[-1]
                return f"Handy last log ({self.handy.last_event}): {name}"
            if self.handy.selected_model:
                name = self.handy.selected_model.replace("\\", "/").rsplit("/", 1)[-1]
                return f"Handy selected (not load proof): {name}"
        if (self.llm and self.llm.server and self.llm.server.kind in ("vllm", "foundry-local")
                and self.llm.server.models):
            source = {"parent": "parent API", "config": "config API"}.get(self.llm.server_source, "API")
            return f"reported ({source}): " + ", ".join(m["name"] for m in self.llm.server.models)
        if self.llm and self.llm.models:
            parts = []
            for i, g in enumerate(self.llm.models):
                if g.kind in ("vision projector", "LoRA adapter"):
                    parts.append(f"+{g.kind}")
                else:
                    parts.append(f"{self.llm.label(i)} {g.params_label} {g.quant}")
            return ", ".join(parts)
        if self.detail.models:
            models = sorted(self.detail.models, key=lambda m: not m.has_model_evidence)
            return ", ".join(("" if m.has_model_evidence else "path-arg (unproven): ") +
                             (m.store_name or os.path.basename(m.path)) +
                             (f" ({m.kind_label})" if m.kind_hint else "") for m in models[:2])
        if self.detail.candidates:
            return "installed (unproven): " + ", ".join(
                f"{m.store_name or os.path.basename(m.path)} {gb(m.size)}" +
                (f" ({m.kind_label})" if m.kind_hint else "") for m in self.detail.candidates[:2])
        return {
            "compositor": "desktop back buffers + composed frames",
            "wsl-vm": "all Linux GPU memory (every WSL distro + WSLg)",
            "external": "a process in another PID namespace (host or another container)",
            "kernel": "driver/kernel allocations",
            "browser": "page textures, video, canvas/WebGL",
            "video": "video frames (decode/encode)",
            "3D/game": "textures, meshes, render targets",
        }.get(self.kind, "")


@dataclass
class Snapshot:
    taken: float
    adapter: Adapter | None
    adapters: list[Adapter]
    nv: DeviceTelemetry | None  # legacy name; may contain telemetry from either provider
    driver: str
    cuda: str
    vram_total: int
    vram_used: int | None  # WDDM adapter-wide dedicated usage
    shared_used: int | None
    procs: list[ProcRow] = field(default_factory=list)
    backend: str = "wddm"  # wddm (Windows) | nvml (native Linux, NVIDIA) | drm (native Linux, AMD/Intel) | ascend (Huawei NPU) | wsl (device-wide only) | metal (macOS)
    process_status: str = ""  # amdgpu only: what the DRM process scan couldn't cover; empty when it saw everything
    host_status: str = ""  # WSL only: Windows host counters via interop -- ok | starting | off (...) | ...
    wsl_split: str = ""  # WSL only: why Linux rows have no (inferred) VRAM, when they don't
    # False on an APU (amdgpu iGPU): its GTT is ordinary GPU memory, not spill from a full VRAM.
    shared_is_spill: bool = True
    counter_status: str = ""  # Counter availability or adapter matching explanation
    counter_adapter_matched: bool | None = None  # Windows PDH LUID matched the selected DXGI adapter
    process_memory_available: bool = True
    nv_lost: bool = False  # NVML faulted for this GPU after a disable/re-enable; restart restores it
    @property
    def host_data(self) -> bool:
        return self.host_status.startswith("ok")

    @property
    def attributed(self) -> int | None:
        if not self.process_memory_available or self.counter_adapter_matched is False or any(
                ("dedicated" not in p.gpu.available or (self.backend == "nvml" and not p.gpu.attributed))
                for p in self.procs if not p.gpu.inferred_by and p.origin != "config"):
            return None
        total = int(sum(p.gpu.dedicated for p in self.procs if p.gpu.attributed))
        if self.backend == "wsl" and self.host_data and self.vram_total and total > self.vram_total:
            return None  # overlapping host counters cannot establish uniquely attributed bytes
        return total


def _is_live(g: ProcGpu) -> bool:
    # graphics_swapped is macOS compressed graphics memory and is not added to dedicated.
    return (
        bool(g.engine_reason) or not g.attributed or g.dedicated >= MiB or g.gpu_pct >= 0.5
        or (g.wddm_dedicated_usage or 0) >= MiB or (g.wddm_shared_usage or 0) >= 16 * MiB
        or g.shared >= 16 * MiB or g.paged_out >= 16 * MiB or (g.graphics_swapped or 0) >= 16 * MiB
    )


def select_adapter(adapters: list[Adapter], gpu_index: int) -> Adapter | None:
    choices = ordered_adapters(adapters)
    if gpu_index < 0:
        raise ValueError("--gpu must be a non-negative index")
    if gpu_index >= len(choices):
        if not choices and gpu_index == 0:
            return None
        raise ValueError(f"--gpu {gpu_index} is out of range; {len(choices)} GPU adapter(s) available")
    return choices[gpu_index]


def ordered_adapters(adapters: list[Adapter]) -> list[Adapter]:
    """List all hardware adapters; retain NVIDIA-first ordering for existing users. TCC GPUs come last, so
    adding one never changes the index of a GPU that Windows lists."""
    return sorted((a for a in adapters if not a.software),
                  key=lambda a: (isinstance(a, TccAdapter), a.vendor_id != VENDOR_NVIDIA, -a.dedicated_vram, a.luid))


def _wddm_bdf(pci_bus) -> tuple[int, int, int] | None:
    """An NVML PCI address in WDDM's ADAPTERADDRESS form. Windows puts the PCI segment above the 8-bit
    bus (segment 15 bus 1 is bus 0xF01), so domain-0 addresses are unchanged."""
    identity = _pci_identity(pci_bus)
    if identity is None:
        return None
    domain, (bus, device, function) = identity
    return domain << 8 | bus, device, function


def select_nv_device(adapter: Adapter | None, devices: list[NvDevice], gpu_index: int,
                     adapters: list[Adapter] | None = None) -> NvDevice | None:
    if adapter:
        if adapter.vendor_id != VENDOR_NVIDIA:
            return None
        if sys.platform != "win32":
            from .linux import pci_key
            matches = [d for d in devices if d.pci_bus and pci_key(d.pci_bus) == adapter.luid]
            return matches[0] if len(matches) == 1 else None
        if isinstance(adapter, TccAdapter):
            matches = [d for d in devices if adapter.matches_pci(d.pci_bus) and d.driver_model == adapter.driver_model]
            return matches[0] if len(matches) == 1 else None
        if adapter.pci_address is None:
            return None
        # MCDM telemetry is ineligible for a DXGI adapter below, so its known compute-only
        # adapter cannot make a WDDM peer at the same BDF in another PCI domain ambiguous.
        if adapters is not None and sum(a.pci_address == adapter.pci_address for a in adapters
                                        if not (isinstance(a, TccAdapter) and a.driver_model == "MCDM")) != 1:
            return None
        matches = [d for d in devices if _wddm_bdf(d.pci_bus) == adapter.pci_address and d.driver_model != "MCDM"]
        if len(matches) == 1:
            return matches[0]
        return None
    matches = [d for d in devices if d.index == gpu_index]
    return matches[0] if len(matches) == 1 else None


_VLLM_EPHEMERAL_PORT_MIN = 32768


def _is_vllm_ephemeral_worker_ports(detail: ProcDetail) -> bool:
    """True when a non-ambiguous vLLM worker listens only on multiple ephemeral IPC/ZMQ ports."""
    return (
        detail.vllm_role == "worker"
        and not detail.vllm_ambiguous
        and len(detail.ports) > 1
        and all(p >= _VLLM_EPHEMERAL_PORT_MIN for p in detail.ports)
    )


def _llm_wanted(row, shared_is_spill: bool, config: AppConfig | None = None) -> bool:
    """Whether a process row gets an LLM breakdown: an AI-kind process with models (or kind LLM),
    not diffusion, and with measured GPU memory. Where spilled system memory is a counter of its own
    (WDDM, discrete DRM cards) both the dedicated and shared counters must be present; on an integrated
    GPU system memory is the memory and there is no spill counter, so dedicated alone suffices.
    Independent model evidence or a recognized LLM runtime can still supply metadata or API discovery
    on an inferred-owner row, without a measured memory budget. Foundry Local and vLLM can report
    metadata/breakdown without counters or mapped files; config.yaml match_pid also opts a row in."""
    if row.detail.is_diffusion:
        return False
    if row.detail.foundry_local or getattr(row.detail, "vllm_role", ""):
        return True
    if config is not None and config.vllm.endpoints:
        if any(ep.match_pid == row.pid for ep in config.vllm.endpoints):
            return True
        if getattr(row.detail, "name", "").startswith("VLLM::"):
            return True
    if row.kind not in AI_KINDS or not (row.detail.has_model_evidence or row.kind == "LLM"):
        return False
    if not row.gpu.attributed and row.gpu.inferred_by:
        return True
    need = {"dedicated", "shared"} if shared_is_spill else {"dedicated"}
    return need <= row.gpu.available


class Collector:
    DEEP_TOP_N = 25  # budgeted scans: the top N VRAM users plus anything busy
    _mac_driver = ""  # "macOS 26.3.1" on the metal backend, read once at startup
    SCAN_BUDGET = 0.15  # seconds of memory-map scanning per sample; leftovers wait for the next sample
    drm_card = None  # the selected GPU's AMD or Intel card, when it is one (drm.DrmCard)
    ascend = None  # AscendCounters, when LinuxCounters owns one
    ascend_chip = None  # the selected GPU's Ascend chip, when it is one (ascend.AscendChip)
    config: AppConfig | None = None  # optional loaded AppConfig; None = zero-config auto-detect

    @property
    def backend(self) -> str:
        if BACKEND == "nvml" and self.adapter and self.adapter.vendor_id == VENDOR_HUAWEI:
            return "ascend"  # Huawei NPU: npu-smi is the source, not NVML/DRM
        return "drm" if self.drm_card is not None else BACKEND

    def __init__(self, gpu_index: int = 0, scan_budget: float | None = SCAN_BUDGET, host: bool = True,
                 adapters: list[Adapter] | None = None, config: AppConfig | None = None) -> None:
        self.config = config
        self.adapters = [a for a in (enumerate_adapters() if adapters is None else adapters) if not a.software]
        self.adapter = select_adapter(self.adapters, gpu_index)
        self._windows_driver = (adapter_driver_version(self.adapter.device_instance_id)
                                if BACKEND == "wddm" and self.adapter else "")
        with ExitStack() as resources:
            if isinstance(self.adapter, TccAdapter):  # no Windows counters exist for it: NVML is the source
                self.nvml = Nvml()
                resources.callback(self.nvml.close)
                self.counters = TccCounters(self.adapter, self.nvml)
                resources.callback(self.counters.close)
            else:
                self.counters = GpuCounters()
                resources.callback(self.counters.close)
                self.nvml = Nvml()
                resources.callback(self.nvml.close)
            self.amdsmi = AmdSmi(self.adapter, self.adapters)
            resources.callback(self.amdsmi.close)
            self.adl = Adl(self.adapter, self.adapters)
            resources.callback(self.adl.close)
            resources.pop_all()
        self._mac_driver = f"macOS {macos_version()}".strip() if BACKEND == "metal" else ""
        # Native Linux on an AMD or Intel GPU: device metrics from the driver's sysfs (Intel's are
        # partial and get their fdinfo-derived used-memory stamp), processes from DRM fdinfo.
        drm = getattr(self.counters, "drm", None)
        if drm is not None and self.adapter and self.adapter.vendor_id != VENDOR_NVIDIA:
            self.drm_card = next((c for c in drm.cards if c.key == self.adapter.luid
                                  and c.vendor_id == self.adapter.vendor_id), None)
        if drm is not None and self.drm_card is None:
            self.counters.drm = None  # an NVIDIA GPU is shown: don't walk every process's fds for nothing
        elif drm is not None:
            drm.start(self.drm_card)  # process baseline plus device counters for the selected GPU only
            if scan_budget is None:
                drm.scan_budget = None  # one-shot output sees every process
        # Huawei Ascend NPU (no DRM): one selected chip and its owned watch process.
        self.ascend_chip = None
        self.ascend = None
        if BACKEND == "nvml" and self.adapter and self.adapter.vendor_id == VENDOR_HUAWEI:
            from dataclasses import replace
            from .ascend import AscendCounters, cached_chips, chip_adapter_key

            chip = next((c for c in cached_chips()
                         if chip_adapter_key(c.pdev, c.chip) == self.adapter.luid), None)
            if chip is not None:
                self.ascend_chip = replace(chip)
                self.ascend = AscendCounters(chips=[self.ascend_chip])
                self.counters.ascend = self.ascend
        self.inspector = ProcInspector()
        self.prober = ServerProber()
        self.scan_budget = scan_budget  # None = scan every eligible local process now (one-shot output)
        self.gpu_index = gpu_index
        # WSL: Linux can't see WSL's own VRAM, but the Windows host can. Ask it when interop is enabled.
        self.host: HostStream | None = None
        self.host_status = ""
        if BACKEND == "wsl":
            if not host:
                self.host_status = "off (--no-host)"
            elif not interop_enabled():
                self.host_status = "off (WSL interop is disabled)"
            else:
                self.host = HostStream()
        self._host_luid: int | None = None  # host adapter for this GPU, matched once and kept
        self.tracker = ContextTracker(self_pid=os.getpid()) if self.host is not None else None
        self._own_start = time.time()  # our NVML context is created now, not when the process started
        self._host_restarted = time.time()

    def _parent_ports(self, pid: int, depth: int | None = None, *, vllm: bool = False) -> list[int]:
        effective_depth = (4 if vllm else 2) if depth is None else depth
        d = self.inspector.get(pid)
        if vllm:
            d.parents = []
            candidates = []
            try:
                p = psutil.Process(pid)
                if not d.create_time or p.create_time() != d.create_time or d.vllm_ambiguous:
                    return []
                child_start = d.create_time
                seen = {pid}
                for _ in range(effective_depth):
                    p = p.parent()
                    if p is None:
                        break
                    parent = self.inspector.get(p.pid)
                    if p.pid in seen or not parent.create_time or parent.create_time != p.create_time():
                        return []
                    if parent.create_time > child_start:
                        return []
                    if not parent.vllm_role:
                        break
                    if parent.vllm_ambiguous:
                        return []
                    seen.add(p.pid)
                    d.parents.append(p.pid)
                    child_start = parent.create_time
                    ports = self.inspector.exclusive_ports(p.pid)
                    if parent.vllm_role == "worker":
                        if parent.ports and not _is_vllm_ephemeral_worker_ports(parent):
                            return []
                        continue
                    if parent.ports and set(parent.ports) != set(ports):
                        return []  # a shared listener cannot identify a server
                    if parent.vllm_role == "server":
                        candidates.extend(ports)
                return candidates if len(candidates) == 1 else []
            except (psutil.Error, OSError):
                return []
        if d.parents is None:
            d.parents = []
            try:
                p = psutil.Process(pid)
                for _ in range(effective_depth):
                    p = p.parent()
                    if p is None:
                        break
                    d.parents.append(p.pid)
            except (psutil.Error, OSError):
                pass
        return [port for ppid in d.parents for port in self.inspector.exclusive_ports(ppid)]

    @staticmethod
    def _loopback_port(url: str) -> int | None:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            return None
        if parsed.port is not None:
            return parsed.port
        return 443 if parsed.scheme.lower() == "https" else 80

    def _vllm_launch_hints(self, detail: ProcDetail) -> dict:
        ports: set[int] = set()
        models: list[str] = []
        launch_models: list[str] = []
        log_file: str | None = None
        log_owner_pid: int | None = None
        seen_pids: set[int] = set()
        chain_details: list[ProcDetail] = [detail]
        for ppid in detail.parents or []:
            if ppid not in seen_pids:
                seen_pids.add(ppid)
                chain_details.append(self.inspector.get(ppid))
        for item in chain_details:
            if not item.cmdline:
                continue
            info = vllm_launch_info(item.cmdline)
            if info["port"] is not None:
                ports.add(int(info["port"]))
            if info["model"]:
                models.append(str(info["model"]))
                launch_models.append(str(info["model"]))
            for alias in info["served_model_names"]:
                models.append(str(alias))
            if log_file is None and info["log_file"]:
                log_file = str(info["log_file"])
                log_owner_pid = item.pid
        return {
            "ports": ports,
            "models": models,
            "launch_models": launch_models,
            "log_file": log_file,
            "log_owner_pid": log_owner_pid,
        }

    def _match_vllm_endpoint(
        self,
        row: ProcRow,
        own_ports: list[int],
        parent_ports: list[int],
        auto_server: ServerInfo | None,
        hints: dict,
    ) -> tuple[VllmEndpointConfig | None, ServerInfo | None, list[str]]:
        """Select and probe a configured vLLM endpoint for this row.

        Order: gpu_index filter -> match_pid -> loopback port -> match_model -> single unconstrained fallback.
        Ambiguous multiple matches stay unmatched with an explanatory note.
        """
        cfg = self.config
        if cfg is None or not cfg.vllm.endpoints:
            return None, None, []

        notes: list[str] = []
        candidates = [ep for ep in cfg.vllm.endpoints if ep.gpu_index is None or ep.gpu_index == self.gpu_index]
        if not candidates:
            return None, None, []

        launch_model_ids = tuple(dict.fromkeys(hints.get("launch_models", ()))) or None
        selected: VllmEndpointConfig | None = None
        preprobed: ServerInfo | None = None

        # 1. Explicit match_pid
        pid_matches = [ep for ep in candidates if ep.match_pid == row.pid]
        if len(pid_matches) == 1:
            selected = pid_matches[0]
        elif len(pid_matches) > 1:
            notes.append(f"Multiple config.yaml vLLM endpoints match PID {row.pid}; endpoint association skipped.")
            return None, None, notes
        else:
            candidates = [ep for ep in candidates if ep.match_pid is None]

        # 2. Loopback port match against own/parent listening ports or launch --port
        if selected is None:
            known_ports = set(own_ports) | set(parent_ports) | set(hints["ports"])
            if known_ports:
                port_matches = [ep for ep in candidates if self._loopback_port(ep.url) in known_ports]
                if len(port_matches) == 1:
                    selected = port_matches[0]
                elif len(port_matches) > 1:
                    notes.append("Multiple config.yaml vLLM endpoints match this process's listening port; "
                                 "set match_pid or match_model to disambiguate.")
                    return None, None, notes

        # 3. match_model against known model hints or endpoint /v1/models
        if selected is None:
            model_candidates = [ep for ep in candidates if ep.match_model is not None]
            if model_candidates:
                known_models = [m["name"] for m in auto_server.models if m.get("name")] if (
                    auto_server and auto_server.kind == "vllm") else []
                known_models.extend(hints["models"])
                known_lower = {m.lower() for m in known_models}
                if known_lower:
                    model_matches = [ep for ep in model_candidates if ep.match_model.lower() in known_lower]
                    if len(model_matches) == 1:
                        selected = model_matches[0]
                    elif len(model_matches) > 1:
                        notes.append("Multiple config.yaml vLLM endpoints match this process's model name; "
                                     "set match_pid or gpu_index to disambiguate.")
                        return None, None, notes
                else:
                    probed_matches: list[tuple[VllmEndpointConfig, ServerInfo]] = []
                    for ep in model_candidates:
                        origin = ("config", ep.url, ep.name)
                        srv = self.prober.get_url(
                            ep.url,
                            key=ep.resolve_api_key(),
                            timeout=ep.timeout_s,
                            origin=origin,
                            extra_model_ids=launch_model_ids,
                        )
                        if srv and any(m.get("name", "").lower() == ep.match_model.lower() for m in srv.models):
                            probed_matches.append((ep, srv))
                    if len(probed_matches) == 1:
                        selected, preprobed = probed_matches[0]
                    elif len(probed_matches) > 1:
                        notes.append("Multiple config.yaml vLLM endpoints serve matching models; "
                                     "set match_pid or gpu_index to disambiguate.")
                        return None, None, notes

        # 4. Single unconstrained endpoint fallback (no match_pid and no match_model)
        if selected is None:
            unconstrained = [ep for ep in candidates if ep.match_pid is None and ep.match_model is None]
            if len(unconstrained) == 1:
                selected = unconstrained[0]
            elif len(unconstrained) > 1:
                notes.append("Multiple config.yaml vLLM endpoints match this GPU/process; "
                             "set match_pid, match_model, or gpu_index to disambiguate.")
                return None, None, notes

        if selected is None:
            return None, None, notes

        if preprobed is not None:
            return selected, preprobed, notes

        key = selected.resolve_api_key()
        origin = ("config", selected.url, selected.name)
        srv = self.prober.get_url(
            selected.url,
            key=key,
            timeout=selected.timeout_s,
            origin=origin,
            extra_model_ids=launch_model_ids,
        )
        if srv is None:
            if self.prober.locked_url(selected.url, key=key, origin=origin):
                notes.append(f"Configured vLLM API at {selected.url} rejected the request (401); "
                             "check api_key or api_key_env.")
            else:
                notes.append(f"Configured vLLM API at {selected.url} is unreachable; "
                             "falling back to init log / NVML data if available.")
        return selected, srv, notes

    @staticmethod
    def _vllm_fd_log_paths(row: ProcRow, hints: dict) -> list[str]:
        """O(1) Linux check of /proc/<pid>/fd/{1,2} for a regular stdout/stderr log file (never calls open_files)."""
        if not sys.platform.startswith("linux"):
            return []
        pids: list[int] = [row.pid]
        owner = hints.get("log_owner_pid")
        if isinstance(owner, int) and owner not in pids:
            pids.append(owner)
        for pid in (row.detail.parents or [])[:4]:
            if pid not in pids:
                pids.append(pid)
        seen: list[str] = []
        for pid in pids:
            if pid <= 0:
                continue
            for fd in ("1", "2"):
                try:
                    dest = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if (not dest.startswith("/")
                        or dest.startswith(("/dev/", "/proc/", "/sys/"))
                        or dest.endswith(" (deleted)")
                        or dest in seen):
                    continue
                if os.path.isfile(dest):
                    seen.append(dest)
        return seen

    def _resolve_vllm_init_profile(
        self,
        row: ProcRow,
        matched_ep: VllmEndpointConfig | None,
        hints: dict,
    ) -> VllmInitProfile | None:
        """Resolve vLLM startup log: endpoint.log_path -> config.vllm.log_paths -> --log-file -> /proc/<pid>/fd/{1,2}."""
        proc_start = row.detail.create_time
        if matched_ep is not None and matched_ep.log_path:
            profile = parse_vllm_init_log(matched_ep.log_path, process_start=proc_start)
            if profile is not None:
                return profile

        cfg = self.config
        if cfg is not None and cfg.vllm.log_paths:
            best_fresh: tuple[float, VllmInitProfile] | None = None
            best_stale: tuple[float, VllmInitProfile] | None = None
            for log_path in cfg.vllm.log_paths:
                profile = parse_vllm_init_log(log_path, process_start=proc_start)
                if profile is None:
                    continue
                rank_time = profile.timestamp_epoch or profile.file_mtime or 0.0
                if not profile.stale:
                    if best_fresh is None or rank_time > best_fresh[0]:
                        best_fresh = (rank_time, profile)
                else:
                    if best_stale is None or rank_time > best_stale[0]:
                        best_stale = (rank_time, profile)
            if best_fresh is not None:
                return best_fresh[1]
            if best_stale is not None:
                return best_stale[1]

        if hints.get("log_file"):
            resolved = target_path(hints.get("log_owner_pid") or row.pid, hints["log_file"])
            if resolved:
                profile = parse_vllm_init_log(resolved, process_start=proc_start)
                if profile is not None:
                    return profile

        vllm_auto_detect = True if cfg is None else cfg.vllm.auto_detect
        if vllm_auto_detect:
            for fd_log in self._vllm_fd_log_paths(row, hints):
                profile = parse_vllm_init_log(fd_log, process_start=proc_start)
                if profile is not None:
                    return profile
        return None

    def sample(self) -> Snapshot:
        per_luid, adapters = self.counters.sample()
        luid = self.adapter.luid if self.adapter else (next(iter(per_luid), 0))
        procs = per_luid.get(luid, {})
        a = adapters.get(luid)
        integrated_gpu = BACKEND == "wddm" and self.adapter is not None and self.adapter.integrated_gpu
        integrated_local_pids = {
            pid for pid, gpu in procs.items()
            if integrated_gpu and "dedicated" in gpu.available and gpu.dedicated >= MiB
        }
        if integrated_gpu:
            for gpu in procs.values():
                if "dedicated" in gpu.available and gpu.dedicated > 0:
                    gpu.shared += gpu.dedicated
                    gpu.dedicated = 0
        if BACKEND == "metal":
            nv = self.counters.device  # IOKit's device stats in the shape the header renders (no NVML on macOS)
        elif self.drm_card is not None:
            from .drm import read_device

            nv = read_device(self.drm_card, self.gpu_index)
        elif self.ascend_chip is not None:
            nv = self.ascend.read_device(self.ascend_chip, self.gpu_index)
        elif self.adapter and self.adapter.vendor_id != VENDOR_NVIDIA:
            # NVML can't describe this GPU, and a sweep of every NVIDIA device costs ~60 ms per device.
            if self.adapter.vendor_id == VENDOR_AMD:
                nv = self.adl.sample() or self.amdsmi.sample()
            elif (self.adapter.vendor_id in (VENDOR_INTEL, VENDOR_QUALCOMM) and BACKEND == "wddm"
                  and a and a.util_gpu is not None):
                nv = DeviceTelemetry(index=self.gpu_index, source="Windows WDDM", util_gpu=round(a.util_gpu))
            else:
                nv = None
        else:
            nvs = self.nvml.sample() if self.nvml.ok else []
            nv = select_nv_device(self.adapter, nvs, self.gpu_index, self.adapters)
        total = (nv.mem_total if nv else 0) or (self.adapter.dedicated_vram if self.adapter else 0)

        used = int(a.dedicated) if a and a.dedicated is not None else (
            int(nv.mem_used) if BACKEND != "wddm" and nv and nv.mem_used is not None else None)
        snap = Snapshot(
            taken=time.time(), adapter=self.adapter, adapters=self.adapters, nv=nv,
            driver=(self._mac_driver
                    or (self.amdsmi.driver if nv and nv.source == "AMD SMI" else "")
                    or (self.nvml.driver if nv and nv.source == "NVML" else "")
                    or (f"kernel {os.uname().release}" if nv and nv.source == "AMD sysfs" else "")
                    or getattr(self, "_windows_driver", "")),
            cuda=self.nvml.cuda_version if nv and nv.source == "NVML" else "", vram_total=total,
            vram_used=used,
            shared_used=int(a.shared) if a and a.shared is not None else None, backend=self.backend,
            shared_is_spill=not (integrated_gpu or (self.drm_card is not None and self.drm_card.integrated)),
        )

        fallbacks = getattr(self.counters, "fallback_counters", None)
        missing_dedicated = getattr(self.counters, "dedicated_counter_missing", False) is True
        if BACKEND == "wddm" and (missing_dedicated or
                                  (isinstance(fallbacks, set) and "dedicated" in fallbacks)):
            snap.process_memory_available = False
        if BACKEND == "metal" and self.counters.hidden_unmeasured:
            snap.process_memory_available = False
            snap.counter_status = "Attributed GPU memory is unknown: some idle Metal clients have unmeasured memory."

        snap.host_status = self.host_status
        if BACKEND == "metal" and not self.counters.process_available:
            snap.process_memory_available = False
            snap.counter_status = "Metal client query failed; per-process GPU readings are unavailable."
        if BACKEND == "wsl" and (not self.adapter or self.adapter.vendor_id == VENDOR_AMD):
            snap.counter_status = " ".join(dict.fromkeys(filter(None, (amdsmi_status(), self.amdsmi.status))))
            if self.adapter is None:
                snap.process_memory_available = False
        if BACKEND == "wddm" and self.adapter:
            snap.counter_adapter_matched = luid in per_luid or luid in adapters
            if isinstance(self.adapter, TccAdapter):
                snap.counter_status = (
                    f"{self.adapter.driver_model} compute-only GPU: VRAM and processes come "
                    "from NVML. Per-process engine load, shared memory and commit are n/a, and per-process VRAM "
                    "is n/a when the driver does not report it." if snap.counter_adapter_matched else
                    f"NVML could not read this {self.adapter.driver_model} GPU; VRAM and processes are unavailable.")
            elif not snap.counter_adapter_matched:
                snap.counter_status = ("Windows GPU counters could not be matched to the selected adapter; "
                                       "VRAM and processes are unavailable. The adapter may be temporarily "
                                       "unavailable, or GPU-PV VMs may not expose these counters.")
            elif a is None or a.dedicated is None or a.shared is None:
                snap.counter_status = ("Some Windows adapter memory counters are unavailable; "
                                       "GPU-PV VMs may not expose them.")
            if (nv is None and self.adapter.vendor_id == VENDOR_NVIDIA and self.adapter.pci_address
                    and any(_wddm_bdf(b) == self.adapter.pci_address for b in nvml_mod.LOST)):
                snap.nv_lost = True
                snap.counter_status = " ".join(filter(None, (
                    snap.counter_status, "NVIDIA telemetry stopped after this GPU was re-enabled; "
                                         "restart gputaskman to restore it.")))
            snap.counter_status = " ".join(filter(None, (snap.counter_status, self.adapter.identity_status)))
            if isinstance(fallbacks, set) and fallbacks:
                snap.counter_status = " ".join(filter(None, (
                    snap.counter_status, "Windows fallback process memory counters may overlap across processes; "
                    "charged memory and Paged are unavailable for affected counters.")))
            if missing_dedicated:
                snap.counter_status = " ".join(filter(None, (
                    snap.counter_status, "Windows per-process dedicated memory counters are unavailable; "
                    "attributed VRAM is unknown.")))
            if getattr(self.counters, "shared_counter_missing", False) is True:
                snap.counter_status = " ".join(filter(None, (
                    snap.counter_status, "Windows per-process shared memory counters are unavailable; "
                    "shared memory and Paged are unknown.")))
        if self.drm_card is not None:
            snap.process_status = self.counters.drm.status
            if self.drm_card.intel and nv is not None:
                from .drm import intel_counter_status

                snap.counter_status = intel_counter_status(self.drm_card, nv)
        if self.ascend is not None:
            snap.process_memory_available = self.ascend.process_available
            snap.counter_status = self.ascend.status
        elif self.backend == "ascend":
            snap.process_memory_available = False
            snap.counter_status = "Ascend chip unavailable: npu-smi could not match the selected adapter"
        if self.backend == "nvml":
            process_status = getattr(self.counters, "process_status", {})
            process_status = process_status.get(luid, "") if isinstance(process_status, dict) else ""
            if process_status:
                snap.process_memory_available = False
                snap.counter_status = process_status
        live = [(pid, g) for pid, g in procs.items() if _is_live(g) or pid in integrated_local_pids]
        live.sort(key=lambda x: -(x[1].shown_vram or x[1].wddm_dedicated_usage or 0))
        for pid, g in live:
            d = self.inspector.get(pid)
            row = ProcRow(pid=pid, name=d.name, kind=classify(d, g.engines), gpu=g, detail=d)
            snap.procs.append(row)

        for pid, g in getattr(self.counters, "host_procs", {}).get(luid, {}).items():
            snap.procs.append(ProcRow(pid, "(unresolved host PID)", "external", g,
                                      ProcDetail(pid, "(unresolved host PID)"), host=True, origin="nvml"))
        if self.host is not None:  # current inferred sizes drive both scan priority and the LLM breakdown
            self._add_host_rows(snap)
        if any(row.gpu.engine_reason == ENGINE_UNAVAILABLE_REASON for row in snap.procs):
            snap.counter_status = " ".join(filter(None, (
                snap.counter_status,
                "Windows GPU engine utilization counters outside 0-1000% are unavailable for affected processes "
                "and device load derived from those counters.",
            )))
        if (snap.backend == "wsl" and snap.host_data and snap.vram_total
                and sum(r.gpu.dedicated for r in snap.procs) > snap.vram_total):
            snap.counter_status = " ".join(filter(None, (
                snap.counter_status, "Windows per-process counters exceed device capacity and may overlap; "
                                     "attributed total is unknown. Use the adapter reading as the device total.")))
        local_rows = [r for r in snap.procs if not r.host]
        if self.scan_budget is None:
            deep_pids = {r.pid for r in local_rows}
        else:
            deep_pids = {r.pid for r in local_rows[: self.DEEP_TOP_N]} | {
                r.pid for r in local_rows if r.gpu.gpu_pct >= 2}
        deadline = None if self.scan_budget is None else time.perf_counter() + self.scan_budget
        ollama_runners = []
        for row in local_rows:  # largest VRAM first, so the budget goes to what matters
            d, g = row.detail, row.gpu
            mem = (g.shown_vram or 0) + g.shared
            if (row.pid in deep_pids and (deadline is None or time.perf_counter() < deadline)
                    and self.inspector.needs_scan(d, mem)):
                self.inspector.scan(d, mem)
            row.kind = classify(d, g.engines)
            # Old Ollama: "ollama.exe runner ..."; newer Ollama: its own lib\ollama\llama-server.exe
            exe = d.exe.lower()
            if row.kind in AI_KINDS and ("\\ollama\\" in exe or "/ollama/" in exe or
                                         (d.name.lower().startswith("ollama") and "runner" in d.cmdline)):
                ollama_runners.append(row)
        matched_config_ids: set[int] = set()
        saw_local_vllm = False
        for row in snap.procs:
            if row.host:
                continue
            if os.path.basename(row.detail.exe).lower() in ("handy.exe", "handy"):
                row.handy = inspect_handy(row.detail.create_time)
            if not _llm_wanted(row, snap.shared_is_spill, self.config):
                continue
            ports = self.inspector.exclusive_ports(row.pid)
            foundry = row.detail.foundry_local
            key = None if foundry else api_key(row.detail.cmdline)
            cfg_vllm_match = bool(
                self.config is not None
                and self.config.vllm.endpoints
                and (row.detail.name.startswith("VLLM::")
                     or any(ep.match_pid == row.pid for ep in self.config.vllm.endpoints))
            )
            vllm = bool(row.detail.vllm_role) or cfg_vllm_match
            if vllm:
                saw_local_vllm = True
            kind = "foundry-local" if foundry else "vllm" if vllm else None
            own_origin = ("process", row.pid, row.detail.create_time)
            if foundry:
                parent_ports = []
                if (not row.detail.create_time or len(ports) != 1
                        or set(ports) != set(row.detail.ports)):
                    ports = []
            elif vllm and _is_vllm_ephemeral_worker_ports(row.detail):
                ports = []
                parent_ports = self._parent_ports(row.pid, depth=4, vllm=True)
            elif vllm and (row.detail.vllm_ambiguous or len(ports) > 1 or
                         set(ports) != set(row.detail.ports)):
                ports = []
                parent_ports = []
            else:
                parent_ports = self._parent_ports(row.pid, depth=4 if vllm else 2, vllm=vllm)
            if vllm and ports and parent_ports:
                ports, parent_ports = [], []  # two possible server processes: do not pick one
            parent_origin = ("parent", tuple((pid, self.inspector.get(pid).create_time)
                                            for pid in row.detail.parents or []))
            hints = self._vllm_launch_hints(row.detail) if vllm else None
            launch_model_ids = tuple(dict.fromkeys(hints["launch_models"])) if hints else ()
            vllm_auto_detect = True if self.config is None else self.config.vllm.auto_detect
            if vllm and not vllm_auto_detect:
                own = None
                host = None
            else:
                own = self.prober.get(
                    ports, key, kind=kind, origin=own_origin,
                    extra_model_ids=launch_model_ids or None,
                )
                host = self.prober.get(
                    parent_ports, kind=kind, origin=parent_origin,
                    extra_model_ids=launch_model_ids or None,
                )
            log = ollama_log_path() if row in ollama_runners and len(ollama_runners) == 1 else None
            # On an APU, GTT is the GPU's ordinary memory (often most of it), not spill: count it as held memory.
            unproven_owner = snap.backend == "nvml" and bool(row.gpu.inferred_by)
            held = 0 if unproven_owner else (
                (row.gpu.shown_vram or 0) + (0 if snap.shared_is_spill else (row.gpu.shared or 0)))
            attributed = row.gpu.shown_vram is not None and not unproven_owner

            if vllm:
                assert hints is not None
                matched_ep, cfg_server, vllm_notes = self._match_vllm_endpoint(
                    row, ports, parent_ports, own or host, hints
                )
                if matched_ep is not None:
                    matched_config_ids.add(id(matched_ep))
                init_profile = self._resolve_vllm_init_profile(row, matched_ep, hints)
                if cfg_server is not None:
                    selected_server = cfg_server
                    selected_host = None
                    selected_source = "config"
                else:
                    selected_server = own or host
                    selected_host = host if own else None
                    selected_source = "process" if own else "parent" if host else None
                vram_measured = row.gpu.attributed and "dedicated" in row.gpu.available
                row.llm = build_view(
                    row.detail,
                    held,
                    row.gpu.shared if snap.shared_is_spill else 0,
                    selected_server,
                    log,
                    host=selected_host,
                    attributed=attributed,
                    server_source=selected_source,
                    vllm_init_profile=init_profile,
                    vllm_endpoint_url=matched_ep.url if matched_ep else None,
                    vllm_endpoint_name=matched_ep.name if matched_ep else None,
                    gpu_id=f"gpu:{self.gpu_index}",
                    device_vram_used=snap.vram_used,
                    device_vram_total=snap.vram_total,
                    vram_measured=vram_measured,
                    sample_time=snap.taken,
                    vllm_notes=vllm_notes,
                )
                for note in vllm_notes:
                    if note not in row.llm.notes:
                        row.llm.notes.append(note)
            else:
                row.llm = build_view(row.detail, held, row.gpu.shared if snap.shared_is_spill else 0,
                                     own or host, log, host=host if own else None,
                                     attributed=attributed,
                                     server_source="process" if own else "parent" if host else None)

            if not (vllm and not vllm_auto_detect):
                locked = None if own else self.prober.locked_port(ports, key, kind=kind, origin=own_origin)
                if locked:
                    why = ("rejected the key from the launch arguments" if key else
                           "wants an API key, and none was found in the launch arguments (keys from LLAMA_API_KEY "
                           "or --api-key-file are not read)")
                    row.llm.notes.append(f"Local API on port {locked} {why}. Server details are skipped.")
                if not host and self.prober.locked_port(parent_ports, kind=kind, origin=parent_origin):
                    row.llm.notes.append("Parent-process API locked (401). Parent probes never send the worker's "
                                         "API key; parent server details are skipped.")

        if self.config is not None and self.config.vllm.endpoints:
            self._append_unmatched_config_vllm_rows(snap, matched_config_ids, saw_local_vllm)

        self.inspector.prune(set(procs))
        return snap

    def _append_unmatched_config_vllm_rows(
        self,
        snap: Snapshot,
        matched_config_ids: set[int],
        saw_local_vllm: bool,
    ) -> None:
        """Synthesize read-only config endpoint rows (Docker/Sidecar/Remote) when no local PID matched."""
        from urllib.parse import urlsplit

        cfg = self.config
        if cfg is None or not cfg.vllm.endpoints:
            return
        for idx, ep in enumerate(cfg.vllm.endpoints):
            if id(ep) in matched_config_ids:
                continue
            if ep.gpu_index is not None and ep.gpu_index != self.gpu_index:
                continue
            if ep.match_pid is not None:
                continue
            key = ep.resolve_api_key()
            origin = ("config", ep.url, ep.name)
            extra_ids = (ep.match_model,) if ep.match_model else None
            srv = self.prober.get_url(
                ep.url,
                key=key,
                timeout=ep.timeout_s,
                origin=origin,
                extra_model_ids=extra_ids,
            )
            if saw_local_vllm:
                if srv is None:
                    continue
                if ep.match_model and not any(m.get("name") == ep.match_model for m in srv.models):
                    continue
            init_profile = parse_vllm_init_log(ep.log_path, process_start=0.0) if ep.log_path else None
            if init_profile is None and cfg.vllm.log_paths:
                for log_path in cfg.vllm.log_paths:
                    init_profile = parse_vllm_init_log(log_path, process_start=0.0)
                    if init_profile is not None:
                        break
            notes = [
                "Configured vLLM endpoint (no local PID matched; e.g. Docker container, sidecar, or remote server)."
            ]
            if srv is None:
                if self.prober.locked_url(ep.url, key=key, origin=origin):
                    notes.append(f"Configured vLLM API at {ep.url} rejected the request (401); "
                                 "check api_key or api_key_env.")
                else:
                    notes.append(f"Configured vLLM API at {ep.url} is unreachable; "
                                 "falling back to init log / NVML data if available.")
            ep_label = ep.name or urlsplit(ep.url).netloc or f"vllm-{idx + 1}"
            detail = ProcDetail(pid=idx + 1, name=ep_label, runtimes=["vLLM"])
            view = build_view(
                detail,
                0,
                0,
                srv,
                None,
                host=None,
                attributed=False,
                server_source="config",
                vllm_init_profile=init_profile,
                vllm_endpoint_url=ep.url,
                vllm_endpoint_name=ep.name,
                gpu_id=f"gpu:{self.gpu_index}",
                device_vram_used=snap.vram_used,
                device_vram_total=snap.vram_total,
                vram_measured=False,
                sample_time=snap.taken,
                vllm_notes=notes,
            )
            for note in notes:
                if note not in view.notes:
                    view.notes.append(note)
            snap.procs.append(
                ProcRow(
                    pid=idx + 1,
                    name=ep_label,
                    kind="LLM",
                    gpu=ProcGpu(dedicated=0, shared=0, attributed=False, available=set()),
                    detail=detail,
                    llm=view,
                    host=True,
                    origin="config",
                )
            )

    def _add_host_rows(self, snap: Snapshot) -> None:
        """Windows processes on this GPU, plus the WSL VM itself (vmwp.exe: the exact total of Linux VRAM)."""
        if self.host.ended and time.time() - self._host_restarted > 30:
            self.host.close()  # PowerShell exited (e.g. killed, or the host was busy): try again
            self.host = HostStream()
            self._host_restarted = time.time()
        hs = self.host.latest()
        if hs is None:
            snap.host_status = "starting" if self.host.status in ("starting", "ok") else self.host.status
            return
        # Match the host adapter once: NVML is read now while the host sample can be a second old, so
        # re-matching every refresh fails whenever memory moves fast (a model loading) and the rows flicker.
        luid = self._host_luid if self._host_luid in hs.adapters else match_adapter(hs, snap.vram_used)
        if luid is None:
            # Memory can swing by GBs between the two readings while a workload loads: retry for a while.
            early = time.time() - self._host_restarted < 20
            snap.host_status = "starting" if early else "unmatched (no Windows adapter matches this GPU's usage)"
            return
        self._host_luid = luid
        age = time.time() - hs.taken
        snap.host_status = "ok" if age < 3 else f"ok (last Windows sample {age:.0f}s old)"
        a = hs.adapters.get(luid)
        if a is not None:
            snap.shared_used = int(a.shared) if a.shared is not None else None
        procs = hs.procs.get(luid, {})
        vm_workers = [pid for pid in procs if hs.names.get(pid, "").lower() == "vmwp"]
        host_rows: list[ProcRow] = []
        for pid, g in procs.items():
            # Keep large commitments visible even when the residency inputs needed for Paged are missing.
            committed_only_activity = "committed" in g.available and g.committed >= 16 * MiB and not g.paged_available
            if not _is_live(g) and not committed_only_activity:
                continue
            raw = hs.names.get(pid, "?")
            if raw.lower() == "vmwp":
                # One VM worker is WSL's; several could include Hyper-V VMs or Windows Sandbox.
                name = "WSL VM (vmwp.exe)" if len(vm_workers) == 1 else "VM worker (vmwp.exe)"
                kind = "wsl-vm"
            else:
                name = raw if raw in ("System", "Idle", "?") or raw.lower().endswith(".exe") else f"{raw}.exe"
                kind = "kernel" if pid == 4 else classify(ProcDetail(pid, name), g.engines)
            host_rows.append(ProcRow(pid, name, kind, g, ProcDetail(pid, name), host=True, origin="windows"))
        # Largest first; Linux rows sit right under the WSL VM row, whose total they share.
        host_rows.sort(key=lambda r: -r.gpu.dedicated)
        at = next((i + 1 for i, r in enumerate(host_rows) if r.kind == "wsl-vm"), 0)
        if len(vm_workers) == 1 and self.tracker is not None:
            self._attribute(snap, luid, hs.contexts.get(luid, {}).get(vm_workers[0], []))
        snap.procs.sort(key=lambda r: -(r.gpu.shown_vram or 0))
        snap.procs = host_rows[:at] + snap.procs + host_rows[at:]

    def _attribute(self, snap: Snapshot, luid: int, contexts: list[float]) -> None:
        linux = [r for r in snap.procs if not r.host and not r.gpu.attributed]
        # This monitor holds a (0-byte) context too through NVML; AMD SMI only holds one if /dev/dxg stays open.
        holders = [(r.pid, r.detail.create_time) for r in linux]
        if not (snap.adapter and snap.adapter.vendor_id == VENDOR_AMD) or os.getpid() in self.counters.dxg_holders:
            holders.append((os.getpid(), self._own_start))
        guesses, reason = self.tracker.update(luid, contexts, holders)
        snap.wsl_split = reason
        for r in linux:
            g = guesses.get(r.pid)
            if g is not None:
                r.gpu.inferred, r.gpu.inferred_by = g.bytes, g.confidence

    def close(self) -> None:
        with ExitStack() as resources:
            if self.host is not None:
                resources.callback(self.host.close)
            resources.callback(self.adl.close)
            resources.callback(self.amdsmi.close)
            resources.callback(self.nvml.close)
            resources.callback(self.counters.close)
