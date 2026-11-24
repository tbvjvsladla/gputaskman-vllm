from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gputaskman import linux
from gputaskman.collector import Collector, ProcRow, Snapshot, select_nv_device
from gputaskman.dxgi import Adapter
from gputaskman.nvml import NvDevice
from gputaskman.gguf import GgufInfo
from gputaskman.llm import LlmView, ServerProber, build_view
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import MappedModel, ProcDetail, _lib_name, _match_runtimes, classify
from gputaskman.report import to_dict, to_text

GiB = 1 << 30
_no_drm = patch("gputaskman.drm.cards", return_value=[])  # keep the test machine's real AMD GPUs out of NVML tests
_no_ascend = patch("gputaskman.ascend.probe", return_value=[])  # and its real Ascend NPUs out too
_no_amdsmi = patch("gputaskman.linux._amdsmi_adapters", return_value=[])  # and its WSL AMD GPUs out too


def setUpModule() -> None:
    _no_drm.start()
    _no_ascend.start()
    _no_amdsmi.start()


def tearDownModule() -> None:
    _no_drm.stop()
    _no_ascend.stop()
    _no_amdsmi.stop()

NOT_AVAILABLE = (1 << 64) - 1  # NVML_VALUE_NOT_AVAILABLE as an unsigned long long


class FakeNvml:
    """Just enough of pynvml for LinuxCounters: one GPU, scripted process lists and utilization samples."""

    def __init__(self, compute=(), graphics=(), samples=(), used=4 * GiB):
        self.compute, self.graphics, self.samples, self.used = list(compute), list(graphics), list(samples), used
        self.util_calls = []

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return "h0"

    def nvmlDeviceGetPciInfo(self, h):
        return SimpleNamespace(busId=b"00000000:01:00.0")

    def nvmlDeviceGetName(self, h):
        return "NVIDIA GeForce RTX 3090 Ti"

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(total=24 * GiB, used=self.used)

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return self.compute

    def nvmlDeviceGetGraphicsRunningProcesses(self, h):
        return self.graphics

    def nvmlDeviceGetProcessUtilization(self, h, since):
        self.util_calls.append(since)
        return self.samples


def proc(pid, used):
    return SimpleNamespace(pid=pid, usedGpuMemory=used)


def sample(pid, sm, enc=0, dec=0, ts=1):
    return SimpleNamespace(pid=pid, smUtil=sm, encUtil=enc, decUtil=dec, timeStamp=ts)


KEY = linux.pci_key("00000000:01:00.0")


class LinuxLibraryTests(unittest.TestCase):
    @patch("gputaskman.collector.sys.platform", "linux")
    def test_nvml_device_matches_linux_pci_key(self):
        gpu = Adapter(KEY, "NVIDIA", 0x10DE, 24 * GiB, 0, False)
        device = NvDevice(index=1, pci_bus="00000000:01:00.0")
        other = NvDevice(index=0, pci_bus="00000000:02:00.0")
        self.assertIs(select_nv_device(gpu, [other, device], 0), device)
        self.assertIsNone(select_nv_device(gpu, [other], 0))
        self.assertIsNone(select_nv_device(gpu, [device, device], 0))

    def test_versioned_shared_objects_are_libraries(self):
        self.assertEqual(_lib_name("libcudart.so.12"), "libcudart.so")
        self.assertEqual(_lib_name("libcuda.so.1.1"), "libcuda.so")
        self.assertEqual(_lib_name("libtorch_cuda.so"), "libtorch_cuda.so")
        self.assertEqual(_lib_name("nvcuda.dll"), "nvcuda.dll")
        self.assertIsNone(_lib_name("model.gguf"))
        self.assertIsNone(_lib_name("weights.safetensors"))
        self.assertIsNone(_lib_name("model.so-weights.bin"))

    def test_linux_runtimes_are_detected(self):
        libs = {_lib_name(n) for n in ("libggml-cuda.so", "libllama.so", "libcudart.so.12", "libcuda.so.1",
                                       "libvulkan.so.1", "libtorch_cuda.so", "libnvidia-encode.so.1")}
        self.assertEqual(_match_runtimes(libs), ["llama.cpp", "ggml", "PyTorch", "CUDA", "Vulkan", "NVENC"])

    def test_linux_compositors_and_browsers(self):
        self.assertEqual(classify(ProcDetail(900, "Xorg"), {}), "compositor")
        self.assertEqual(classify(ProcDetail(901, "gnome-shell"), {}), "compositor")
        self.assertEqual(classify(ProcDetail(902, "firefox"), {}), "browser")

    def test_pid_4_is_only_special_on_windows(self):
        with patch("gputaskman.procinfo.sys.platform", "linux"):
            self.assertEqual(classify(ProcDetail(4, "kworker/0:0"), {}), "app")
        with patch("gputaskman.procinfo.sys.platform", "win32"):
            self.assertEqual(classify(ProcDetail(4, "System"), {}), "kernel")


class PdhImportTests(unittest.TestCase):
    def test_pdh_module_is_complete_without_loading_the_dll(self):
        from gputaskman import pdh
        for name in ("PDH_FMT_DOUBLE", "PDH_FMT_NOCAP100", "PDH_MORE_DATA", "_Item", "_load_pdh"):
            self.assertTrue(hasattr(pdh, name), name)


class LinuxCountersTests(unittest.TestCase):
    def setUp(self):
        # Same PID namespace (bare metal): NVML's PIDs are local processes holding an NVIDIA device.
        for p in (patch.object(linux, "nvidia_holders", return_value={100, 101, 200, 300, 555}),
                  patch.object(linux, "in_child_pid_namespace", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def counters(self, fake, wsl):
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=wsl):
            return linux.LinuxCounters()

    def collect(self, fake):
        collector = Collector.__new__(Collector)
        collector.adapter = Adapter(KEY, "NVIDIA GeForce RTX 3090 Ti", 0x10DE, 24 * GiB, 0, False)
        collector.adapters = [collector.adapter]
        collector.gpu_index, collector.scan_budget, collector.host, collector.host_status = 0, None, None, ""
        collector.counters = self.counters(fake, wsl=False)
        collector.nvml = Mock(ok=True, driver="", cuda_version="")
        collector.nvml.sample.return_value = [
            NvDevice(index=0, pci_bus="00000000:01:00.0", mem_total=24 * GiB, mem_used=fake.used)
        ]
        collector.drm_card = collector.ascend_chip = collector.ascend = None
        collector.inspector = Mock()
        collector.inspector.get.side_effect = lambda pid: ProcDetail(pid, "gpu-client")
        collector.inspector.needs_scan.return_value = False
        collector.prober = Mock()
        with patch.object(linux, "N", fake), \
                patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.sys.platform", "linux"):
            return collector.sample()

    def test_native_linux_attributes_vram_and_engines_per_pid(self):
        fake = FakeNvml(compute=[proc(100, 3 * GiB), proc(200, 1 * GiB)],
                        graphics=[proc(200, 1 * GiB), proc(300, 256 << 20)],
                        samples=[sample(100, 80, ts=5), sample(100, 90, ts=9), sample(300, 30, dec=20, ts=7)])
        c = self.counters(fake, wsl=False)
        with patch.object(linux, "N", fake):
            procs, adapters = c.sample()
            c.sample()
        per = procs[KEY]
        self.assertEqual(adapters[KEY].dedicated, 4 * GiB)
        self.assertEqual(per[100].dedicated, 3 * GiB)
        self.assertEqual(per[200].dedicated, 1 * GiB)  # in both lists: counted once
        self.assertEqual(per[100].engines["compute"], 90)  # latest sample wins
        self.assertEqual(per[300].engines["3d"], 30)  # graphics-only process: SM load is 3D work
        self.assertEqual(per[300].engines["decode"], 20)
        self.assertTrue(all(g.attributed for g in per.values()))
        self.assertEqual(fake.util_calls, [0, 9])  # next poll asks only for newer samples

    def test_collector_keeps_hidden_vram_unknown_and_engine_load_measured(self):
        fake = FakeNvml(compute=[proc(100, NOT_AVAILABLE), proc(101, 2 * GiB)],
                        samples=[sample(100, 47, enc=3, dec=5)])
        snap = self.collect(fake)
        data = to_dict(snap)
        rows = {row["pid"]: row for row in data["processes"]}

        self.assertIsNone(snap.attributed)
        self.assertEqual(snap.vram_used, 4 * GiB)
        self.assertIsNone(data["attributed"])
        self.assertEqual(rows[101]["vram_bytes"], 2 * GiB)
        self.assertIsNone(rows[100]["vram_bytes"])
        self.assertFalse(rows[100]["vram_attributed"])
        self.assertEqual(rows[100]["engines"], {"compute": 47, "encode": 3, "decode": 5})

    def test_collector_preserves_measured_zero_process_memory(self):
        fake = FakeNvml(compute=[proc(100, 0)], samples=[sample(100, 3)])
        snap = self.collect(fake)
        data = to_dict(snap)

        self.assertEqual(snap.attributed, 0)
        self.assertEqual(data["attributed"], 0)
        self.assertEqual(data["processes"][0]["vram_bytes"], 0)
        self.assertEqual(data["processes"][0]["gpu_pct"], 3)

    def test_hidden_per_process_memory_is_unknown_not_zero(self):
        fake = FakeNvml(compute=[proc(100, NOT_AVAILABLE), proc(101, None)])
        data = self.report(fake)
        self.assertIsNone(data["attributed"])
        self.assertEqual(data["vram_used"], 4 * GiB)
        for row in data["processes"]:
            self.assertFalse(row["vram_attributed"])
            self.assertIsNone(row["vram_bytes"])

    def test_conflicting_nvml_process_lists_do_not_supply_measured_vram(self):
        for compute_memory, graphics_memory in ((2 * GiB, NOT_AVAILABLE), (NOT_AVAILABLE, 2 * GiB)):
            with self.subTest(compute_memory=compute_memory, graphics_memory=graphics_memory):
                fake = FakeNvml(compute=[proc(100, compute_memory)],
                                graphics=[proc(100, graphics_memory), proc(300, GiB)],
                                samples=[sample(100, 47, enc=3, dec=5)])
                data = self.report(fake)
                self.assertIsNone(data["attributed"])
                hidden, measured = data["processes"]
                self.assertIsNone(hidden["vram_bytes"])
                self.assertEqual(hidden["engines"], {"compute": 47, "encode": 3, "decode": 5})
                self.assertEqual(measured["vram_bytes"], GiB)

    def test_measured_nvml_zero_counts_as_zero(self):
        data = self.report(FakeNvml(compute=[proc(100, 0)], graphics=[proc(100, 0)]))
        self.assertEqual(data["attributed"], 0)
        self.assertEqual(data["processes"][0]["vram_bytes"], 0)

    def report(self, fake):
        c = self.counters(fake, wsl=False)
        with patch.object(linux, "N", fake):
            procs, adapters = c.sample()
        rows = [ProcRow(pid, "gpu-client", "app", g, ProcDetail(pid, "gpu-client")) for pid, g in procs[KEY].items()]
        return to_dict(Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="",
                                vram_total=24 * GiB, vram_used=int(adapters[KEY].dedicated), shared_used=None,
                                backend="nvml", procs=rows))

    def test_nvml_process_list_failure_makes_total_unknown_and_recovers(self):
        fake = FakeNvml(graphics=[proc(300, GiB)])
        fake.nvmlDeviceGetComputeRunningProcesses = Mock(side_effect=RuntimeError("permission denied"))
        counters = self.counters(fake, wsl=False)
        collector = Collector.__new__(Collector)
        collector.adapter, collector.adapters = None, []
        collector.counters = counters
        collector.nvml = SimpleNamespace(ok=False)
        collector.drm_card = collector.ascend_chip = collector.ascend = None
        collector.gpu_index, collector.scan_budget = 0, None
        collector.host = None
        collector.host_status = ""
        collector._mac_driver = ""
        collector.inspector = Mock()
        collector.inspector.get.return_value = ProcDetail(300, "python3")
        collector.prober = Mock()

        with patch.object(linux, "N", fake), patch("gputaskman.collector.BACKEND", "nvml"):
            failed = collector.sample()
            fake.nvmlDeviceGetComputeRunningProcesses = lambda _handle: []
            fake.graphics = []
            fake.nvmlDeviceGetProcessUtilization = Mock(side_effect=RuntimeError("NVML_ERROR_NOT_FOUND"))
            recovered = collector.sample()

        self.assertIsNone(failed.attributed)
        self.assertIn("compute process list", failed.counter_status)
        self.assertIn("permission denied", failed.counter_status)
        self.assertEqual(failed.procs[0].gpu.dedicated, GiB)
        self.assertEqual(recovered.attributed, 0)
        self.assertEqual(recovered.counter_status, "")

        fake.nvmlDeviceGetComputeRunningProcesses = Mock(side_effect=RuntimeError("compute unavailable"))
        fake.nvmlDeviceGetGraphicsRunningProcesses = Mock(side_effect=RuntimeError("graphics unavailable"))
        with patch.object(linux, "N", fake), patch("gputaskman.collector.BACKEND", "nvml"):
            both_failed = collector.sample()
        self.assertIsNone(both_failed.attributed)
        self.assertIn("compute process list", both_failed.counter_status)
        self.assertIn("graphics process list", both_failed.counter_status)

    def test_utilization_for_unlisted_pids_is_ignored(self):
        fake = FakeNvml(compute=[proc(100, GiB)], samples=[sample(555, 50)])
        c = self.counters(fake, wsl=False)
        with patch.object(linux, "N", fake):
            procs, _ = c.sample()
        self.assertNotIn(555, procs[KEY])

    def test_wsl_never_uses_windows_host_pids(self):
        # Under WSL NVML reports Windows PIDs (dwm 2744, System 4); they must not become Linux rows.
        fake = FakeNvml(samples=[sample(2744, 12), sample(4, 88)], used=7 * GiB)
        c = self.counters(fake, wsl=True)
        with patch.object(linux, "N", fake), patch.object(linux, "dxg_pids", return_value={4242}):
            procs, adapters = c.sample()
        self.assertEqual(adapters[KEY].dedicated, 7 * GiB)
        self.assertEqual(set(procs[KEY]), {4242})
        self.assertFalse(procs[KEY][4242].attributed)
        self.assertEqual(fake.util_calls, [])

    def test_wsl_amd_collects_dxg_pids_without_nvml(self):
        amd = Adapter(0x3000, "AMD Radeon RX 7900 XTX", 0x1002, 24 * GiB, 0, False, (3, 0, 0))
        with patch.object(linux, "N", None), patch.object(linux, "is_wsl", return_value=True), \
                patch.object(linux, "_amdsmi_adapters", return_value=[amd]):
            self.assertEqual(linux.enumerate_adapters(), [amd])
            c = linux.LinuxCounters()
        self.assertTrue(c.ok)
        for own_context in (True, False):
            holders = {6020, 229213} | ({os.getpid()} if own_context else set())
            with self.subTest(own_context=own_context), \
                    patch.object(linux, "dxg_pids", return_value=holders) as scan:
                procs, _ = c.sample()
                scan.assert_called_once_with()
                self.assertEqual(c.dxg_holders, holders)
                self.assertEqual(set(procs[0x3000]), {6020, 229213})
                self.assertTrue(all(not g.attributed for g in procs[0x3000].values()))

    def test_adapters_come_from_nvml(self):
        with patch.object(linux, "N", FakeNvml()):
            adapters = linux.enumerate_adapters()
        self.assertEqual([(a.luid, a.name, a.dedicated_vram) for a in adapters],
                         [(KEY, "NVIDIA GeForce RTX 3090 Ti", 24 * GiB)])

    def test_a_failing_gpu_is_skipped_not_fatal(self):
        fake = FakeNvml()
        fake.nvmlDeviceGetCount = lambda: 2
        real = fake.nvmlDeviceGetHandleByIndex
        def handle(i):
            if i == 0:
                raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
            return real(0)
        fake.nvmlDeviceGetHandleByIndex = handle
        with patch.object(linux, "N", fake):
            self.assertEqual(len(linux.enumerate_adapters()), 1)
        fake.nvmlDeviceGetCount = lambda: (_ for _ in ()).throw(RuntimeError("NVML_ERROR_UNKNOWN"))
        with patch.object(linux, "N", fake):
            self.assertEqual(linux.enumerate_adapters(), [])

    def test_other_users_processes_stay_local_on_bare_metal(self):
        # Without root, another user's /proc/<pid>/fd is unreadable, so it isn't an NVIDIA holder.
        fake = FakeNvml(compute=[proc(6000, 20 * GiB)])
        c = self.counters(fake, wsl=False)
        with patch.object(linux, "N", fake), patch.object(linux, "nvidia_holders", return_value={100}), \
                patch.object(linux, "has_cuda", return_value=True):
            procs, _ = c.sample()
        self.assertEqual((procs[KEY][6000].dedicated, procs[KEY][6000].inferred_by), (20 * GiB, ""))
        self.assertNotIn(100, procs[KEY])
        self.assertEqual(c.host_procs, {})


class ContainerTests(unittest.TestCase):
    """Measured on a Vast.ai Docker container: NVML lists host PIDs (2029701...) while the processes are
    547 and 563 inside; the host's display server shows up only in the graphics list."""

    def sample(self, c, fake, holders, cuda):
        with patch.object(linux, "N", fake), patch.object(linux, "nvidia_holders", return_value=set(holders)), \
                patch.object(linux, "has_cuda", side_effect=lambda pid: pid in cuda), \
                patch.object(linux, "in_child_pid_namespace", return_value=self.child):
            return c.sample()[0][KEY]

    child = True

    def snapshot(self, fake, detail=None, *, prober=None):
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False):
            counters = linux.LinuxCounters()
        c = Collector.__new__(Collector)
        c.adapter, c.adapters = Adapter(KEY, "NVIDIA", 0x10DE, 24 * GiB, 0, False), []
        c.counters = counters
        c.nvml = SimpleNamespace(ok=False)
        c.drm_card = c.ascend_chip = c.ascend = None
        c.gpu_index, c.scan_budget = 0, None
        c.host, c.host_status, c._mac_driver = None, "", ""
        c.inspector = Mock()
        c.inspector.get.return_value = detail or ProcDetail(547, "python3")
        c.inspector.needs_scan.return_value = False
        c.inspector.exclusive_ports.return_value = c.inspector.get.return_value.ports
        c.prober = prober if prober is not None else Mock()
        if prober is None:
            c.prober.get.return_value = c.prober.locked_port.return_value = None
        c._parent_ports = Mock(return_value=[])
        with patch.object(linux, "N", fake), patch.object(linux, "nvidia_holders", return_value={547}), \
                patch.object(linux, "has_cuda", return_value=True), \
                patch.object(linux, "in_child_pid_namespace", return_value=self.child), \
                patch("gputaskman.collector.BACKEND", "nvml"):
            snap = c.sample()
        c.inspector.get.assert_called_once_with(547)  # never inspect a host PID locally
        return snap

    def test_host_pids_are_matched_to_container_pids_by_creation_order(self):
        fake = FakeNvml(compute=[proc(2029701, GiB), proc(2029747, 2 * GiB)],
                        graphics=[proc(1145, 58 << 20), proc(1360, 21 << 20)],
                        samples=[sample(2029747, 95, ts=3)])
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False):
            c = linux.LinuxCounters()
        # 564 is an NVML-only tool (nvidia-smi, this monitor): it holds the devices but has no CUDA context
        per = self.sample(c, fake, holders={547, 563, 564}, cuda={547, 563})
        self.assertEqual({pid: (g.inferred, g.inferred_by) for pid, g in per.items()},
                         {547: (GiB, "order"), 563: (2 * GiB, "order")})
        self.assertEqual(per[563].engines, {})  # no measured local engine owner
        self.assertEqual(c.host_procs[KEY][2029747].engines["compute"], 95)
        self.assertEqual(set(c.host_procs[KEY]), {1145, 1360, 2029701, 2029747})

    def test_equal_counts_do_not_prove_context_ownership(self):
        # A local libcuda user can hold a device without a context; the sole NVML PID may be another container.
        fake = FakeNvml(compute=[proc(9000, 3 * GiB)])
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False):
            c = linux.LinuxCounters()
        per = self.sample(c, fake, holders={547}, cuda={547})
        self.assertEqual((per[547].inferred, per[547].inferred_by), (3 * GiB, "order"))
        self.assertEqual(set(c.host_procs[KEY]), {9000})
        row = ProcRow(547, "python3", "AI", per[547], ProcDetail(547, "python3"))
        snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=8 * GiB,
                        vram_used=3 * GiB, shared_used=None, backend="nvml", procs=[row])
        self.assertIn("Container process/NVML start correlation only keeps a pair stable",
                      to_text(snap))

    def test_simultaneous_starts_keep_a_stable_pair_without_proving_ownership(self):
        matcher = linux.PidMatcher()
        self.assertEqual(matcher.update([], []), ({}, []))
        self.assertEqual(matcher.update([9000], [547]), ({9000: (547, "order")}, []))
        self.assertEqual(matcher.update([9000, 9001], [547, 600]),
                         ({9000: (547, "order"), 9001: (600, "order")}, []))

    def test_foreign_context_stays_measured_only_under_its_host_pid(self):
        from gputaskman.summary import summarize

        fake = FakeNvml(compute=[proc(9000, 3 * GiB)], samples=[sample(9000, 75)])
        snap = self.snapshot(fake)
        data = to_dict(snap)
        local, host = data["processes"]
        self.assertEqual((local["pid"], local["vram_attributed"], local["vram_bytes"]), (547, False, None))
        self.assertEqual((local["vram_inferred_bytes"], local["vram_inferred_by"]), (3 * GiB, "order"))
        self.assertIsNone(local["gpu_pct"])
        self.assertIsNone(local["engines"])
        self.assertIn("local owner is unresolved", local["engine_unavailable_reason"])
        self.assertEqual((host["pid"], host["host"], host["vram_bytes"], host["gpu_pct"]),
                         (9000, "nvml", 3 * GiB, 75))
        self.assertEqual(data["attributed"], 3 * GiB)  # not counted again for the local guess
        self.assertEqual(snap.procs[1].pid_label, "H:9000")
        text = to_text(snap)
        self.assertIn("3.00 GB?", text)
        self.assertIn("local owner is unresolved", text)
        self.assertNotIn("Windows GPU engine utilization counters", text)
        self.assertNotIn("outside this container", text)
        peaks = {p["host"]: p for p in summarize([data])["runs"][0]["processes"]}
        self.assertIsNone(peaks[None]["peak_vram_bytes"])
        self.assertIsNone(peaks[None]["peak_gpu_pct"])
        self.assertEqual(peaks["nvml"]["peak_vram_bytes"], 3 * GiB)

    def test_unproven_bytes_do_not_constrain_model_breakdown(self):
        d = ProcDetail(547, "llama-server", cmdline=["llama-server", "-c", "4096"],
                       models=[MappedModel("/m/x.gguf", 8 * GiB, None, True)])
        model = GgufInfo(path="/m/x.gguf", file_size=8 * GiB, version=3, tensor_bytes=8 * GiB,
                         metadata={"general.architecture": "llama", "llama.block_count": 32,
                                   "llama.context_length": 4096, "llama.attention.head_count": 32,
                                   "llama.embedding_length": 4096})
        for memory in (3 * GiB, 12 * GiB, None):
            with self.subTest(memory=memory), \
                    patch("gputaskman.llm.read_gguf", return_value=model), \
                    patch("gputaskman.collector.build_view", wraps=build_view) as build:
                snap = self.snapshot(FakeNvml(compute=[proc(9000, memory)],
                                             samples=[sample(9000, 75)]), detail=d)
                view = snap.procs[0].llm
                self.assertIsNotNone(view)
                build.assert_called_once()
                self.assertEqual(build.call_args.args[1:3], (0, 0))
                self.assertFalse(build.call_args.kwargs["attributed"])
                self.assertEqual(view.models, [model])
                self.assertEqual(view.files[0].sources, ["mapped"])
                self.assertEqual([(s.kind, s.bytes) for s in view.segments],
                                 [("weights", 8 * GiB), ("kv", 2 * GiB)])
                self.assertTrue(all(not s.exact for s in view.segments))
                self.assertTrue(any("VRAM total is unknown" in n for n in view.notes))
                self.assertIn("evidence: in use (mapped)", to_text(snap))
                local, host = to_dict(snap)["processes"]
                self.assertFalse(local["vram_attributed"])
                self.assertIsNone(local["vram_bytes"])
                self.assertIsNone(local["engines"])
                self.assertEqual(local["vram_inferred_bytes"], memory)
                self.assertEqual(host["vram_bytes"], memory)
                self.assertEqual(snap.procs[0].gpu.available, set())
                self.assertEqual(snap.attributed, memory)

    def test_recognized_runtime_probes_own_listener_without_model_files(self):
        base = "http://127.0.0.1:8080"
        responses = {
            base + "/props": {"default_generation_settings": {"n_ctx": 4096}},
            base + "/v1/models": {"data": [{"id": "api-only-model"}]},
            base + "/slots": [],
            base + "/metrics": None,
        }
        for name, runtimes in (("llama-server", []), ("python3", ["llama.cpp"])):
            for memory in (3 * GiB, None):
                with self.subTest(name=name, memory=memory), \
                        patch("gputaskman.llm._get", side_effect=lambda url, **kw: responses[url]) as get, \
                        patch("gputaskman.collector.build_view", wraps=build_view) as build:
                    d = ProcDetail(547, name, runtimes=runtimes, ports=[8080])
                    snap = self.snapshot(FakeNvml(compute=[proc(9000, memory)],
                                                 samples=[sample(9000, 75)]), d, prober=ServerProber())
                    self.assertEqual({call.args[0] for call in get.call_args_list}, set(responses))
                    self.assertEqual(get.call_count, 4)
                    self.assertEqual(build.call_args.args[1:3], (0, 0))
                    self.assertFalse(build.call_args.kwargs["attributed"])
                    local, host = to_dict(snap)["processes"]
                    self.assertEqual(local["kind"], "LLM")
                    self.assertEqual(local["llm"]["server"]["models"][0]["name"], "api-only-model")
                    self.assertEqual(local["llm"]["server_source"], "process")
                    self.assertEqual(local["llm"]["segments"], [])
                    self.assertTrue(any("VRAM total is unknown" in n for n in local["llm"]["notes"]))
                    self.assertFalse(local["vram_attributed"])
                    self.assertIsNone(local["vram_bytes"])
                    self.assertEqual(local["vram_inferred_bytes"], memory)
                    self.assertIsNone(local["engines"])
                    self.assertEqual(snap.procs[0].gpu.available, set())
                    self.assertEqual((host["vram_bytes"], host["gpu_pct"]), (memory, 75))
                    self.assertEqual(snap.attributed, memory)

    def test_generic_process_with_only_model_hints_is_not_classified_or_probed(self):
        for sources in ([], ["path-arg"], ["installed"]):
            with self.subTest(sources=sources), patch("gputaskman.collector.build_view") as build, \
                    patch("gputaskman.llm._get") as get, patch("gputaskman.llm.read_gguf") as read:
                d = ProcDetail(547, "python3", runtimes=["CUDA"], ports=[8080],
                               models=[MappedModel("/m/x.gguf", GiB, None, True, sources=sources, kind="LLM")])
                snap = self.snapshot(FakeNvml(compute=[proc(9000, 3 * GiB)]), d, prober=ServerProber())
                self.assertEqual(snap.procs[0].kind, "compute")
                self.assertIsNone(snap.procs[0].llm)
                build.assert_not_called()
                get.assert_not_called()
                read.assert_not_called()

    def test_api_daemon_does_not_use_guessed_memory_as_a_budget(self):
        # An independently detected API daemon may still report loaded IDs, without a guessed budget.
        for memory in (3 * GiB, None):
            with self.subTest(memory=memory), \
                    patch("gputaskman.collector.build_view", return_value=LlmView()) as build:
                snap = self.snapshot(FakeNvml(compute=[proc(9000, memory)]),
                                     detail=ProcDetail(547, "foundrylocald"))
                self.assertIsNotNone(snap.procs[0].llm)
                build.assert_called_once()
                self.assertEqual(build.call_args.args[1:3], (0, 0))
                self.assertFalse(build.call_args.kwargs["attributed"])

    def test_host_memory_availability_is_independent_of_host_engines(self):
        for memory, expected in ((None, None), (NOT_AVAILABLE, None), (0, 0)):
            with self.subTest(memory=memory):
                snap = self.snapshot(FakeNvml(compute=[proc(9000, memory)], samples=[sample(9000, 47)]))
                data = to_dict(snap)
                local, host = data["processes"]
                self.assertEqual(data["attributed"], expected)
                self.assertIsNone(local["vram_bytes"])
                self.assertEqual(local["vram_inferred_bytes"], expected)
                self.assertIsNone(local["gpu_pct"])
                self.assertEqual(host["vram_bytes"], expected)
                self.assertEqual(host["gpu_pct"], 47)
                self.assertIn("local owner is unresolved", to_text(snap))

    def test_known_local_memory_is_independent_of_engine_availability(self):
        self.child = False
        for memory in (None, NOT_AVAILABLE, 0, 3 * GiB):
            with self.subTest(memory=memory):
                snap = self.snapshot(FakeNvml(compute=[proc(547, memory)], samples=[sample(547, 47)]))
                data = to_dict(snap)
                row, = data["processes"]
                expected = None if memory in (None, NOT_AVAILABLE) else memory
                self.assertEqual(data["attributed"], expected)
                self.assertEqual(row["vram_bytes"], expected)
                self.assertIsNone(row["vram_inferred_bytes"])
                self.assertEqual(row["gpu_pct"], 47)
                self.assertIsNone(row["engine_unavailable_reason"])
                self.assertIsNone(row["host"])

    def test_unknown_namespace_does_not_make_an_equal_pid_a_measured_local_owner(self):
        self.child = None
        data = to_dict(self.snapshot(FakeNvml(compute=[proc(547, 3 * GiB)], samples=[sample(547, 47)])))
        local, host = data["processes"]
        self.assertEqual((local["pid"], local["vram_bytes"], local["gpu_pct"]), (547, None, None))
        self.assertEqual((host["pid"], host["host"], host["vram_bytes"]), (547, "nvml", 3 * GiB))
        self.assertEqual(data["attributed"], 3 * GiB)

    def test_extra_outside_compute_stays_a_guess_after_a_start_event(self):
        fake = FakeNvml(compute=[proc(9000, GiB), proc(2029701, 3 * GiB)])  # 9000: another container
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False):
            c = linux.LinuxCounters()
        per = self.sample(c, fake, holders={547}, cuda={547})
        self.assertEqual((per[547].inferred, per[547].inferred_by), (3 * GiB, "order"))
        self.assertEqual(set(c.host_procs[KEY]), {9000, 2029701})
        fake.compute.append(proc(2029800, 2 * GiB))  # a new local job starts: one new PID on each side
        per = self.sample(c, fake, holders={547, 600}, cuda={547, 600})
        self.assertEqual((per[600].inferred, per[600].inferred_by), (2 * GiB, "order"))

    def test_bare_metal_is_unchanged(self):
        fake = FakeNvml(compute=[proc(547, GiB)])
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False):
            c = linux.LinuxCounters()
        self.child = False
        per = self.sample(c, fake, holders={547}, cuda={547})
        self.assertEqual((per[547].dedicated, per[547].inferred_by), (GiB, ""))
        self.assertEqual(c.host_procs, {})

    def test_a_host_pid_equal_to_a_local_pid_is_still_a_host_pid(self):
        # Host PIDs 100 and 200 are container PIDs 10 and 100: local PID 100 is not host PID 100.
        fake = FakeNvml(compute=[proc(100, GiB), proc(200, 4 * GiB)])
        with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False):
            c = linux.LinuxCounters()
        per = self.sample(c, fake, holders={10, 100}, cuda={10, 100})
        self.assertEqual({pid: (g.inferred, g.inferred_by) for pid, g in per.items()},
                         {10: (GiB, "order"), 100: (4 * GiB, "order")})

    def test_namespace_detection(self):
        linux.in_child_pid_namespace.cache_clear()
        self.addCleanup(linux.in_child_pid_namespace.cache_clear)
        with patch.object(linux.os, "readlink", return_value="pid:[4026531836]"):
            self.assertFalse(linux.in_child_pid_namespace())
        linux.in_child_pid_namespace.cache_clear()
        with patch.object(linux.os, "readlink", return_value="pid:[4026532481]"):
            self.assertTrue(linux.in_child_pid_namespace())
        linux.in_child_pid_namespace.cache_clear()
        with patch.object(linux.os, "readlink", side_effect=OSError):
            self.assertIsNone(linux.in_child_pid_namespace())

    def test_outside_rows_never_borrow_a_local_identity(self):
        from gputaskman.collector import ProcRow as Row
        row = Row(1145, "(outside this container)", "external", ProcGpu(dedicated=58 << 20),
                  ProcDetail(1145, "(outside this container)"), host=True, origin="outside")
        self.assertEqual((row.key, row.pid_label), ("o1145", "H:1145"))
        snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=8 * GiB,
                        vram_used=GiB, shared_used=0, backend="nvml", procs=[row])
        self.assertIn("H:1145", to_text(snap))
        self.assertEqual(to_dict(snap)["processes"][0]["host"], "outside")


class UnattributedReportTests(unittest.TestCase):
    def snapshot(self):
        return Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=24 * GiB,
                        vram_used=7 * GiB, shared_used=0, backend="wsl",
                        procs=[ProcRow(4242, "python3", "AI", ProcGpu(attributed=False), ProcDetail(4242, "python3"))])

    def test_reports_say_unknown_instead_of_zero(self):
        data = to_dict(self.snapshot())
        self.assertEqual(data["backend"], "wsl")
        for key in ("vram_bytes", "shared_bytes", "committed_bytes", "paged_out_bytes", "gpu_pct", "engines"):
            self.assertIsNone(data["processes"][0][key], key)
        self.assertFalse(data["processes"][0]["vram_attributed"])
        text = to_text(self.snapshot())
        row = next(line for line in text.splitlines() if "4242" in line)
        self.assertIn("n/a", row)
        self.assertNotIn("0 B", row)
        self.assertIn("WSL:", text)
        self.assertNotIn("shared (in system RAM)", text)

    def test_json_unknown_counters_are_null_not_zero(self):
        snap = self.snapshot()
        win = ProcRow(2744, "dwm.exe", "compositor", ProcGpu(dedicated=GiB, shared=6 << 20, committed=2 * GiB),
                      ProcDetail(2744, "dwm.exe"), host=True, origin="windows")
        snap.procs.append(win)
        linux_row, win_row = to_dict(snap)["processes"]
        for k in ("shared_bytes", "committed_bytes", "paged_out_bytes", "gpu_pct", "engines"):
            self.assertIsNone(linux_row[k], k)
        self.assertEqual((win_row["shared_bytes"], win_row["committed_bytes"]), (6 << 20, 2 * GiB))
        self.assertEqual(win_row["gpu_pct"], 0.0)  # measured idle
        snap.backend = "nvml"  # native Linux: NVML measures VRAM and load, but there is no spill/commit
        snap.procs = [ProcRow(547, "python3", "AI", ProcGpu(dedicated=GiB), ProcDetail(547, "python3"))]
        row = to_dict(snap)["processes"][0]
        self.assertEqual((row["vram_bytes"], row["gpu_pct"], row["shared_bytes"]), (GiB, 0.0, None))

    def test_wsl_rows_sort_by_the_size_they_show(self):
        from gputaskman.app import GpuTaskManager
        rows = [ProcRow(p, "python3", "AI", ProcGpu(attributed=False, inferred=v, inferred_by="consistent"),
                        ProcDetail(p, "python3")) for p, v in ((1, GiB), (2, 5 * GiB), (3, None))]
        key = GpuTaskManager.SORTS[0][1]
        self.assertEqual([r.pid for r in sorted(rows, key=key)], [2, 1, 3])

    def test_native_hidden_vram_does_not_hide_measured_engine_load(self):
        snap = self.snapshot()
        snap.backend = "nvml"
        snap.procs[0].gpu.engines["compute"] = 47
        row = to_dict(snap)["processes"][0]
        self.assertIsNone(row["vram_bytes"])
        self.assertEqual(row["gpu_pct"], 47)
        self.assertEqual(row["engines"], {"compute": 47})
        for key in ("shared_bytes", "committed_bytes", "paged_out_bytes"):
            self.assertIsNone(row[key], key)
        text = next(line for line in to_text(snap).splitlines() if "4242" in line)
        self.assertIn("47", text)

    def test_measured_zero_is_not_changed_to_unknown(self):
        snap = self.snapshot()
        snap.backend = "nvml"
        snap.procs[0].gpu.attributed = True
        row = to_dict(snap)["processes"][0]
        self.assertEqual(row["vram_bytes"], 0)
        self.assertEqual(row["gpu_pct"], 0)
        self.assertEqual(row["engines"], {})
        self.assertIsNone(row["shared_bytes"])
        snap.backend = "wsl"
        snap.procs[0].host, snap.procs[0].origin = True, "windows"
        row = to_dict(snap)["processes"][0]
        for key in ("vram_bytes", "shared_bytes", "committed_bytes", "paged_out_bytes", "gpu_pct"):
            self.assertEqual(row[key], 0, key)

    def test_model_detail_header_uses_inferred_or_unknown_vram(self):
        for inferred, label in ((None, "n/a"), (0, "0 MB*"), (3 * GiB, "3.00 GB*")):
            with self.subTest(inferred=inferred):
                snap = self.snapshot()
                p = snap.procs[0]
                p.llm = LlmView()
                p.gpu.inferred, p.gpu.inferred_by = inferred, "consistent"
                self.assertIn(f"== python3 (4242)  VRAM {label}", to_text(snap))
                self.assertIsNone(to_dict(snap)["processes"][0]["vram_bytes"])

    def test_empty_wsl_view_explains_itself(self):
        snap = self.snapshot()
        snap.procs = []
        self.assertIn("No Linux process", to_text(snap))

    def test_llm_view_does_not_claim_cpu_inference_when_vram_is_unknown(self):
        d = ProcDetail(4242, "llama-server", cmdline=["llama-server", "-c", "4096"])
        d.models = [MappedModel("/m/x.gguf", GiB, 0, True)]
        with patch("gputaskman.llm.read_gguf", return_value=GgufInfo(path="/m/x.gguf", file_size=GiB, version=3)):
            known = build_view(d, 0, 0, None)
            unknown = build_view(d, 0, 0, None, attributed=False)
        self.assertTrue(any("CPU inference" in n for n in known.notes))
        self.assertFalse(any("CPU inference" in n for n in unknown.notes))
        self.assertTrue(any("unknown" in n for n in unknown.notes))


if __name__ == "__main__":
    unittest.main()
