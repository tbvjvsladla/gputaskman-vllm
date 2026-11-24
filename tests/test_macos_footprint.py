"""Provider-to-snapshot coverage for macOS per-process footprint availability."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gputaskman import macos
from gputaskman.collector import Collector, _llm_wanted
from gputaskman.procinfo import ProcDetail
from gputaskman.report import to_dict

GiB = 1 << 30
ME, OTHER = 501, 502


def accel(allocated=9 * GiB):
    return [{"IOClass": "AGXAcceleratorG16G", "model": "Apple M4", "gpu-core-count": 10,
             "PerformanceStatistics": {"Device Utilization %": 75, "Alloc system memory": allocated}}]


def client(pid, name, ticks, cid):
    return {"IOUserClientCreator": f"pid {pid}, {name}",
            "AppUsage": [{"API": "Metal", "accumulatedGPUTime": ticks}], "IORegistryEntryID": cid}


class Inspector:
    def get(self, pid):
        return ProcDetail(pid, "python")

    def needs_scan(self, detail, memory):
        return False

    def prune(self, pids):
        pass


def fixture_collector(counters):
    collector = Collector.__new__(Collector)
    collector.adapter = None
    collector.adapters = []
    collector.counters = counters
    collector.host = None
    collector.host_status = ""
    collector.scan_budget = 0
    collector.inspector = Inspector()
    return collector


class MacFootprintCollectorTests(unittest.TestCase):
    def test_idle_other_user_client_hides_row_but_invalidates_total_until_gone(self):
        now = 0.0
        idle = [client(103, "backupd", 0, 4)]
        poller = macos.FootprintPoller(run=lambda *args, **kwargs: self.fail("unexpected footprint probe"))
        with patch.object(poller, "start"), \
                patch.object(macos, "time", SimpleNamespace(monotonic=lambda: now)), \
                patch.object(macos, "_ioreg", side_effect=[accel(), idle, accel(), idle, accel(), []]), \
                patch.object(macos, "physical_memory", return_value=16 * GiB), \
                patch.object(macos.MacCounters, "_name", return_value="backupd"), \
                patch("gputaskman.collector.BACKEND", "metal"):
            counters = macos.MacCounters(
                poller=poller, uid=ME, root=False, identity=lambda pid: "same", owner=lambda pid: OTHER,
            )
            self.addCleanup(counters.close)
            collector = fixture_collector(counters)
            now += 2
            hidden = collector.sample()
            now += 2
            recovered = collector.sample()

        for snap, attributed in ((hidden, None), (recovered, 0)):
            data = to_dict(snap)
            self.assertEqual(data["processes"], [])
            self.assertEqual(snap.attributed, attributed)
            self.assertEqual(data["attributed"], attributed)
            self.assertEqual(data["vram_used"], 9 * GiB)
            self.assertEqual(snap.nv.util_gpu, 75)
        self.assertIn("idle Metal clients", to_dict(hidden)["counter_status"])
        self.assertIsNone(to_dict(recovered)["counter_status"])

    def test_unknown_zero_and_recovered_footprints_through_json(self):
        measurements = {100: 4 * GiB, 101: None, 102: 0}
        now = 0.0
        fake_time = SimpleNamespace(monotonic=lambda: now)

        def run(args, **kwargs):
            pid = int(args[-1])
            size = measurements.get(pid)
            if size is None:
                return
            with open(args[args.index("-j") + 1], "w", encoding="utf-8") as f:
                json.dump({"processes": [{"pid": pid, "categories": {
                    "IOAccelerator (graphics)": {"dirty": size}}}]}, f)

        poller = macos.FootprintPoller(run=run, clock=fake_time.monotonic)
        poller.start = lambda: None
        with patch.object(macos, "_ioreg", return_value=[]), \
                patch.object(macos, "time", fake_time), \
                patch.object(macos, "physical_memory", return_value=16 * GiB):
            counters = macos.MacCounters(
                poller=poller, uid=ME, root=False, identity=lambda pid: "same",
                owner=lambda pid: OTHER if pid == 103 else ME,
            )
        counters._names[103] = "WindowServer"

        collector = fixture_collector(counters)

        def clients(ticks, include_other=True):
            entries = [
                client(100, "python", ticks, 1),
                client(101, "llama-server", ticks, 2),
                client(102, "python-zero", ticks, 3),
                client(103, "WindowServer", ticks, 4),
            ]
            return entries if include_other else entries[:3]

        with patch("gputaskman.collector.BACKEND", "metal"), \
                patch.object(macos, "time", fake_time), \
                patch.object(macos, "_ioreg", side_effect=[
                    accel(), clients(10_000_000_000), accel(), clients(11_000_000_000),
                    accel(), clients(12_000_000_000, False), accel(), clients(13_000_000_000, False),
                    accel(), clients(14_000_000_000, False),
                ]), \
                patch.object(macos, "physical_memory", return_value=16 * GiB):
            now += poller.CYCLE
            pending = collector.sample()
            pending_row = next(row for row in pending.procs if row.pid == 101)
            self.assertNotIn("dedicated", pending_row.gpu.available)
            self.assertIn("engine", pending_row.gpu.available)
            self.assertIsNone(pending.attributed)
            pending_data = to_dict(pending)
            pending_process = next(p for p in pending_data["processes"] if p["pid"] == 101)
            self.assertIsNone(pending_process["vram_bytes"])
            self.assertIsNotNone(pending_process["gpu_pct"])
            # The first pass measures 100 and 102 but cannot measure 101.
            poller.poll_once()
            now += poller.CYCLE
            measured = collector.sample()
            row101 = next(row for row in measured.procs if row.pid == 101)
            row101.kind = "LLM"

            measured_data = to_dict(measured)
            self.assertIsNone(measured.attributed)
            self.assertEqual(measured.vram_used, 9 * GiB)
            processes = {p["pid"]: p for p in measured_data["processes"]}
            self.assertEqual(processes[100]["vram_bytes"], 4 * GiB)
            self.assertIsNone(processes[101]["vram_bytes"])
            self.assertIsNone(processes[103]["vram_bytes"])
            self.assertFalse(_llm_wanted(row101, True))
            self.assertIn("engine", row101.gpu.available)
            zero = next(row for row in measured.procs if row.pid == 102)
            self.assertTrue(zero.gpu.attributed)
            self.assertIn("dedicated", zero.gpu.available)
            self.assertEqual(zero.gpu.dedicated, 0)
            self.assertEqual(processes[102]["vram_bytes"], 0)

            # The pending own-user footprint succeeds; the inaccessible other-user process exits.
            measurements[101] = 2 * GiB
            poller.poll_once()
            now += poller.CYCLE
            recovered = collector.sample()
            self.assertEqual(recovered.attributed, 6 * GiB)

            # A positive sample interval keeps 101 busy, so the idle schedule cannot skip these passes.
            measurements[101] = None
            for _ in range(2):
                now += poller.CYCLE
                poller.poll_once()
                self.assertEqual(poller.get(101, "same"), (2 * GiB, None))
            now += poller.CYCLE
            poller.poll_once()
            failed = collector.sample()
            failed_row = next(row for row in failed.procs if row.pid == 101)
            self.assertIsNone(failed.attributed)
            self.assertNotIn("dedicated", failed_row.gpu.available)
            self.assertIn("engine", failed_row.gpu.available)

            # A later successful pass restores measured attribution.
            now += poller.FAIL_RETRY
            measurements[101] = 2 * GiB
            poller.poll_once()
            recovered_again = collector.sample()
            self.assertEqual(recovered_again.attributed, 6 * GiB)
            self.assertEqual(to_dict(recovered_again)["attributed"], 6 * GiB)

        counters.close()


if __name__ == "__main__":
    unittest.main()
