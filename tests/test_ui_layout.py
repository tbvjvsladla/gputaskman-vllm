"""Tests for TUI 3-pane layout cycling (shortcut 'l') and vLLM Memory Breakdown rendering."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from rich.text import Text
from textual.widgets import DataTable, Footer, Static

from gputaskman.app import HELP_MD, PANE_LAYOUT_LABELS, PANE_LAYOUTS, GpuTaskManager
from gputaskman.collector import ProcRow, Snapshot
from gputaskman.dxgi import Adapter
from gputaskman.llm import LlmView, Segment, ServerInfo, VllmInitProfile, build_view
from gputaskman.nvml import NvDevice
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import ProcDetail

MiB = 1 << 20
GiB = 1 << 30


def _nv_snapshot(rows: list[ProcRow], *, vram_used: int = 6 * GiB, vram_total: int = 8 * GiB) -> Snapshot:
    adapter = Adapter(1, "NVIDIA GeForce RTX 4070", 0x10DE, vram_total, 0, False)
    return Snapshot(
        taken=time.time(),
        adapter=adapter,
        adapters=[adapter],
        nv=NvDevice(index=0, name="NVIDIA GeForce RTX 4070", util_gpu=35, temp=52, power_w=110, power_limit_w=200),
        driver="560.35",
        cuda="12.6",
        vram_total=vram_total,
        vram_used=vram_used,
        shared_used=0,
        procs=rows,
        backend="nvml",
    )


def _vllm_row(
    pid: int = 100,
    *,
    process_vram: int = 6 * GiB,
    attributed: bool = True,
    vram_measured: bool = True,
    device_vram_used: int | None = 7 * GiB,
    device_vram_total: int | None = 8 * GiB,
    init_profile: VllmInitProfile | None = None,
    server: ServerInfo | None = None,
    server_source: str | None = "parent",
    endpoint_url: str | None = "http://127.0.0.1:8000",
    endpoint_name: str | None = "local-vllm",
) -> ProcRow:
    detail = ProcDetail(
        pid=pid,
        name="VLLM::EngineCore",
        runtimes=["CUDA", "PyTorch"],
        cmdline=["VLLM::EngineCore"],
        create_time=1_700_000_000.0,
    )
    gpu = ProcGpu(dedicated=process_vram if attributed else 0, shared=0, attributed=attributed)
    view = build_view(
        detail,
        process_vram if attributed else 0,
        0,
        server=server,
        attributed=attributed,
        server_source=server_source if server else None,
        vllm_init_profile=init_profile,
        vllm_endpoint_url=endpoint_url,
        vllm_endpoint_name=endpoint_name,
        gpu_id="0",
        device_vram_used=device_vram_used,
        device_vram_total=device_vram_total,
        vram_measured=vram_measured,
        sample_time=1_700_000_100.0,
    )
    return ProcRow(
        pid=pid,
        name="VLLM::EngineCore",
        kind="LLM",
        gpu=gpu,
        detail=detail,
        llm=view,
    )


def _plain_inspector_text(app: GpuTaskManager) -> str:
    content = app.query_one("#inspector", Static).content
    return "\n".join(getattr(line, "plain", "") for line in content.renderables)


class PaneLayoutCycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_cycles_four_combinations_and_preserves_footer_selection_focus(self):
        rows = [
            ProcRow(10, "python", "compute", ProcGpu(dedicated=2 * GiB, shared=0), ProcDetail(10, "python", runtimes=["CUDA"])),
            ProcRow(20, "worker", "compute", ProcGpu(dedicated=1 * GiB, shared=0), ProcDetail(20, "worker", runtimes=["CUDA"])),
        ]
        snap = _nv_snapshot(rows, vram_used=3 * GiB, vram_total=8 * GiB)

        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            async with app.run_test(size=(160, 46)) as pilot:
                app.apply(snap)
                await pilot.pause()

                # Select the second row so we can confirm selection & focus persist across layout changes.
                await pilot.press("down")
                await pilot.pause()
                self.assertEqual(app.selected_key, "20")
                self.assertEqual(app.focused.id, "procs")

                expected_sequence = list(zip(PANE_LAYOUTS, PANE_LAYOUT_LABELS)) + [(PANE_LAYOUTS[0], PANE_LAYOUT_LABELS[0])]
                for step, (expected_order, expected_label) in enumerate(expected_sequence):
                    if step > 0:
                        await pilot.press("l")
                        await pilot.pause()

                    self.assertEqual(app.layout_order, expected_order)
                    self.assertEqual(app.layout_label, expected_label)
                    self.assertIn(expected_label, str(app.query_one("#inspector", Static).border_subtitle))

                    # Verify DOM child order matches layout_order with Footer last.
                    children = [c for c in app.screen.children if c.id in ("dashboard", "procs", "inspector") or isinstance(c, Footer)]
                    pane_ids = tuple(c.id for c in children[:-1])
                    self.assertEqual(pane_ids, expected_order)
                    self.assertIsInstance(children[-1], Footer)

                    # Verify vertical coordinates (region.y) strictly follow layout_order and Footer stays at bottom.
                    ys = [app.query_one(f"#{pane_id}").region.y for pane_id in expected_order]
                    self.assertEqual(ys, sorted(ys))
                    self.assertLess(ys[0], ys[1])
                    self.assertLess(ys[1], ys[2])

                    footer = app.query_one(Footer)
                    for pane_id in expected_order:
                        self.assertLessEqual(app.query_one(f"#{pane_id}").region.bottom, footer.region.y)
                    self.assertEqual(footer.region.bottom, 46)

                    # Selection and focus remain intact.
                    self.assertEqual(app.selected_key, "20")
                    self.assertIsNotNone(app.focused)
                    self.assertEqual(app.focused.id, "procs")

                # Switch to (3,1,2) and verify resize keeps layout order, #procs min height, and Footer bottom.
                await pilot.press("l", "l")
                await pilot.pause()
                self.assertEqual(app.layout_order, ("inspector", "dashboard", "procs"))
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                ys = [app.query_one(f"#{pane_id}").region.y for pane_id in app.layout_order]
                self.assertLess(ys[0], ys[1])
                self.assertLess(ys[1], ys[2])
                self.assertGreaterEqual(app.query_one("#procs", DataTable).region.height, 4)
                footer = app.query_one(Footer)
                self.assertLessEqual(app.query_one("#procs", DataTable).region.bottom, footer.region.y)
                self.assertLessEqual(footer.region.bottom, 24)

    async def test_footer_binding_and_help_document_layout_and_vllm_breakdown(self):
        bindings = {b.key: (b.action, b.description) for b in GpuTaskManager.BINDINGS}
        self.assertIn("l", bindings)
        self.assertEqual(bindings["l"], ("cycle_layout", "Layout"))
        for token in ("`l` layout", "(1,2,3)", "(1,3,2)", "(3,1,2)", "(3,2,1)", "Activation Peak", "KV Cache Unused"):
            self.assertIn(token, HELP_MD)


class VllmTwoBarBreakdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_strategy_b_two_bar_breakdown_and_non_overlapping_bar1(self):
        init = VllmInitProfile(
            path="/var/log/vllm.log",
            weights_bytes=2 * GiB,
            kv_allocated_bytes=3 * GiB,
            activation_peak_bytes=512 * MiB,
            cuda_graph_bytes=256 * MiB,
            kv_cache_size_tokens=16384,
            num_gpu_blocks=1024,
            timestamp="2026-11-24T12:00:10+00:00",
            stale=False,
        )
        server = ServerInfo(
            kind="vllm",
            url="http://127.0.0.1:8000",
            models=[{"name": "Qwen/Qwen2.5-0.5B"}],
            requests_running=2,
            requests_waiting=1,
            kv_cache_usage=0.4,
            cache_config={
                "block_size": 16,
                "cache_dtype": "auto",
                "enable_prefix_caching": True,
                "num_gpu_blocks": 1024,
                "kv_cache_size_tokens": 16384,
            },
            probed_at=1_700_000_100.0,
        )
        row = _vllm_row(
            process_vram=6 * GiB,
            device_vram_used=7 * GiB,
            device_vram_total=8 * GiB,
            init_profile=init,
            server=server,
            server_source="config",
        )
        # Bar 1 segments must be non-overlapping and sum exactly to process VRAM (6 GiB),
        # never adding activation_peak_bytes (512 MiB) or kv_pool_occupied_bytes (1.2 GiB).
        self.assertEqual(sum(seg.bytes for seg in row.llm.segments), 6 * GiB)
        self.assertNotIn("compute", [seg.kind for seg in row.llm.segments])

        snap = _nv_snapshot([row], vram_used=7 * GiB, vram_total=8 * GiB)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "100"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()

                renderables = app.query_one("#inspector", Static).content.renderables
                self.assertTrue(all(isinstance(item, Text) for item in renderables))
                text = _plain_inspector_text(app)

                self.assertIn("vLLM Memory Breakdown", text)
                self.assertIn("local-vllm (http://127.0.0.1:8000)", text)
                self.assertIn("Process VRAM", text)
                self.assertIn("6.00 GB / 8.00 GB", text)
                self.assertIn("[NVML]", text)
                self.assertIn("Weights", text)
                self.assertIn("2.00 GB  [Init profile]", text)
                self.assertIn("KV Cache Allocated", text)
                self.assertIn("3.00 GB  [Init profile]", text)
                self.assertIn("Other/Unattributed", text)
                self.assertIn("1.00 GB  [NVML diff, derived]", text)
                self.assertIn("Activation Peak", text)
                self.assertIn("512 MB  [Init peak, NOT current]", text)
                self.assertIn("Activation Current", text)
                self.assertIn("N/A  [requires instrumentation]", text)
                self.assertIn("CUDA Graph", text)
                self.assertIn("256 MB  [Init profile]", text)
                self.assertNotIn("KV Cache Pool", text)
                self.assertNotIn("Occupied 40.0% · Available 60.0%", text)
                self.assertIn("1.20 GB  (40.0%)  [derived; includes prefix cache]", text)
                self.assertIn("1.80 GB  [derived]", text)
                self.assertIn("Active Requests KV", text)
                self.assertIn("Running / Waiting", text)
                self.assertIn("2 / 1", text)
                self.assertIn("blocks 1,024 · block_size 16 · capacity 16,384 tokens · dtype auto · prefix_cache on", text)
                self.assertIn("Sources: NVML + Init Profile + Metrics", text)
                self.assertIn("Timestamps:", text)
                self.assertIn("(configured endpoint; API report, not worker residency)", text)

                # Verify updated hex colors for Activation Peak (#60afff) and KV Cache Unused (#8e6202).
                all_styles = [str(span.style) for item in renderables for span in item.spans]
                self.assertIn("#60afff", all_styles)
                self.assertIn("bold #60afff", all_styles)
                self.assertIn("#8e6202", all_styles)
                self.assertIn("bold #8e6202", all_styles)
                self.assertFalse(any("orange1" in s or "aquamarine1" in s for s in all_styles))

    async def test_zero_values_preserved_and_not_coerced_to_na(self):
        init = VllmInitProfile(
            path="/var/log/vllm.log",
            weights_bytes=2 * GiB,
            kv_allocated_bytes=4 * GiB,
            stale=False,
        )
        server = ServerInfo(
            kind="vllm",
            url="http://127.0.0.1:8000",
            models=[{"name": "org/model"}],
            requests_running=0,
            requests_waiting=0,
            kv_cache_usage=0.0,
            probed_at=1_700_000_100.0,
        )
        row = _vllm_row(process_vram=6 * GiB, init_profile=init, server=server)
        snap = _nv_snapshot([row], vram_used=6 * GiB, vram_total=8 * GiB)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "100"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = _plain_inspector_text(app)
                self.assertIn("0 MB  [NVML diff, derived]", text)
                self.assertNotIn("KV Cache Pool", text)
                self.assertIn("0 MB  (0.0%)  [derived; includes prefix cache]", text)
                self.assertIn("4.00 GB  [derived]", text)
                self.assertIn("0 / 0", text)
                self.assertIn("Activation Current", text)
                self.assertIn("Active Requests KV", text)

    async def test_metrics_unavailable_fallback_omits_metrics_source(self):
        server = ServerInfo(
            kind="vllm",
            url="http://127.0.0.1:8000",
            models=[{"name": "org/model"}],
            probed_at=1_700_000_100.0,
        )
        row = _vllm_row(process_vram=4 * GiB, init_profile=None, server=server, server_source="parent")
        snap = _nv_snapshot([row], vram_used=4 * GiB, vram_total=8 * GiB)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "100"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = _plain_inspector_text(app)
                self.assertNotIn("KV Cache Pool", text)
                self.assertIn("[metrics unavailable]", text)
                self.assertIn("N/A / N/A", text)
                self.assertIn("Sources: NVML", text)
                self.assertNotIn("Sources: NVML + Metrics", text)
                self.assertNotIn("vLLM /metrics", text)

    async def test_stale_init_log_withholds_bar1_and_marks_stale(self):
        init = VllmInitProfile(
            path="/var/log/vllm.log",
            weights_bytes=2 * GiB,
            kv_allocated_bytes=3 * GiB,
            activation_peak_bytes=400 * MiB,
            stale=True,
            stale_reason="init log predates process start",
        )
        row = _vllm_row(process_vram=6 * GiB, init_profile=init, server=None)
        snap = _nv_snapshot([row], vram_used=6 * GiB, vram_total=8 * GiB)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "100"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = _plain_inspector_text(app)
                self.assertIn("[STALE]", text)
                self.assertIn("VRAM Composition: withheld (init log stale; excluded from VRAM segments)", text)
                self.assertIn("[Init profile, STALE]", text)
                self.assertIn("[Init peak, NOT current, STALE]", text)
                self.assertIn("Other/Unattributed", text)
                self.assertIn("N/A  [requires reconciliation]", text)
                self.assertIn("! STALE init log: init log predates process start", text)

    async def test_unattributed_wsl_vram_renders_profile_stacked_bar_and_defers_order_guess(self):
        init = VllmInitProfile(
            path="/var/log/vllm.log",
            weights_bytes=2 * GiB,
            kv_allocated_bytes=3 * GiB,
            activation_peak_bytes=512 * MiB,
            cuda_graph_bytes=128 * MiB,
            stale=False,
        )
        server = ServerInfo(
            kind="vllm",
            url="http://127.0.0.1:8000",
            models=[{"name": "Qwen/Qwen2.5-0.5B-Instruct"}],
            requests_running=1,
            requests_waiting=0,
            kv_cache_usage=0.25,
            probed_at=1_700_000_100.0,
        )
        row = _vllm_row(
            process_vram=0,
            attributed=False,
            vram_measured=False,
            device_vram_used=6 * GiB,
            device_vram_total=8 * GiB,
            init_profile=init,
            server=server,
            server_source="parent",
        )
        row.gpu = ProcGpu(dedicated=0, shared=0, attributed=False, inferred=5 * GiB, inferred_by="order")
        snap = _nv_snapshot([row], vram_used=6 * GiB, vram_total=8 * GiB)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "100"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = _plain_inspector_text(app)
                self.assertIn("Process VRAM", text)
                self.assertIn("N/A  [unattributed]", text)
                self.assertIn("Device VRAM", text)
                self.assertIn("6.00 GB / 8.00 GB  [NVML device-wide]", text)
                # Colored stacked bar must render even when per-process VRAM is unattributed on WSL2.
                self.assertIn("VRAM █", text)
                self.assertIn("profile / 8.00 GB GPU  (device 6.00 GB)", text)
                self.assertIn("Model Weights", text)
                self.assertIn("Activation Peak Budget", text)
                self.assertIn("KV Cache Used", text)
                self.assertIn("KV Cache Unused", text)
                self.assertIn("2.00 GB  [Init profile]", text)
                self.assertIn("3.00 GB  [Init profile]", text)
                self.assertIn("N/A  [requires reconciliation]", text)
                # Verbose '? order guess' boilerplate must appear AFTER vLLM Memory Breakdown, not push it off-screen.
                self.assertIn("? order guess:", text)
                self.assertLess(text.index("vLLM Memory Breakdown"), text.index("? order guess:"))

    async def test_non_vllm_llama_cpp_inspector_unchanged(self):
        detail = ProcDetail(pid=55, name="llama-server", runtimes=["CUDA", "ggml"])
        view = LlmView(
            segments=[
                Segment("weights", 2 * GiB, "weights"),
                Segment("KV cache (8,192 ctx, f16) est.", 512 * MiB, "kv"),
                Segment("compute buffers + runtime", 512 * MiB, "runtime"),
            ],
            server=ServerInfo(kind="llama-server", url="http://127.0.0.1:8080", n_ctx=8192),
            server_source="process",
        )
        row = ProcRow(
            pid=55,
            name="llama-server",
            kind="LLM",
            gpu=ProcGpu(dedicated=3 * GiB, shared=0),
            detail=detail,
            llm=view,
        )
        snap = _nv_snapshot([row], vram_used=3 * GiB, vram_total=8 * GiB)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "55"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = _plain_inspector_text(app)
                self.assertNotIn("vLLM Memory Breakdown", text)
                self.assertIn("weights 2.00 GB", text)
                self.assertIn("KV cache (8,192 ctx, f16) est. 512 MB", text)
                self.assertIn("server llama-server http://127.0.0.1:8080", text)


if __name__ == "__main__":
    unittest.main()
