from __future__ import annotations

import contextlib
import datetime
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil
from textual.widgets import Static

from gputaskman.app import GpuTaskManager
from gputaskman.collector import Collector
from gputaskman.config import AppConfig, VllmConfig, VllmEndpointConfig
from gputaskman.llm import (ServerInfo, ServerProber, TokenSample, VllmInitProfile, VllmMemoryBreakdown,
                            build_view, parse_vllm_init_log, probe_server, probe_vllm_url,
                            vllm_cache_config, vllm_metrics, vllm_models, vllm_rate_text,
                            vllm_token_counters, vllm_token_rates)
from gputaskman.pdh import AdapterGpu, ProcGpu
from gputaskman.procinfo import ProcDetail, ProcInspector, classify, vllm_launch_info
from gputaskman.report import to_dict, to_text

MiB = 1 << 20
LAUNCH = ["vllm", "serve", "org/model"]
# Title/layout fixtures follow vllm-project/vllm at 499a3d1e1e338e3e866c0719cfe1f429e2b0b762:
# utils/system_utils.py:set_process_title, v1/executor/multiproc_executor.py:setup_proc_title_and_log_prefix,
# entrypoints/cli/serve.py:cmd, and entrypoints/launchers/api_server/entry.py:run_server.

# Constructed (not captured from a live server) from vllm/v1/metrics/loggers.py PrometheusStatLogger:
# gauges and counters labelled model_name/engine; prometheus_client adds _total and _created to counters.
METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="org/model"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="org/model"} 0.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="org/model",reason="capacity"} 0.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="org/model"} 0.42
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{engine="0",model_name="org/model"} 1234.0
vllm:prompt_tokens_created{engine="0",model_name="org/model"} 1.7e+09
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.3",model_name="org/model"} 1.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="org/model"} 2.0
"""
NO_GAUGES = {"requests_running": None, "requests_waiting": None, "kv_cache_usage": None}


@contextlib.contextmanager
def listener(*, locked=False, redirect=False, metrics=None, model_id="org/model"):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path == "/metrics" and metrics is not None:
                body = (metrics() if callable(metrics) else metrics).encode()
            else:
                body = json.dumps({"data": [{"id": model_id, "root": "/not/a/model/file",
                                             "max_model_len": 8192, "size": 999999}]}).encode()
            status = 401 if locked and not self.headers.get("Authorization") else 302 if redirect else 200
            self.send_response(status)
            self.send_header("Content-Length", str(len(body) if status == 200 else 0))
            if redirect:
                self.send_header("Location", "/elsewhere")
            self.end_headers()
            if status == 200:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def collector(details, ports):
    c = Collector.__new__(Collector)
    c.adapter, c.adapters = None, []
    c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
    c.nvml = Mock(ok=False)
    c.counters = Mock(host_procs={})
    c.counters.sample.return_value = (
        {0: {10: ProcGpu(dedicated=512 * MiB, shared=0, engines={"compute": 10})}},
        {0: AdapterGpu(dedicated=512 * MiB, shared=0)})
    c.inspector = ProcInspector()
    c.inspector._cache = details
    c.inspector._ports = ports
    c.inspector._ports_time = float("inf")
    c.inspector.needs_scan = Mock(return_value=False)
    c.inspector.prune = Mock()
    c.prober = ServerProber()
    return c


def chain(details, parents):
    nodes = {pid: Mock(pid=pid) for pid in details}
    for pid, node in nodes.items():
        node.create_time.return_value = details[pid].create_time
        node.parent.return_value = nodes.get(parents.get(pid))
    return nodes


class VllmTests(unittest.TestCase):
    def test_vllm_topology_flags_are_value_aware(self):
        # vllm/engine/arg_utils.py aliases (-dp, -n, -r, -tp, -pp); vllm/utils/argparse_utils.py
        # FlexibleArgumentParser maps underscores to dashes and keeps argparse prefix matching.
        allowed = [
            [],
            ["--data-parallel-size", "1"], ["--data-parallel-size=1"], ["-dp", "1"], ["-dp=1"],
            ["--api-server-count", "1"], ["--api-server-count=1"],
            ["--distributed-executor-backend", "mp"], ["--distributed-executor-backend=mp"],
            ["--nnodes", "1"], ["--nnodes=1"], ["-n", "1"], ["-n=1"],
            ["--data_parallel_size", "1"], ["--api_server_count=1"], ["--distributed_executor_backend", "mp"],
            ["--tensor-parallel-size", "2"], ["-tp", "2"],
            ["--pipeline-parallel-size", "1"], ["--pipeline-parallel-size=1"], ["-pp", "1"], ["-pp=1"],
            ["--pipeline_parallel_size=1"],
            ["--data-parallel-size=1", "--api-server-count=1", "--distributed-executor-backend=mp"],
        ]
        rejected = [
            ["--data-parallel-size", "2"], ["--data-parallel-size=2"], ["-dp", "2"], ["-dp=2"],
            ["-dp2"], ["-dpn", "0"], ["-dpl", "2"],
            ["--data-parallel-rank", "0"], ["--data-parallel-size-local", "1"],
            ["--data-parallel-external-lb"],
            ["--api-server-count", "2"], ["--api-server-count=0"],
            ["--distributed-executor-backend", "ray"],
            ["--distributed-executor-backend=external_launcher"],
            ["--nnodes", "2"], ["--nnodes=2"], ["-n", "2"], ["-n2"],
            ["--node-rank", "0"], ["--node-rank=0"], ["-r", "0"], ["-r0"],
            ["--headless"], ["--config", "launch.yaml"], ["--config=launch.yaml"],
            ["--data_parallel_size", "2"], ["--api_server_count=2"], ["--distributed_executor_backend", "ray"],
            ["--node_rank", "0"], ["--data_parallel_rank=0"],
            ["--api-server", "2"], ["--nnode", "2"], ["--distributed-executor", "ray"], ["--head"],
            ["--conf", "launch.yaml"], ["--node-r", "0"],
            ["--pipeline-parallel-size", "2"], ["--pipeline-parallel-size=2"],
            ["-pp", "2"], ["-pp=2"], ["-pp2"],
            ["--pipeline-parallel", "1"], ["--pipeline-parallel=1"],
            ["--data-parallel-size"], ["-dp"], ["--api-server-count"],
            ["--distributed-executor-backend"], ["--nnodes"], ["--pipeline-parallel-size"],
            ["--data-parallel-size", "--api-server-count", "1"],
            ["--data-parallel-size=1", "--config", "launch.yaml"],
            ["-dp", "2", "--api-server-count", "1"],
        ]
        for args in allowed:
            with self.subTest(allowed=args):
                self.assertFalse(ProcDetail(10, "python", cmdline=LAUNCH + args).vllm_ambiguous)
        for args in rejected:
            with self.subTest(rejected=args):
                self.assertTrue(ProcDetail(10, "python", cmdline=LAUNCH + args).vllm_ambiguous)

    def test_signatures_are_hints_even_without_mapped_models_or_titles(self):
        cases = [
            ("vllm", LAUNCH, "server"),
            ("python3", ["/env/bin/python3", "/env/bin/vllm", "serve", "org/model"], "server"),
            ("python", ["python", "-u", "-m", "vllm.entrypoints.openai.api_server", "--model", "org/model"], "server"),
            ("VLLM::EngineCore", [], "worker"),
            ("VLLM::Worker", [], "worker"),
            ("VLLM::Worker_TP12", [], "worker"),
            ("python", ["VLLM::Worker_TP0"], "worker"),
            ("python", LAUNCH, "server"),  # inherited command line when setproctitle is absent
            ("vllm", ["vllm", "bench", "serve"], ""),
            ("python", ["python", "script.py", "-m", "vllm.entrypoints.openai.api_server"], ""),
            ("echo", ["echo", "-m", "vllm.entrypoints.openai.api_server"], ""),
            ("VLLM::Worker_TP0-extra", [], ""),
            ("VLLM::EngineCore_DP0", [], ""),
            ("VLLM::Worker_DP0_TP0", [], ""),
            ("VLLM::Worker_PP0_TP0", [], ""),
            ("VLLM::APIServer_0", [], ""),
            ("vllm-helper", [], ""),
        ]
        for name, cmd, role in cases:
            with self.subTest(name=name, cmd=cmd):
                d = ProcDetail(10, name, cmdline=cmd)
                self.assertEqual(d.vllm_role, role)
                self.assertEqual(classify(d, {}), "LLM" if role else "app")

    def test_models_are_full_identifiers_only(self):
        for bad in (None, [], {}, {"data": None}, {"data": {}}, {"data": [None]},
                    {"data": [{"id": 5}]}, {"data": [{"id": ""}]}, {"data": [{"id": "  "}]}):
            with self.subTest(response=bad):
                self.assertIsNone(vllm_models(bad))
        self.assertEqual(vllm_models({"data": []}), [])
        model = vllm_models({"data": [{"id": "org/model", "size": 100, "max_model_len": 200}]})[0]
        self.assertEqual(model["name"], "org/model")
        self.assertTrue(all(value is None for key, value in model.items() if key != "name"))

    def test_one_endpoint_cache_origin_and_credential_isolation(self):
        with listener(locked=True) as (port, seen):
            p = ServerProber()
            own = ("process", 1, 100)
            parent = ("parent", ((1, 100),))
            self.assertIsNotNone(p.get([port], "private-key", kind="vllm", origin=own))
            self.assertIsNotNone(p.get([port], "private-key", kind="vllm", origin=own))
            self.assertEqual(len(seen), 2)  # /v1/models, then a keyless /metrics
            self.assertIsNone(p.get([port], kind="vllm", origin=parent))
            self.assertIsNone(p.get([port], kind="vllm", origin=parent))
            self.assertEqual(p.locked_port([port], kind="vllm", origin=parent), port)
            self.assertIsNone(p.locked_port([port], "private-key", kind="vllm", origin=own))
            self.assertEqual(len(seen), 3)  # a 401 on /v1/models stops before /metrics
            self.assertIsNone(seen[-1][1])
            self.assertIsNotNone(p.get([port], "different-key", kind="vllm", origin=own))
            self.assertEqual(len(seen), 5)  # changed keys do not use the previous cache
            self.assertIsNotNone(p.get([port], "private-key", kind="vllm", origin=("process", 1, 101)))
            self.assertEqual(len(seen), 7)  # reused PID/port does not use old identity
            self.assertEqual({path for path, _ in seen}, {"/v1/models", "/metrics"})
            self.assertTrue(all(auth is None for path, auth in seen if path == "/metrics"))
            self.assertNotIn("private-key", repr(vars(p)))
            with patch("gputaskman.llm.time.time", return_value=time.time() + 31):
                self.assertIsNone(p.get([port], kind="vllm", origin=parent))
            self.assertEqual(len(seen), 8)
            self.assertEqual(p.locked_port([port], kind="vllm", origin=parent), port)

    def test_metrics_gauges_zero_stays_zero(self):
        self.assertEqual(vllm_metrics(METRICS, ["org/model"]),
                         {"requests_running": 3, "requests_waiting": 0, "kv_cache_usage": 0.42})

    def test_metrics_served_model_aliases(self):
        # vLLM labels metrics with the first --served-model-name; /v1/models lists every alias.
        text = METRICS.replace('model_name="org/model"', 'model_name="production-llm"')
        self.assertEqual(vllm_metrics(text, ["production-llm", "my-model"])["requests_running"], 3)
        self.assertEqual(vllm_metrics(text, ["my-model", "production-llm"])["requests_running"], 3)
        self.assertEqual(vllm_metrics(text, ["org/model"]), NO_GAUGES)

    def test_metrics_identity_must_be_one_engine_and_one_model(self):
        cases = {
            "second engine": METRICS + 'vllm:num_requests_running{engine="1",model_name="org/model"} 1\n',
            "second model": METRICS + 'vllm:num_requests_running{engine="0",model_name="other"} 1\n',
            "no model label": METRICS.replace(',model_name="org/model"} 3.0', "} 3.0"),
            "no engine label anywhere": METRICS.replace('engine="0",', ""),
            "one sample without engine": METRICS.replace('engine="0",model_name="org/model"} 3.0',
                                                         'model_name="org/model"} 3.0'),
            "empty engine label": METRICS.replace('engine="0"', 'engine=""'),
            "empty model label": METRICS.replace('model_name="org/model"', 'model_name=""'),
            "malformed labels": METRICS.replace('engine="0",model_name="org/model"} 3.0', 'engine=0} 3.0'),
            "no gauges": "# nothing here\nvllm:prompt_tokens_total{engine=\"0\",model_name=\"org/model\"} 1\n",
            "empty": "",
        }
        for reason, text in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(vllm_metrics(text, ["org/model", "other"]), NO_GAUGES)

    def test_metrics_invalid_samples_stay_unknown_individually(self):
        run = 'vllm:num_requests_running{engine="0",model_name="org/model"} 3.0'
        kv = 'vllm:kv_cache_usage_perc{engine="0",model_name="org/model"} 0.42'
        cases = [
            (METRICS.replace(run, run + "\n" + run), "requests_running"),  # duplicate sample
            (METRICS.replace(run, run.replace("3.0", "-1")), "requests_running"),
            (METRICS.replace(run, run.replace("3.0", "2.5")), "requests_running"),
            (METRICS.replace(run, run.replace("3.0", "abc")), "requests_running"),
            (METRICS.replace(run + "\n", ""), "requests_running"),  # missing
            (METRICS.replace(kv, kv.replace("0.42", "NaN")), "kv_cache_usage"),
            (METRICS.replace(kv, kv.replace("0.42", "+Inf")), "kv_cache_usage"),
            (METRICS.replace(kv, kv.replace("0.42", "42")), "kv_cache_usage"),  # a percent, not a fraction
            (METRICS.replace(kv, kv.replace("0.42", "-0.1")), "kv_cache_usage"),
        ]
        for text, key in cases:
            with self.subTest(text=text.splitlines()[2:3], key=key):
                gauges = vllm_metrics(text, ["org/model"])
                self.assertIsNone(gauges[key])
                self.assertEqual(gauges["requests_waiting"], 0)  # other gauges are unaffected

    def test_metrics_exposition_details(self):
        cases = [
            METRICS.replace('engine="0",model_name="org/model"}', 'engine="0",model_name="org/model",}'),
            METRICS.replace('"org/model"} 3.0', '"org/model"} 3.0 1700000000000'),  # timestamp
            METRICS.replace('engine="0",model_name="org/model"', 'model_name="org/model",engine="0"'),
        ]
        for text in cases:
            with self.subTest(text=text.splitlines()[2]):
                self.assertEqual(vllm_metrics(text, ["org/model"])["requests_running"], 3)
        quoted = METRICS.replace('model_name="org/model"', r'model_name="org/\"q\"\\x"')
        self.assertEqual(vllm_metrics(quoted, ['org/"q"\\x'])["kv_cache_usage"], 0.42)

    def test_metrics_probe_is_keyless_and_only_after_models(self):
        with listener(metrics=METRICS) as (port, seen):
            server = probe_server(port, key="own-key", kind="vllm")
        self.assertEqual(seen, [("/v1/models", "Bearer own-key"), ("/metrics", None)])
        self.assertEqual((server.requests_running, server.requests_waiting, server.kv_cache_usage), (3, 0, 0.42))
        with listener(locked=True, metrics=METRICS) as (port, seen):
            server = probe_server(port, key="own-key", kind="vllm")
        self.assertEqual(server.models[0]["name"], "org/model")  # a locked /metrics keeps the model IDs
        self.assertEqual((server.requests_running, server.requests_waiting, server.kv_cache_usage), (None,) * 3)
        with listener(metrics=METRICS) as (port, seen), patch("gputaskman.llm.vllm_models", return_value=None):
            self.assertIsNone(probe_server(port, kind="vllm"))
        self.assertEqual([path for path, _ in seen], ["/v1/models"])

    def test_metrics_reported_with_provenance_and_no_memory_change(self):
        with listener(metrics=METRICS) as (port, _):
            details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = self.sample(details, {20: [port]}, {10: 20})
        row = snap.procs[0]
        self.assertEqual(snap.attributed, 512 * MiB)
        self.assertEqual([(s.kind, s.bytes) for s in row.llm.segments], [("unknown", 512 * MiB)])
        server = to_dict(snap)["processes"][0]["llm"]["server"]
        self.assertEqual((server["requests_running"], server["requests_waiting"], server["kv_cache_usage"]),
                         (3, 0, 0.42))
        text = to_text(snap)
        self.assertIn("vLLM /metrics (runtime-reported): requests running 3, waiting 0, KV-cache pool 42.0%", text)
        self.assertIn("not of GPU memory", text)
        self.assertIn("server source: parent process", text)

    def test_unknown_gauges_are_null_and_not_displayed(self):
        with listener() as (port, _):  # /metrics answers JSON, not Prometheus text
            details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = self.sample(details, {20: [port]}, {10: 20})
        server = to_dict(snap)["processes"][0]["llm"]["server"]
        self.assertEqual({k: server[k] for k in NO_GAUGES}, NO_GAUGES)
        text = to_text(snap)
        self.assertNotIn("vLLM /metrics", text)
        self.assertNotIn("KV-cache pool", text)
        self.assertNotIn("token rates", text)
        self.assertNotIn("token_sample", json.dumps(to_dict(snap)))

    def test_token_counters_need_one_verified_identity(self):
        gen = 'vllm:generation_tokens_total{engine="0",model_name="org/model"} 4000.0\n'
        text = METRICS + gen
        self.assertEqual(vllm_token_counters(text, ["org/model"], 5.0),
                         TokenSample(5.0, ("0", "org/model"), 1234.0, 4000.0))
        rejected = {
            "second engine": text + gen.replace('engine="0"', 'engine="1"'),
            "no engine": text.replace('vllm:generation_tokens_total{engine="0",', "vllm:generation_tokens_total{"),
            "unadvertised model": text.replace('model_name="org/model"', 'model_name="other"'),
            "no counters": METRICS.replace("vllm:prompt_tokens_total", "vllm:prompt_tokens_other"),
        }
        for reason, bad in rejected.items():
            with self.subTest(reason=reason):
                self.assertIsNone(vllm_token_counters(bad, ["org/model"], 5.0))
        for bad_value in ("-1", "NaN", "abc"):
            with self.subTest(value=bad_value):
                sample = vllm_token_counters(text.replace("4000.0", bad_value), ["org/model"], 5.0)
                self.assertEqual((sample.prompt, sample.generation), (1234.0, None))
        sample = vllm_token_counters(text + gen, ["org/model"], 5.0)  # duplicated series
        self.assertEqual((sample.prompt, sample.generation), (1234.0, None))

    def test_token_rates_between_two_readings(self):
        identity = ("0", "org/model")
        before = TokenSample(100.0, identity, 10000.0, 4000.0)
        self.assertEqual(vllm_token_rates(before, TokenSample(103.0, identity, 10600.0, 4300.0)), (200.0, 100.0))
        self.assertEqual(vllm_token_rates(before, TokenSample(103.0, identity, 10000.0, 4000.0)), (0.0, 0.0))
        cases = {
            "first reading": (None, TokenSample(103.0, identity, 10600.0, 4300.0), (None, None)),
            "other engine": (before, TokenSample(103.0, ("1", "org/model"), 10600.0, 4300.0), (None, None)),
            "other model": (before, TokenSample(103.0, ("0", "other"), 10600.0, 4300.0), (None, None)),
            "no time passed": (before, TokenSample(100.0, identity, 10600.0, 4300.0), (None, None)),
            "clock backwards": (before, TokenSample(99.0, identity, 10600.0, 4300.0), (None, None)),
            "stale baseline": (before, TokenSample(160.5, identity, 10600.0, 4300.0), (None, None)),
            "prompt reset": (before, TokenSample(103.0, identity, 50.0, 4300.0), (None, 100.0)),
            "missing generation": (before, TokenSample(103.0, identity, 10600.0, None), (200.0, None)),
            "missing baseline": (TokenSample(100.0, identity, None, 4000.0),
                                 TokenSample(103.0, identity, 10600.0, 4300.0), (None, 100.0)),
        }
        for reason, (a, b, expected) in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(vllm_token_rates(a, b), expected)

    def test_token_rates_across_cached_probes(self):
        counts = {"prompt": 10000, "generation": 4000}
        clock = [100.0]

        def metrics():
            return (METRICS.replace("vllm:prompt_tokens_total{engine=\"0\",model_name=\"org/model\"} 1234.0",
                                    f'vllm:prompt_tokens_total{{engine="0",model_name="org/model"}} {counts["prompt"]}')
                    + f'vllm:generation_tokens_total{{engine="0",model_name="org/model"}} {counts["generation"]}\n')

        def probe(origin=("process", 1, 100), step=3.0, **new_counts):
            clock[0] += step
            counts.update(new_counts)
            return p.get([port], kind="vllm", origin=origin)

        with listener(metrics=metrics) as (port, seen), \
                patch("gputaskman.llm.time.monotonic", side_effect=lambda: clock[0]), \
                patch("gputaskman.llm.time.time", side_effect=lambda: clock[0]):
            p = ServerProber()
            first = probe(step=0.0)
            self.assertEqual((first.prompt_tok_s, first.gen_tok_s), (None, None))  # one reading: --once stays null
            self.assertIsNone(vllm_rate_text(first))
            self.assertIs(probe(step=1.0), first)  # cached: no new reading, no rate from refreshes
            busy = probe(step=2.0, prompt=10600, generation=4300)
            self.assertEqual((busy.prompt_tok_s, busy.gen_tok_s), (200.0, 100.0))
            idle = probe()
            self.assertEqual((idle.prompt_tok_s, idle.gen_tok_s), (0.0, 0.0))
            self.assertEqual(vllm_rate_text(idle), "vLLM token rates (derived from /metrics counters): "
                                                   "prompt 0.0 tok/s, generation 0.0 tok/s")
            reset = probe(prompt=50, generation=20)
            self.assertEqual((reset.prompt_tok_s, reset.gen_tok_s), (None, None))
            after = probe(prompt=350, generation=170)
            self.assertEqual((after.prompt_tok_s, after.gen_tok_s), (100.0, 50.0))
            restarted = probe(origin=("process", 1, 101), prompt=650, generation=320)
            self.assertEqual((restarted.prompt_tok_s, restarted.gen_tok_s), (None, None))
            stale = probe(step=61.0, prompt=950, generation=470)
            self.assertEqual((stale.prompt_tok_s, stale.gen_tok_s), (None, None))
        self.assertEqual(sum(path == "/metrics" for path, _ in seen), 7)

    def test_token_rates_keep_provenance_and_stay_out_of_json(self):
        server = ServerInfo("vllm", "http://127.0.0.1:8000", models=[{"name": "org/model"}],
                            prompt_tok_s=200.0, gen_tok_s=None,
                            token_sample=TokenSample(1.0, ("0", "org/model"), 10.0, None))
        self.assertEqual(vllm_rate_text(server), "vLLM token rates (derived from /metrics counters): "
                                                 "prompt 200.0 tok/s, generation n/a tok/s")
        view = build_view(ProcDetail(10, "VLLM::EngineCore"), 512 * MiB, 0, server)
        notes = " ".join(view.notes)
        self.assertIn("derived from two /metrics counter readings", notes)
        self.assertIn("neither submitted nor computed prefill tokens", notes)
        self.assertEqual([(s.kind, s.bytes) for s in view.segments], [("unknown", 512 * MiB)])
        from gputaskman.report import _server_dict
        data = _server_dict(server)
        self.assertNotIn("token_sample", data)
        self.assertEqual((data["prompt_tok_s"], data["gen_tok_s"]), (200.0, None))
        json.dumps(data)
        self.assertIsNone(vllm_rate_text(ServerInfo("llama-server", "u", gen_tok_s=5.0)))

    def test_unauthenticated_parent_redirect_is_not_followed(self):
        with listener(redirect=True) as (port, seen):
            self.assertIsNone(probe_server(port, kind="vllm"))
            self.assertEqual(seen, [("/v1/models", None)])

    def sample(self, details, ports, parents, *, nodes=None, prober=None):
        c = collector(details, ports)
        if prober is not None:
            c.prober = prober
        nodes = nodes or chain(details, parents)
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes[pid]), \
                patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details[pid].create_time):
            return c, c.sample()

    def test_collector_two_hops_no_files_no_invented_memory(self):
        with listener() as (port, seen):
            details = {10: ProcDetail(10, "VLLM::Worker_TP0", create_time=3),
                       20: ProcDetail(20, "VLLM::EngineCore", create_time=2),
                       30: ProcDetail(30, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = self.sample(details, {30: [port]}, {10: 20, 20: 30})
        row = snap.procs[0]
        self.assertEqual(row.kind, "LLM")
        self.assertEqual(row.llm.server_source, "parent")
        self.assertEqual(row.llm.server.models[0]["name"], "org/model")
        self.assertIn("reported (parent API): org/model", row.inside)
        self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)])
        self.assertEqual(row.gpu.dedicated, 512 * MiB)
        self.assertEqual(snap.attributed, 512 * MiB)
        self.assertEqual([(s.kind, s.bytes) for s in row.llm.segments], [("unknown", 512 * MiB)])
        self.assertEqual(row.llm.models, [])
        self.assertEqual(row.llm.files, [])
        self.assertIsNone(row.llm.kv_per_token)
        self.assertIsNone(row.llm.n_ctx)
        data = to_dict(snap)["processes"][0]["llm"]
        self.assertEqual(data["server_source"], "parent")
        self.assertEqual(data["server"]["models"][0]["name"], "org/model")
        text = to_text(snap)
        self.assertIn("parent process", text)
        self.assertIn("API-reported model ID: org/model", text)
        self.assertIn("do not prove this process's model residency", text)

    def test_upstream_default_single_gpu_and_tp_layouts(self):
        # Default CLI calls run_server in the launcher process (no set_process_title there).
        # APIServer_<index> titles belong to APIServerProcessManager's multi-API worker path.
        with listener() as (port, seen):
            for name in ("VLLM::EngineCore", "VLLM::Worker", "VLLM::Worker_TP0", "VLLM::Worker_TP1"):
                with self.subTest(name=name):
                    details = {10: ProcDetail(10, name, create_time=3),
                               20: ProcDetail(20, "VLLM::EngineCore", create_time=2),
                               30: ProcDetail(30, "python", cmdline=LAUNCH, create_time=1)}
                    parents = {10: 30} if name == "VLLM::EngineCore" else {10: 20, 20: 30}
                    _, snap = self.sample(details, {30: [port]}, parents)
                    self.assertEqual(snap.procs[0].llm.server_source, "parent")
                    self.assertEqual(snap.procs[0].llm.server.models[0]["name"], "org/model")
            self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)] * 4)

    def test_unclear_ancestry_or_listeners_never_probe(self):
        for reason in ("two servers", "shared port", "two ports", "unrelated parent",
                       "newer parent", "reused parent", "reused worker", "unknown start", "cycle",
                       "distributed parent", "distributed worker", "protected parent", "no parent",
                       "own and parent", "worker listener"):
            with self.subTest(reason=reason):
                details = {10: ProcDetail(10, "VLLM::Worker_TP0", create_time=3),
                           20: ProcDetail(20, "python", cmdline=list(LAUNCH), create_time=2),
                           30: ProcDetail(30, "python", cmdline=list(LAUNCH), create_time=1)}
                ports, parents = {20: [8000]}, {10: 20}
                if reason == "two servers":
                    ports[30], parents[20] = [8001], 30
                elif reason == "shared port":
                    ports[30] = [8000]
                elif reason == "two ports":
                    ports[20] = [8000, 8001]
                elif reason == "unrelated parent":
                    details[20].cmdline = ["python", "unrelated.py"]
                elif reason == "newer parent":
                    details[20].create_time = 4
                elif reason == "unknown start":
                    details[20].create_time = 0
                elif reason == "cycle":
                    parents[20] = 10
                elif reason.startswith("distributed"):
                    details[20 if reason.endswith("parent") else 10].cmdline += ["--data-parallel-size", "2"]
                elif reason == "no parent":
                    parents = {}
                elif reason == "own and parent":
                    ports[10] = [8001]
                elif reason == "worker listener":
                    details[20].name, details[20].cmdline = "VLLM::EngineCore", []
                nodes = chain(details, parents)
                if reason in ("reused parent", "reused worker"):
                    nodes[20 if reason.endswith("parent") else 10].create_time.return_value = 99
                if reason == "protected parent":
                    nodes[10].parent.side_effect = psutil.AccessDenied(10)
                with patch("gputaskman.llm.probe_server") as probe:
                    _, snap = self.sample(details, ports, parents, nodes=nodes)
                probe.assert_not_called()
                self.assertIsNone(snap.procs[0].llm.server)

    def test_parent_401_does_not_receive_worker_key_or_use_authenticated_cache(self):
        with listener(locked=True) as (port, seen):
            details = {10: ProcDetail(10, "VLLM::EngineCore",
                                      cmdline=LAUNCH + ["--api-key", "private-worker-key"], create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH + ["--api-key", "private-worker-key"],
                                      create_time=1)}
            prober = ServerProber()
            self.assertIsNotNone(prober.get([port], "private-worker-key", kind="vllm", origin=("process", 20, 1)))
            c, snap = self.sample(details, {20: [port]}, {10: 20}, prober=prober)
        self.assertIsNone(snap.procs[0].llm.server)
        self.assertIn("Parent-process API locked (401)", " ".join(snap.procs[0].llm.notes))
        self.assertEqual([path for path, _ in seen], ["/v1/models", "/metrics", "/v1/models"])
        self.assertIsNotNone(seen[0][1])
        self.assertIsNone(seen[1][1])  # /metrics never receives a key, even on the own port
        self.assertIsNone(seen[2][1])
        self.assertNotIn("private-worker-key", json.dumps(to_dict(snap)))
        self.assertNotIn("private-worker-key", to_text(snap))
        self.assertNotIn("private-worker-key", repr(vars(c.prober)))

    def test_own_listener_keeps_own_provenance(self):
        with listener() as (port, seen):
            details = {10: ProcDetail(10, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = self.sample(details, {10: [port]}, {})
        self.assertEqual(snap.procs[0].llm.server_source, "process")
        self.assertIn("reported (API): org/model", snap.procs[0].inside)
        self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)])

    def test_api_unavailable_does_not_change_measured_memory(self):
        for response in (None, {"error": "disabled"}, {"data": [None]}):
            with self.subTest(response=response):
                details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                           20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
                with patch("gputaskman.llm._get", return_value=response):
                    _, snap = self.sample(details, {20: [8000]}, {10: 20})
                self.assertIsNone(snap.procs[0].llm.server)
                self.assertEqual(snap.attributed, 512 * MiB)

    def test_real_child_ancestry_and_http_through_collector(self):
        """Real cheap processes/listener, not a vLLM runtime or a synthetic ancestry lookup."""
        with listener() as (port, seen):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                c = collector({}, {})
                c.inspector = ProcInspector()
                original_new = c.inspector._new

                def details(pid):
                    d = original_new(pid)
                    if pid == os.getpid():
                        d.cmdline = LAUNCH
                    elif pid == child.pid:
                        d.cmdline = LAUNCH  # no setproctitle; inherited launch signature
                    return d

                c.inspector._new = details
                c.inspector.needs_scan = Mock(return_value=False)
                c.counters.sample.return_value = (
                    {0: {child.pid: ProcGpu(dedicated=512 * MiB, shared=0)}},
                    {0: AdapterGpu(dedicated=512 * MiB, shared=0)})
                connections = [SimpleNamespace(status=psutil.CONN_LISTEN, pid=os.getpid(),
                                               laddr=SimpleNamespace(port=port))]
                with patch("gputaskman.collector.BACKEND", "nvml"), \
                        patch("gputaskman.procinfo.psutil.net_connections", return_value=connections):
                    snap = c.sample()
                    again = c.sample()
                self.assertEqual(snap.procs[0].llm.server_source, "parent")
                self.assertEqual(again.procs[0].llm.server.models[0]["name"], "org/model")
                self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)])
            finally:
                child.terminate()
                child.wait(timeout=5)


class VllmUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_report_in_table_inspector_and_json(self):
        with listener() as (port, _):
            details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = VllmTests().sample(details, {20: [port]}, {10: 20})
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "10"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = "\n".join(getattr(line, "plain", "") for line in
                                 app.query_one("#inspector", Static).content.renderables)
                self.assertIn("parent process; API report, not worker residency", text)
                self.assertIn("API-reported model ID: org/model", text)
                self.assertNotIn("vLLM /metrics", text)

    async def test_metrics_gauges_in_inspector(self):
        with listener(metrics=METRICS) as (port, _):
            details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = VllmTests().sample(details, {20: [port]}, {10: 20})
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "10"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = "\n".join(getattr(line, "plain", "") for line in
                                 app.query_one("#inspector", Static).content.renderables)
                self.assertIn("vLLM /metrics (runtime-reported): requests running 3, waiting 0, "
                              "KV-cache pool 42.0%", text)


GiB = 1 << 30
CACHE_CONFIG_METRIC = (
    'vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",'
    'engine="0",gpu_memory_utilization="0.9",kv_cache_memory_bytes="None",'
    'kv_cache_size_tokens="32768",num_cpu_blocks="128",num_gpu_blocks="2048"} 1.0\n'
)


class VllmStrategyBTests(unittest.TestCase):
    def test_vllm_launch_info_extracts_safe_fields_only(self):
        info = vllm_launch_info([
            "vllm", "serve", "Qwen/Qwen2.5-0.5B", "--port", "8000",
            "--served-model-name", "qwen-prod", "qwen-alias",
            "--log-file", "/var/log/vllm.log", "--api-key", "secret-val",
        ])
        self.assertEqual(info["port"], 8000)
        self.assertEqual(info["model"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(info["served_model_names"], ["qwen-prod", "qwen-alias"])
        self.assertEqual(info["log_file"], "/var/log/vllm.log")
        self.assertNotIn("secret-val", repr(info))

        api_info = vllm_launch_info([
            "python3", "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model=org/model", "--port=8001", "--log_file=/tmp/vllm.log",
            "--served_model_name=my-model",
        ])
        self.assertEqual(api_info["port"], 8001)
        self.assertEqual(api_info["model"], "org/model")
        self.assertEqual(api_info["served_model_names"], ["my-model"])
        self.assertEqual(api_info["log_file"], "/tmp/vllm.log")

        py_serve = vllm_launch_info(["/env/bin/python", "/env/bin/vllm", "serve", "org/m2", "-p", "8002"])
        self.assertEqual(py_serve["model"], "org/m2")
        self.assertEqual(py_serve["port"], 8002)

    def test_parse_vllm_init_log_v1_info_and_debug(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "vllm.log"
            now_iso = "2026-11-24T12:00:10+00:00"
            log.write_text("\n".join([
                f"{now_iso} INFO vLLM API server version 0.8.5",
                f"{now_iso} INFO Initializing a V1 LLM engine (v0.8.5) with config: ...",
                f"{now_iso} INFO [model_runner.py:462] Model loading took 1.50 GiB memory and 2.345678 seconds",
                f"{now_iso} DEBUG [mem_utils.py:233] Memory profiling takes 1.12 seconds. "
                "Total non KV cache memory: 2.10GiB; torch peak memory increase: 0.40GiB; "
                "non-torch forward increase memory: 0.10 GiB; total consumed (from mem_get_info): 2.20GiB; "
                "weights memory: 1.50GiB.",
                f"{now_iso} INFO [gpu_worker.py:863] Available KV cache memory: 3.00 GiB",
                f"{now_iso} INFO [kv_cache_utils.py:1124] GPU KV cache size: 49,152 tokens, "
                "Maximum concurrency for 4,096 tokens per request: 12.00x",
                f"{now_iso} INFO [model_runner.py:1248] Graph capturing finished in 4 secs, took 0.25 GiB",
                f"{now_iso} INFO [entry.py:139] Starting vLLM server on http://0.0.0.0:8000",
            ]), encoding="utf-8")
            profile = parse_vllm_init_log(str(log), process_start=1795435200.0)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.weights_bytes, round(1.50 * GiB))
        self.assertEqual(profile.kv_allocated_bytes, round(3.00 * GiB))
        self.assertEqual(profile.kv_cache_size_tokens, 49152)
        self.assertEqual(profile.torch_peak_increase_bytes, round(0.40 * GiB))
        self.assertEqual(profile.non_torch_increase_bytes, round(0.10 * GiB))
        self.assertEqual(profile.activation_peak_bytes, round(0.40 * GiB) + round(0.10 * GiB))
        self.assertEqual(profile.cuda_graph_bytes, round(0.25 * GiB))
        self.assertFalse(profile.stale)

    def test_parse_vllm_init_log_v0_info_and_json_lines(self):
        with tempfile.TemporaryDirectory() as td:
            v0_log = Path(td) / "v0.log"
            v0_log.write_text("\n".join([
                "2026-11-24 12:00:05,123 INFO Loading model weights took 2.00 GB",
                "2026-11-24 12:00:08,000 INFO model weights take 2.00GiB; non_torch_memory takes 0.15GiB; "
                "PyTorch activation peak memory takes 0.85GiB; the rest of the memory reserved for KV Cache is 4.00GiB.",
                "2026-11-24 12:00:09,000 INFO # GPU blocks: 1024, # CPU blocks: 256",
            ]), encoding="utf-8")
            p0 = parse_vllm_init_log(str(v0_log))
            self.assertIsNotNone(p0)
            self.assertEqual(p0.weights_bytes, 2 * GiB)
            self.assertEqual(p0.kv_allocated_bytes, 4 * GiB)
            self.assertEqual(p0.activation_peak_bytes, round(0.85 * GiB) + round(0.15 * GiB))
            self.assertEqual(p0.num_gpu_blocks, 1024)

            json_log = Path(td) / "json.log"
            json_log.write_text("\n".join([
                json.dumps({"asctime": "2026-11-24T12:00:01Z",
                            "message": "Model loading took 1.25 GiB memory and 1.500000 seconds"}),
                json.dumps({"timestamp": 1795435210.0,
                            "msg": "Available KV cache memory: 2.50 GiB"}),
                json.dumps({"ts": 1795435212000,
                            "message": "GPU KV cache size: 16,384 tokens, Maximum concurrency for 2,048 tokens per request: 8.00x"}),
            ]), encoding="utf-8")
            pj = parse_vllm_init_log(str(json_log), process_start=1795435200.0)
            self.assertIsNotNone(pj)
            self.assertEqual(pj.weights_bytes, round(1.25 * GiB))
            self.assertEqual(pj.kv_allocated_bytes, round(2.50 * GiB))
            self.assertEqual(pj.kv_cache_size_tokens, 16384)
            self.assertFalse(pj.stale)

    def test_parse_vllm_init_log_restart_takes_last_block_and_stale_detection(self):
        t_run2 = datetime.datetime(2026, 11, 24, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "restart.log"
            log.write_text("\n".join([
                "2026-11-24T10:00:00Z INFO Initializing a V1 LLM engine (v0.8.5)",
                "2026-11-24T10:00:02Z INFO Model loading took 4.00 GiB memory and 3.000000 seconds",
                "2026-11-24T10:00:04Z DEBUG Memory profiling takes 1.00 seconds. Total non KV cache memory: 4.50GiB; "
                "torch peak memory increase: 0.75GiB; total consumed (from mem_get_info): 4.60GiB; weights memory: 4.00GiB.",
                "2026-11-24T10:00:05Z INFO Available KV cache memory: 2.00 GiB",
                "2026-11-24T10:00:06Z INFO Starting vLLM server on http://0.0.0.0:8000",
                # Second run without DEBUG profiling
                "2026-11-24T12:00:00Z INFO Initializing a V1 LLM engine (v0.8.5)",
                "2026-11-24T12:00:02Z INFO Model loading took 1.00 GiB memory and 1.000000 seconds",
                "2026-11-24T12:00:04Z INFO Available KV cache memory: 5.00 GiB",
                "2026-11-24T12:00:05Z INFO Starting vLLM server on http://0.0.0.0:8000",
            ]), encoding="utf-8")
            fresh = parse_vllm_init_log(str(log), process_start=t_run2 - 10)
            self.assertIsNotNone(fresh)
            self.assertEqual(fresh.weights_bytes, 1 * GiB)
            self.assertEqual(fresh.kv_allocated_bytes, 5 * GiB)
            self.assertIsNone(fresh.activation_peak_bytes)  # no contamination from run 1 DEBUG peak
            self.assertFalse(fresh.stale)

            stale = parse_vllm_init_log(str(log), process_start=t_run2 + 3600)
            self.assertIsNotNone(stale)
            self.assertTrue(stale.stale)
            self.assertIn("timestamp", stale.stale_reason)

            # Bannerless restart via duplicate Model loading took also clears old DEBUG peak
            dup_log = Path(td) / "dup.log"
            dup_log.write_text("\n".join([
                "2026-11-24T10:00:02Z INFO Model loading took 3.00 GiB memory and 2.000000 seconds",
                "2026-11-24T10:00:03Z DEBUG Memory profiling takes 1.00 seconds. Total non KV cache memory: 3.50GiB; "
                "torch peak memory increase: 0.50GiB; total consumed (from mem_get_info): 3.60GiB; weights memory: 3.00GiB.",
                "2026-11-24T12:00:02Z INFO Model loading took 1.25 GiB memory and 1.000000 seconds",
                "2026-11-24T12:00:03Z INFO Available KV cache memory: 4.50 GiB",
            ]), encoding="utf-8")
            dup_prof = parse_vllm_init_log(str(dup_log), process_start=t_run2 - 10)
            self.assertEqual(dup_prof.weights_bytes, round(1.25 * GiB))
            self.assertEqual(dup_prof.kv_allocated_bytes, round(4.50 * GiB))
            self.assertIsNone(dup_prof.activation_peak_bytes)

            # Short timestamp without year falls back to file mtime, and mtime cache invalidates on update
            short_log = Path(td) / "short.log"
            short_log.write_text("INFO 11-24 12:00:00 [model_runner.py:462] Model loading took 1.00 GiB memory and 1.000000 seconds\n",
                                 encoding="utf-8")
            os.utime(short_log, (1700000000.0, 1700000000.0))
            short_stale = parse_vllm_init_log(str(short_log), process_start=1700000100.0)
            self.assertIsNotNone(short_stale)
            self.assertTrue(short_stale.stale)
            self.assertIn("mtime", short_stale.stale_reason)

            short_log.write_text("INFO 11-24 12:00:00 [model_runner.py:462] Model loading took 2.00 GiB memory and 1.000000 seconds\n",
                                 encoding="utf-8")
            os.utime(short_log, (1700000110.0, 1700000110.0))
            short_fresh = parse_vllm_init_log(str(short_log), process_start=1700000100.0)
            self.assertEqual(short_fresh.weights_bytes, 2 * GiB)
            self.assertFalse(short_fresh.stale)

    def test_parse_vllm_init_log_naive_iso_timestamp_falls_back_to_mtime_regardless_of_tz(self):
        utc_ts = datetime.datetime(2026, 11, 24, 3, 30, 0, tzinfo=datetime.timezone.utc).timestamp()
        for tz_name in ("UTC-9", "UTC+5", "UTC"):
            with self.subTest(tz=tz_name):
                try:
                    with patch.dict(os.environ, {"TZ": tz_name}):
                        if hasattr(time, "tzset"):
                            time.tzset()
                        with tempfile.TemporaryDirectory() as td:
                            log = Path(td) / "docker_utc.log"
                            log.write_text("\n".join([
                                "2026-11-24 03:30:05,000 INFO Model loading took 1.50 GiB memory and 1.200000 seconds",
                                "2026-11-24 03:30:08,000 INFO Available KV cache memory: 3.00 GiB",
                            ]), encoding="utf-8")
                            # Fresh mtime (UTC epoch 8s after process_start) must not be marked stale even on UTC+9 hosts.
                            os.utime(log, (utc_ts + 8.0, utc_ts + 8.0))
                            with patch("gputaskman.llm._datetime_epoch",
                                       side_effect=AssertionError("naive timestamp must not use local epoch")):
                                fresh = parse_vllm_init_log(str(log), process_start=utc_ts)
                            self.assertIsNotNone(fresh)
                            self.assertEqual(fresh.timestamp, "2026-11-24 03:30:08,000")
                            self.assertIsNone(fresh.timestamp_epoch)
                            self.assertFalse(fresh.stale)
                            self.assertIsNone(fresh.stale_reason)

                            # Stale mtime (UTC epoch 600s before process_start) must be marked stale even on UTC-5 hosts.
                            os.utime(log, (utc_ts - 600.0, utc_ts - 600.0))
                            stale = parse_vllm_init_log(str(log), process_start=utc_ts)
                            self.assertIsNotNone(stale)
                            self.assertIsNone(stale.timestamp_epoch)
                            self.assertTrue(stale.stale)
                            self.assertIn("mtime", stale.stale_reason)

                            # JSON line with naive ISO asctime also falls back to file_mtime.
                            jlog = Path(td) / "docker_utc_json.log"
                            jlog.write_text(json.dumps({
                                "asctime": "2026-11-24 03:30:08,000",
                                "message": "Available KV cache memory: 2.00 GiB",
                            }) + "\n", encoding="utf-8")
                            os.utime(jlog, (utc_ts + 8.0, utc_ts + 8.0))
                            jfresh = parse_vllm_init_log(str(jlog), process_start=utc_ts)
                            self.assertIsNotNone(jfresh)
                            self.assertEqual(jfresh.timestamp, "2026-11-24 03:30:08,000")
                            self.assertIsNone(jfresh.timestamp_epoch)
                            self.assertFalse(jfresh.stale)
                finally:
                    if hasattr(time, "tzset"):
                        time.tzset()

    def test_vllm_cache_config_engine_only_labels(self):
        cfg = vllm_cache_config(METRICS + CACHE_CONFIG_METRIC)
        self.assertEqual(cfg, {
            "engine": "0",
            "block_size": 16,
            "num_gpu_blocks": 2048,
            "num_cpu_blocks": 128,
            "gpu_memory_utilization": 0.9,
            "cache_dtype": "auto",
            "enable_prefix_caching": True,
            "kv_cache_memory_bytes": None,
            "kv_cache_size_tokens": 32768,
        })
        # Explicit kv_cache_memory_bytes
        explicit = CACHE_CONFIG_METRIC.replace('kv_cache_memory_bytes="None"', 'kv_cache_memory_bytes="2147483648"')
        self.assertEqual(vllm_cache_config(explicit)["kv_cache_memory_bytes"], 2 * GiB)
        # Missing or multi-engine samples return None
        self.assertIsNone(vllm_cache_config(METRICS))
        self.assertIsNone(vllm_cache_config(CACHE_CONFIG_METRIC + CACHE_CONFIG_METRIC.replace('engine="0"', 'engine="1"')))

        with listener(metrics=METRICS + CACHE_CONFIG_METRIC) as (port, _):
            srv = probe_vllm_url(f"http://127.0.0.1:{port}")
        self.assertIsNotNone(srv)
        self.assertEqual(srv.cache_config["num_gpu_blocks"], 2048)
        self.assertIsNotNone(srv.probed_at)
        from gputaskman.report import _server_dict
        self.assertNotIn("probed_at", _server_dict(srv))

    def test_vllm_breakdown_provenance_segments_and_invariants(self):
        server = ServerInfo(
            "vllm", "http://127.0.0.1:8000", models=[{"name": "org/model"}],
            requests_running=2, requests_waiting=1, kv_cache_usage=0.25,
            cache_config={"engine": "0", "block_size": 16, "num_gpu_blocks": 1024,
                          "cache_dtype": "auto", "enable_prefix_caching": True,
                          "gpu_memory_utilization": 0.9, "kv_cache_memory_bytes": None,
                          "kv_cache_size_tokens": 16384},
            probed_at=1795435220.0,
        )
        profile = VllmInitProfile(
            path="/tmp/vllm.log",
            weights_bytes=2 * GiB,
            kv_allocated_bytes=4 * GiB,
            kv_cache_size_tokens=16384,
            torch_peak_increase_bytes=512 * MiB,
            non_torch_increase_bytes=128 * MiB,
            activation_peak_bytes=640 * MiB,
            cuda_graph_bytes=256 * MiB,
            timestamp="2026-11-24T12:00:10Z",
            timestamp_epoch=1795435210.0,
            stale=False,
        )
        view = build_view(
            ProcDetail(10, "VLLM::EngineCore", create_time=1795435200.0),
            7 * GiB, 0, server, attributed=True, server_source="parent",
            vllm_init_profile=profile, gpu_id="gpu:0",
            device_vram_used=7 * GiB + 256 * MiB, device_vram_total=8 * GiB,
            vram_measured=True, sample_time=1795435225.0,
        )
        bd = view.vllm_breakdown
        self.assertIsInstance(bd, VllmMemoryBreakdown)
        self.assertEqual(bd.weights_bytes, 2 * GiB)
        self.assertEqual(bd.kv_allocated_bytes, 4 * GiB)
        self.assertEqual(bd.other_runtime_bytes, 1 * GiB)
        self.assertEqual(bd.activation_peak_bytes, 640 * MiB)
        self.assertIsNone(bd.activation_current_bytes)
        self.assertIsNone(bd.kv_active_request_bytes)
        self.assertEqual(bd.kv_pool_occupied_bytes, 1 * GiB)
        self.assertEqual(bd.kv_pool_available_bytes, 3 * GiB)
        self.assertEqual(bd.measurement_confidence["total_vram_used_bytes"], "measured")
        self.assertEqual(bd.measurement_confidence["weights_bytes"], "init_profile")
        self.assertEqual(bd.measurement_confidence["kv_allocated_bytes"], "init_profile")
        self.assertEqual(bd.measurement_confidence["other_runtime_bytes"], "derived")
        self.assertEqual(bd.measurement_confidence["kv_pool_occupied_bytes"], "derived")
        self.assertEqual(bd.measurement_confidence["activation_current_bytes"], "unavailable")
        self.assertEqual(bd.measurement_confidence["kv_active_request_bytes"], "unavailable")
        self.assertEqual(
            [(s.kind, s.bytes) for s in view.segments],
            [("weights", 2 * GiB), ("kv", 4 * GiB), ("runtime", 1 * GiB)],
        )
        self.assertEqual(sum(s.bytes for s in view.segments), 7 * GiB)

        # Never estimate kv_allocated from gpu_memory_utilization * total_vram
        view_no_log = build_view(
            ProcDetail(10, "VLLM::EngineCore"), 7 * GiB, 0, server,
            gpu_id="gpu:0", device_vram_total=8 * GiB, vram_measured=True,
        )
        self.assertIsNone(view_no_log.vllm_breakdown.kv_allocated_bytes)
        self.assertIsNone(view_no_log.vllm_breakdown.kv_pool_occupied_bytes)
        self.assertEqual([(s.kind, s.bytes) for s in view_no_log.segments], [("unknown", 7 * GiB)])

        # Explicit cache_config.kv_cache_memory_bytes is accepted when init log lacks KV
        server_explicit_kv = ServerInfo(
            "vllm", "http://127.0.0.1:8000", models=[{"name": "org/model"}],
            kv_cache_usage=0.5,
            cache_config={"engine": "0", "kv_cache_memory_bytes": 2 * GiB},
            probed_at=1795521620.0,
        )
        view_explicit = build_view(
            ProcDetail(10, "VLLM::EngineCore"), 5 * GiB, 0, server_explicit_kv, vram_measured=True,
        )
        self.assertEqual(view_explicit.vllm_breakdown.kv_allocated_bytes, 2 * GiB)
        self.assertEqual(view_explicit.vllm_breakdown.kv_pool_occupied_bytes, 1 * GiB)
        self.assertEqual(view_explicit.vllm_breakdown.kv_pool_available_bytes, 1 * GiB)
        self.assertEqual([(s.kind, s.bytes) for s in view_explicit.segments], [("kv", 2 * GiB), ("unknown", 3 * GiB)])

        # Reconciliation mismatch (weights + kv > total)
        view_mismatch = build_view(
            ProcDetail(10, "VLLM::EngineCore"), 5 * GiB, 0, server,
            vllm_init_profile=profile, vram_measured=True,
        )
        self.assertIsNone(view_mismatch.vllm_breakdown.other_runtime_bytes)
        self.assertTrue(any("Reconciliation mismatch" in n for n in view_mismatch.vllm_breakdown.notes))
        self.assertEqual([(s.kind, s.bytes) for s in view_mismatch.segments], [("unknown", 5 * GiB)])

        # Stale profile excluded from segments & derived totals
        stale_profile = VllmInitProfile(
            path="/tmp/old.log", weights_bytes=2 * GiB, kv_allocated_bytes=4 * GiB,
            stale=True, stale_reason="init log timestamp predates process start",
        )
        view_stale = build_view(
            ProcDetail(10, "VLLM::EngineCore"), 7 * GiB, 0, server,
            vllm_init_profile=stale_profile, vram_measured=True,
        )
        self.assertTrue(view_stale.vllm_breakdown.stale)
        self.assertIsNone(view_stale.vllm_breakdown.other_runtime_bytes)
        self.assertIsNone(view_stale.vllm_breakdown.kv_pool_occupied_bytes)
        self.assertEqual([(s.kind, s.bytes) for s in view_stale.segments], [("unknown", 7 * GiB)])

        # Unproven container / WSL: total_vram_used_bytes stays None and no segments budgeted
        view_unproven = build_view(
            ProcDetail(10, "VLLM::EngineCore"), 0, 0, server, attributed=False,
            vllm_init_profile=profile, device_vram_used=7 * GiB, device_vram_total=8 * GiB,
            vram_measured=False,
        )
        self.assertIsNone(view_unproven.vllm_breakdown.total_vram_used_bytes)
        self.assertIsNone(view_unproven.vllm_breakdown.other_runtime_bytes)
        self.assertEqual(view_unproven.vllm_breakdown.device_vram_used_bytes, 7 * GiB)
        self.assertEqual(view_unproven.segments, [])

    def test_collector_depth_4_ppid_chain(self):
        with listener(metrics=METRICS + CACHE_CONFIG_METRIC) as (port, seen):
            details = {
                10: ProcDetail(10, "VLLM::Worker_TP0", create_time=4),
                20: ProcDetail(20, "VLLM::Worker", create_time=3),
                30: ProcDetail(30, "VLLM::EngineCore", create_time=2),
                40: ProcDetail(40, "python", cmdline=LAUNCH, create_time=1),
            }
            _, snap = VllmTests().sample(details, {40: [port]}, {10: 20, 20: 30, 30: 40})
        row = snap.procs[0]
        self.assertEqual(row.llm.server_source, "parent")
        self.assertEqual(row.llm.server.models[0]["name"], "org/model")
        self.assertEqual(row.llm.vllm_breakdown.kv_num_gpu_blocks, 2048)
        self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)])

    def test_collector_config_yaml_matching_auth_and_api_failure_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "vllm.log"
            log.write_text(
                "2026-11-24T12:00:05Z INFO Model loading took 0.10 GiB memory and 1.000000 seconds\n"
                "2026-11-24T12:00:06Z INFO Available KV cache memory: 0.20 GiB\n",
                encoding="utf-8",
            )
            with listener(locked=True, metrics=METRICS + CACHE_CONFIG_METRIC) as (port, seen):
                # 1) match_pid on non-vllm_role VLLM::EngineCore_DP0 + auto_detect=False + api_key_env
                cfg = AppConfig(vllm=VllmConfig(
                    auto_detect=False,
                    endpoints=[
                        VllmEndpointConfig(
                            url=f"http://127.0.0.1:{port}",
                            name="local-qwen",
                            log_path=str(log),
                            api_key_env="TEST_VLLM_KEY",
                            gpu_index=0,
                            match_pid=10,
                        ),
                        VllmEndpointConfig(
                            url="http://127.0.0.1:9999",
                            name="other-gpu",
                            gpu_index=1,
                        ),
                    ],
                ))
                details = {10: ProcDetail(10, "VLLM::EngineCore_DP0", create_time=1795435200.0)}
                c = collector(details, {10: [port]})
                c.config = cfg
                nodes = chain(details, {})
                with patch.dict(os.environ, {"TEST_VLLM_KEY": "cfg-secret-key"}), \
                        patch("gputaskman.collector.BACKEND", "nvml"), \
                        patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes[pid]), \
                        patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details[pid].create_time):
                    snap = c.sample()
                row = snap.procs[0]
                self.assertEqual(row.llm.server_source, "config")
                self.assertIn("reported (config API): org/model", row.inside)
                self.assertEqual(row.llm.vllm_breakdown.endpoint_name, "local-qwen")
                self.assertEqual(row.llm.vllm_breakdown.weights_bytes, round(0.10 * GiB))
                self.assertEqual(row.llm.vllm_breakdown.kv_allocated_bytes, round(0.20 * GiB))
                self.assertIsNotNone(row.llm.vllm_breakdown.other_runtime_bytes)
                self.assertEqual(seen, [("/v1/models", "Bearer cfg-secret-key"), ("/metrics", None)])
                self.assertNotIn("cfg-secret-key", repr(vars(c.prober)))

            # 2) match_model disambiguation across two endpoints
            with listener(metrics=METRICS) as (port_a, _), listener(metrics=METRICS) as (port_b, _):
                cfg_m = AppConfig(vllm=VllmConfig(
                    auto_detect=False,
                    endpoints=[
                        VllmEndpointConfig(url=f"http://127.0.0.1:{port_a}", name="ep-a", match_model="org/other"),
                        VllmEndpointConfig(url=f"http://127.0.0.1:{port_b}", name="ep-b", match_model="org/model"),
                    ],
                ))
                details_m = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                             20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
                c_m = collector(details_m, {})
                c_m.config = cfg_m
                nodes_m = chain(details_m, {10: 20})
                with patch("gputaskman.collector.BACKEND", "nvml"), \
                        patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes_m[pid]), \
                        patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details_m[pid].create_time):
                    snap_m = c_m.sample()
                self.assertEqual(snap_m.procs[0].llm.vllm_breakdown.endpoint_name, "ep-b")

            # 3) API unreachable fallback still yields init-log + NVML breakdown
            cfg_down = AppConfig(vllm=VllmConfig(
                auto_detect=False,
                log_paths=[str(log)],
                endpoints=[VllmEndpointConfig(url="http://127.0.0.1:1", name="down-ep", timeout_s=0.1)],
            ))
            details_d = {10: ProcDetail(10, "VLLM::EngineCore", create_time=1795435200.0)}
            c_d = collector(details_d, {})
            c_d.config = cfg_down
            nodes_d = chain(details_d, {})
            with patch("gputaskman.llm._get", return_value=None), \
                    patch("gputaskman.collector.BACKEND", "nvml"), \
                    patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes_d[pid]), \
                    patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details_d[pid].create_time):
                snap_d = c_d.sample()
            row_d = snap_d.procs[0]
            self.assertIsNone(row_d.llm.server)
            self.assertEqual(row_d.llm.vllm_breakdown.endpoint_name, "down-ep")
            self.assertEqual(row_d.llm.vllm_breakdown.weights_bytes, round(0.10 * GiB))
            self.assertEqual(row_d.llm.vllm_breakdown.kv_allocated_bytes, round(0.20 * GiB))
            self.assertEqual(row_d.llm.vllm_breakdown.total_vram_used_bytes, 512 * MiB)
            self.assertTrue(any("unreachable" in n for n in row_d.llm.notes))

            # 4) Unproven container/WSL row keeps total_vram_used_bytes=None
            c_u = collector(details_d, {})
            c_u.config = cfg_down
            c_u.counters.sample.return_value = (
                {0: {10: ProcGpu(dedicated=0, shared=0, inferred=512 * MiB, inferred_by="order",
                                 attributed=False, available=set())}},
                {0: AdapterGpu(dedicated=512 * MiB, shared=0)},
            )
            with patch("gputaskman.llm._get", return_value=None), \
                    patch("gputaskman.collector.BACKEND", "nvml"), \
                    patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes_d[pid]), \
                    patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details_d[pid].create_time):
                snap_u = c_u.sample()
            self.assertIsNone(snap_u.procs[0].llm.vllm_breakdown.total_vram_used_bytes)
            self.assertEqual(snap_u.procs[0].llm.vllm_breakdown.device_vram_used_bytes, 512 * MiB)
            self.assertEqual(snap_u.procs[0].llm.segments, [])

            # 5) 401 on configured endpoint records locked note without leaking secret; multiple ambiguous endpoints stay unmatched
            with listener(locked=True, metrics=METRICS) as (port_lock, _):
                cfg_lock = AppConfig(vllm=VllmConfig(
                    auto_detect=False,
                    endpoints=[VllmEndpointConfig(url=f"http://127.0.0.1:{port_lock}", name="locked-ep")],
                ))
                c_l = collector(details_d, {})
                c_l.config = cfg_lock
                with patch("gputaskman.collector.BACKEND", "nvml"), \
                        patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes_d[pid]), \
                        patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details_d[pid].create_time):
                    snap_l = c_l.sample()
                self.assertIsNone(snap_l.procs[0].llm.server)
                self.assertTrue(any("401" in n for n in snap_l.procs[0].llm.notes))

                cfg_amb = AppConfig(vllm=VllmConfig(
                    auto_detect=False,
                    endpoints=[
                        VllmEndpointConfig(url=f"http://127.0.0.1:{port_lock}", name="a"),
                        VllmEndpointConfig(url="http://127.0.0.1:9998", name="b"),
                    ],
                ))
                c_a = collector(details_d, {})
                c_a.config = cfg_amb
                with patch("gputaskman.collector.BACKEND", "nvml"), \
                        patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes_d[pid]), \
                        patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details_d[pid].create_time):
                    snap_a = c_a.sample()
                self.assertIsNone(snap_a.procs[0].llm.server)
                self.assertTrue(any("Multiple config.yaml" in n for n in snap_a.procs[0].llm.notes))

    def test_vllm_metrics_and_tokens_match_hf_snapshot_and_launch_model_path(self):
        hf_snap = "/cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167"
        hf_metrics = (
            METRICS.replace('model_name="org/model"', f'model_name="{hf_snap}"')
            + f'vllm:generation_tokens_total{{engine="0",model_name="{hf_snap}"}} 500.0\n'
        )
        # (b) Standard HF Hub cache snapshot path matches served <org>/<repo>
        self.assertEqual(
            vllm_metrics(hf_metrics, ["Qwen/Qwen2.5-0.5B-Instruct"]),
            {"requests_running": 3, "requests_waiting": 0, "kv_cache_usage": 0.42},
        )
        t1 = vllm_token_counters(hf_metrics, ["Qwen/Qwen2.5-0.5B-Instruct"], 10.0)
        self.assertEqual(t1, TokenSample(10.0, ("0", hf_snap), 1234.0, 500.0))
        t2 = vllm_token_counters(
            hf_metrics.replace("1234.0", "1534.0").replace("500.0", "650.0"),
            ["Qwen/Qwen2.5-0.5B-Instruct"],
            13.0,
        )
        self.assertEqual(vllm_token_rates(t1, t2), (100.0, 50.0))

        # Repo containing '--' preserves everything after the first '--' separator
        dashed_snap = "/cache/hub/models--my-org--my--repo/snapshots/rev123/"
        dashed_metrics = METRICS.replace('model_name="org/model"', f'model_name="{dashed_snap}"')
        self.assertEqual(vllm_metrics(dashed_metrics, ["my-org/my--repo"])["requests_running"], 3)

        # Unrelated HF snapshot repo or unrelated alias stays rejected
        self.assertEqual(vllm_metrics(hf_metrics, ["Other/Model"]), NO_GAUGES)
        self.assertIsNone(vllm_token_counters(hf_metrics, ["Other/Model"], 10.0))
        unrelated = METRICS.replace('model_name="org/model"', 'model_name="production-llm"')
        self.assertEqual(vllm_metrics(unrelated, ["org/model"]), NO_GAUGES)

        # (c) Caller-supplied launch --model path (non-HF local dir) matches when /v1/models advertises a model
        local_dir = "/mnt/weights/custom-qwen"
        local_metrics = (
            METRICS.replace('model_name="org/model"', f'model_name="{local_dir}"')
            + f'vllm:generation_tokens_total{{engine="0",model_name="{local_dir}"}} 250.0\n'
        )
        self.assertEqual(vllm_metrics(local_metrics, ["custom-alias"]), NO_GAUGES)
        self.assertEqual(
            vllm_metrics(local_metrics, ["custom-alias"], extra_model_ids=[local_dir + "/"]),
            {"requests_running": 3, "requests_waiting": 0, "kv_cache_usage": 0.42},
        )
        self.assertEqual(
            vllm_token_counters(local_metrics, ["custom-alias"], 5.0, extra_model_ids=[local_dir]),
            TokenSample(5.0, ("0", local_dir), 1234.0, 250.0),
        )
        # Empty /v1/models never accepts metrics even with extra_model_ids
        self.assertEqual(vllm_metrics(local_metrics, [], extra_model_ids=[local_dir]), NO_GAUGES)

        # End-to-end probe_vllm_url with HF snapshot metrics and served model ID
        with listener(metrics=hf_metrics + CACHE_CONFIG_METRIC, model_id="Qwen/Qwen2.5-0.5B-Instruct") as (port, _):
            srv = probe_vllm_url(f"http://127.0.0.1:{port}")
        self.assertIsNotNone(srv)
        self.assertEqual(srv.models[0]["name"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual((srv.requests_running, srv.requests_waiting, srv.kv_cache_usage), (3, 0, 0.42))
        self.assertEqual(srv.cache_config["num_gpu_blocks"], 2048)

    def test_collector_ephemeral_zmq_worker_ports_still_probe_parent_server(self):
        hf_snap = "/cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae55760"
        hf_metrics = METRICS.replace('model_name="org/model"', f'model_name="{hf_snap}"') + CACHE_CONFIG_METRIC
        zmq_ports = [33273, 38365, 39287, 44623, 45009, 46845]

        # 1) Single-GPU vLLM V1: VLLM::EngineCore has multiple ephemeral ZMQ ports -> still probes parent server
        with listener(metrics=hf_metrics, model_id="Qwen/Qwen2.5-0.5B-Instruct") as (port, seen):
            details = {
                10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                20: ProcDetail(
                    20, "python",
                    cmdline=["vllm", "serve", hf_snap, "--served-model-name", "Qwen/Qwen2.5-0.5B-Instruct"],
                    create_time=1,
                ),
            }
            _, snap = VllmTests().sample(details, {10: zmq_ports, 20: [port]}, {10: 20})
        row = snap.procs[0]
        self.assertEqual(row.llm.server_source, "parent")
        self.assertEqual(row.llm.server.models[0]["name"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual((row.llm.server.requests_running, row.llm.server.requests_waiting, row.llm.server.kv_cache_usage),
                         (3, 0, 0.42))
        self.assertEqual(row.llm.vllm_breakdown.kv_num_gpu_blocks, 2048)
        self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)])

        # 2) Multi-hop TP layout: VLLM::Worker_TP0 -> VLLM::EngineCore (ephemeral ZMQ ports) -> vllm serve
        local_dir = "/mnt/weights/local-qwen"
        local_metrics = METRICS.replace('model_name="org/model"', f'model_name="{local_dir}"')
        with listener(metrics=local_metrics, model_id="local-alias") as (port, seen):
            details_tp = {
                10: ProcDetail(10, "VLLM::Worker_TP0", create_time=3),
                20: ProcDetail(20, "VLLM::EngineCore", create_time=2),
                30: ProcDetail(
                    30, "python",
                    cmdline=["vllm", "serve", local_dir, "--served-model-name", "local-alias"],
                    create_time=1,
                ),
            }
            _, snap_tp = VllmTests().sample(
                details_tp,
                {10: [41000, 41001], 20: zmq_ports, 30: [port]},
                {10: 20, 20: 30},
            )
        row_tp = snap_tp.procs[0]
        self.assertEqual(row_tp.llm.server_source, "parent")
        self.assertEqual(row_tp.llm.server.models[0]["name"], "local-alias")
        self.assertEqual(row_tp.llm.server.requests_running, 3)
        self.assertEqual(seen, [("/v1/models", None), ("/metrics", None)])

        # 3) Negative guards: mixed low+ephemeral worker ports, multi-ephemeral server, or ambiguous worker never probe
        neg_cases = [
            ("mixed worker ports", {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                                    20: ProcDetail(20, "python", cmdline=list(LAUNCH), create_time=1)},
             {10: [8001, 40000], 20: [8000]}, {10: 20}),
            ("multi-ephemeral server", {10: ProcDetail(10, "python", cmdline=list(LAUNCH), create_time=1)},
             {10: [40000, 40001]}, {}),
            ("ambiguous ephemeral worker",
             {10: ProcDetail(10, "VLLM::EngineCore", cmdline=list(LAUNCH) + ["--data-parallel-size", "2"], create_time=2),
              20: ProcDetail(20, "python", cmdline=list(LAUNCH), create_time=1)},
             {10: zmq_ports, 20: [8000]}, {10: 20}),
        ]
        for label, neg_details, neg_ports, neg_parents in neg_cases:
            with self.subTest(case=label), patch("gputaskman.llm.probe_server") as probe:
                _, neg_snap = VllmTests().sample(neg_details, neg_ports, neg_parents)
            probe.assert_not_called()
            self.assertIsNone(neg_snap.procs[0].llm.server)


class VllmReportAndCompatTests(unittest.TestCase):
    def _vllm_snapshot(
        self,
        *,
        taken: float = 1795435225.0,
        vram_used: int = 7 * GiB,
        kv_usage: float | None = 0.25,
        with_init_profile: bool = True,
        stale: bool = False,
        attributed: bool = True,
        running: int | None = 2,
        waiting: int | None = 1,
    ):
        from gputaskman.collector import ProcRow, Snapshot

        server = ServerInfo(
            "vllm",
            "http://127.0.0.1:8000",
            models=[{"name": "org/model"}],
            requests_running=running,
            requests_waiting=waiting,
            kv_cache_usage=kv_usage,
            cache_config={
                "engine": "0",
                "block_size": 16,
                "num_gpu_blocks": 1024,
                "cache_dtype": "auto",
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
                "kv_cache_memory_bytes": None,
                "kv_cache_size_tokens": 16384,
            },
            probed_at=taken - 5.0,
        )
        profile = (
            VllmInitProfile(
                path="/tmp/vllm.log",
                weights_bytes=2 * GiB,
                kv_allocated_bytes=4 * GiB,
                kv_cache_size_tokens=16384,
                torch_peak_increase_bytes=512 * MiB,
                non_torch_increase_bytes=128 * MiB,
                activation_peak_bytes=640 * MiB,
                cuda_graph_bytes=256 * MiB,
                timestamp="2026-11-24T12:00:10Z",
                timestamp_epoch=1795435210.0,
                stale=stale,
                stale_reason="init log timestamp predates process start" if stale else None,
            )
            if with_init_profile
            else None
        )
        vllm_detail = ProcDetail(10, "VLLM::EngineCore", create_time=1795435200.0)
        vllm_gpu = ProcGpu(
            dedicated=vram_used if attributed else 0,
            shared=0,
            attributed=attributed,
            available={"dedicated", "engine"} if attributed else set(),
            engines={"compute": 35.0} if attributed else {},
        )
        vllm_view = build_view(
            vllm_detail,
            vram_used if attributed else 0,
            0,
            server,
            attributed=attributed,
            server_source="config",
            vllm_init_profile=profile,
            vllm_endpoint_url="http://127.0.0.1:8000",
            vllm_endpoint_name="local-qwen",
            gpu_id="gpu:0",
            device_vram_used=vram_used + 256 * MiB,
            device_vram_total=8 * GiB,
            vram_measured=attributed,
            sample_time=taken,
        )
        vllm_row = ProcRow(10, "VLLM::EngineCore", "LLM", vllm_gpu, vllm_detail, llm=vllm_view)

        llama_detail = ProcDetail(20, "llama-server", cmdline=["llama-server", "-c", "4096"], create_time=1795435100.0)
        llama_server = ServerInfo("llama-server", "http://127.0.0.1:8080", n_ctx=4096, slots_busy=1, slots_total=4)
        llama_view = build_view(llama_detail, 1 * GiB, 0, llama_server, attributed=True)
        llama_row = ProcRow(20, "llama-server", "LLM", ProcGpu(dedicated=1 * GiB, shared=0), llama_detail, llm=llama_view)

        app_row = ProcRow(30, "desktop-app", "app", ProcGpu(dedicated=256 * MiB, shared=0), ProcDetail(30, "desktop-app"))

        return Snapshot(
            taken=taken,
            adapter=None,
            adapters=[],
            nv=None,
            driver="570.00",
            cuda="12.8",
            vram_total=8 * GiB,
            vram_used=vram_used + 256 * MiB,
            shared_used=0,
            procs=[vllm_row, llama_row, app_row],
            backend="nvml",
        )

    def test_report_to_dict_additive_vllm_memory_and_non_vllm_unchanged(self):
        snap = self._vllm_snapshot()
        data = to_dict(snap)
        encoded = json.dumps(data)
        self.assertNotIn("token_sample", encoded)
        self.assertNotIn("probed_at", encoded)

        vllm_proc, llama_proc, app_proc = data["processes"]
        self.assertIn("vllm_memory", vllm_proc["llm"])
        vm = vllm_proc["llm"]["vllm_memory"]
        self.assertEqual(vm["endpoint_name"], "local-qwen")
        self.assertEqual(vm["endpoint_url"], "http://127.0.0.1:8000")
        self.assertEqual(vm["weights_bytes"], 2 * GiB)
        self.assertEqual(vm["kv_allocated_bytes"], 4 * GiB)
        self.assertEqual(vm["other_runtime_bytes"], 1 * GiB)
        self.assertEqual(vm["activation_peak_bytes"], 640 * MiB)
        self.assertIsNone(vm["activation_current_bytes"])
        self.assertEqual(vm["kv_pool_usage_ratio"], 0.25)
        self.assertEqual(vm["kv_pool_occupied_bytes"], 1 * GiB)
        self.assertEqual(vm["kv_pool_available_bytes"], 3 * GiB)
        self.assertIsNone(vm["kv_active_request_bytes"])
        self.assertEqual(vm["running_requests"], 2)
        self.assertEqual(vm["waiting_requests"], 1)
        self.assertFalse(vm["stale"])
        self.assertIsNone(vm["stale_reason"])
        self.assertEqual(vm["data_sources"]["total_vram_used_bytes"], "NVML")
        self.assertEqual(vm["data_sources"]["weights_bytes"], "init_log:/tmp/vllm.log")
        self.assertEqual(vm["data_sources"]["kv_pool_usage_ratio"], "vllm:/metrics:kv_cache_usage_perc")
        self.assertEqual(vm["measurement_confidence"]["total_vram_used_bytes"], "measured")
        self.assertEqual(vm["measurement_confidence"]["weights_bytes"], "init_profile")
        self.assertEqual(vm["measurement_confidence"]["kv_pool_occupied_bytes"], "derived")
        self.assertEqual(vm["measurement_confidence"]["activation_current_bytes"], "unavailable")
        self.assertEqual(vm["measurement_confidence"]["kv_active_request_bytes"], "unavailable")
        self.assertIsNotNone(vm["measurement_timestamps"]["total_vram_used_bytes"])
        self.assertEqual(vm["measurement_timestamps"]["weights_bytes"], "2026-11-24T12:00:10Z")
        self.assertIsNotNone(vm["measurement_timestamps"]["kv_pool_usage_ratio"])

        # Existing additive contract: non-vLLM LLM rows and regular apps stay untouched
        self.assertIn("llm", llama_proc)
        self.assertNotIn("vllm_memory", llama_proc["llm"])
        self.assertEqual(llama_proc["llm"]["server"]["kind"], "llama-server")
        self.assertNotIn("llm", app_proc)

    def test_report_to_text_vllm_breakdown_and_fallbacks(self):
        # 1) Full Strategy B breakdown
        text = to_text(self._vllm_snapshot())
        self.assertIn("vLLM Memory Breakdown  local-qwen (http://127.0.0.1:8000)  [config]", text)
        self.assertIn("GPU Used: 7.00 GB / 8.00 GB  [NVML]", text)
        self.assertIn("Weights: 2.00 GB  [Init profile]", text)
        self.assertIn("Activation Peak: 640 MB  [Init peak, NOT current]", text)
        self.assertIn("Activation Current: N/A  [requires instrumentation]", text)
        self.assertIn("CUDA Graph: 256 MB  [Init profile]", text)
        self.assertIn("KV Cache Allocated: 4.00 GB  [Init profile]", text)
        self.assertIn("KV Pool Occupied: 1.00 GB (25.0%)  [derived; includes prefix cache]", text)
        self.assertIn("KV Pool Available: 3.00 GB  [derived]", text)
        self.assertIn("Active Requests KV: N/A  [requires instrumentation]", text)
        self.assertIn("Other/Unattributed: 1.00 GB  [NVML diff, derived]", text)
        self.assertIn("Running / Waiting: 2 / 1", text)
        self.assertIn("Sources: NVML, Init Profile, Metrics", text)
        self.assertEqual(text.count("vLLM Memory Breakdown"), 1)

        # 2) Unknown gauges + missing init log must not emit 'vLLM /metrics', 'KV-cache pool', or 'token rates'
        unknown_text = to_text(self._vllm_snapshot(
            kv_usage=None, with_init_profile=False, running=None, waiting=None,
        ))
        self.assertIn("Weights: N/A  [init log unavailable]", unknown_text)
        self.assertIn("Activation Peak: N/A  [init DEBUG log unavailable]", unknown_text)
        self.assertIn("KV Cache Allocated: N/A  [init log / engine config unavailable]", unknown_text)
        self.assertIn("KV Pool Occupied: N/A  [metrics unavailable]", unknown_text)
        self.assertIn("Other/Unattributed: N/A  [requires reconciliation]", unknown_text)
        self.assertIn("Running / Waiting: N/A / N/A", unknown_text)
        self.assertNotIn("vLLM /metrics", unknown_text)
        self.assertNotIn("KV-cache pool", unknown_text)
        self.assertNotIn("token rates", unknown_text)

        # 3) Stale init profile warning
        stale_text = to_text(self._vllm_snapshot(stale=True))
        self.assertIn("[STALE]", stale_text)
        self.assertIn("[Init peak, NOT current, STALE]", stale_text)
        self.assertIn("Other/Unattributed: N/A  [requires reconciliation]", stale_text)
        self.assertIn("! STALE init log: init log timestamp predates process start", stale_text)

        # 4) Unattributed process VRAM (WSL/container) reports device VRAM separately
        unattr_text = to_text(self._vllm_snapshot(attributed=False))
        self.assertIn("GPU Used: N/A  [unattributed]", unattr_text)
        self.assertIn("Device VRAM: 7.25 GB / 8.00 GB  [NVML device-wide]", unattr_text)
        self.assertIn("Other/Unattributed: N/A  [requires reconciliation]", unattr_text)

    def test_record_diff_summary_replay_roundtrip_with_vllm_memory(self):
        from gputaskman import diff as diffmod
        from gputaskman import record as recordmod
        from gputaskman import replay as replaymod
        from gputaskman import summary as summarymod

        snaps = [
            self._vllm_snapshot(taken=1795435220.0, vram_used=6 * GiB + 512 * MiB, kv_usage=0.10, running=1, waiting=0),
            self._vllm_snapshot(taken=1795435222.0, vram_used=7 * GiB, kv_usage=0.50, running=3, waiting=1),
        ]
        it = iter(snaps)
        clock_now = [1000.0]

        def clock():
            return clock_now[0]

        def sleep(sec):
            clock_now[0] += sec

        with tempfile.TemporaryDirectory() as td:
            rec_path = os.path.join(td, "vllm.jsonl")
            with recordmod.append_to(rec_path) as out:
                count = recordmod.run(
                    recordmod.make_sample(lambda: next(it)),
                    out,
                    interval=1.0,
                    duration=2.0,
                    clock=clock,
                    sleep=sleep,
                )
            self.assertEqual(count, 2)

            with open(rec_path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["processes"][0]["llm"]["vllm_memory"]["kv_pool_usage_ratio"], 0.10)
            self.assertEqual(lines[1]["processes"][0]["llm"]["vllm_memory"]["kv_pool_usage_ratio"], 0.50)
            self.assertNotIn("vllm_memory", lines[0]["processes"][1]["llm"])

            # diff compatibility
            before_path = os.path.join(td, "before.json")
            after_path = os.path.join(td, "after.json")
            Path(before_path).write_text(json.dumps(lines[0]), encoding="utf-8")
            Path(after_path).write_text(json.dumps(lines[1]), encoding="utf-8")
            d = diffmod.diff(diffmod.load(before_path), diffmod.load(after_path))
            self.assertEqual(d["device"]["vram_used"]["delta"], 512 * MiB)
            vllm_diff = next(p for p in d["processes"] if p["name"] == "VLLM::EngineCore")
            self.assertEqual(vllm_diff["measured"]["vram_bytes"]["delta"], 512 * MiB)
            self.assertEqual(vllm_diff["runtime"]["segments"]["weights"]["after"], 2 * GiB)
            self.assertIn("VLLM::EngineCore", diffmod.to_text(d))

            # summary compatibility
            samples, skipped = summarymod.load(rec_path)
            self.assertEqual((len(samples), skipped), (2, 0))
            summ = summarymod.summarize(samples, skipped)
            self.assertEqual(len(summ["runs"]), 1)
            run_procs = {p["name"]: p for p in summ["runs"][0]["processes"]}
            self.assertEqual(run_procs["VLLM::EngineCore"]["peak_vram_bytes"], 7 * GiB)
            self.assertIn("VLLM::EngineCore", summarymod.to_text(summ))

            # replay compatibility
            with replaymod.load_recording(rec_path) as rec:
                self.assertEqual(rec.count, 2)
                self.assertTrue(rec.timed)
                self.assertEqual(rec.warnings, [])
                pb = replaymod.Playback(rec.count, rec.taken, timed=rec.timed)
                snap0 = rec.snapshot(0)
                self.assertEqual(snap0["processes"][0]["llm"]["vllm_memory"]["weights_bytes"], 2 * GiB)
                desc = replaymod.describe(rec, pb, snap0)
                self.assertIn("GPU Session Replay", desc)
                rows, _ = replaymod.process_rows(snap0)
                self.assertTrue(any(r[1] == "VLLM::EngineCore" for r in rows))
                self.assertEqual(replaymod.peak_memory_index(rec.frames, 0), 1)

    def test_vllm_fd_log_autodiscovery_without_config(self):
        import tempfile
        from pathlib import Path
        from gputaskman.collector import ProcRow

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "vllm_stdout.log"
            log_path.write_text(
                "INFO 11-24 13:47:57 [model_runner.py:441] Model loading took 0.93 GiB memory and 4.7 seconds\n"
                "DEBUG 11-24 13:48:04 [gpu_worker.py:794] Memory profiling takes 6.1 seconds. "
                "torch peak memory increase: 0.07GiB\n"
                "INFO 11-24 13:48:04 [gpu_worker.py:795] Available KV cache memory: 4.12 GiB\n",
                encoding="utf-8",
            )
            detail = ProcDetail(10, "VLLM::EngineCore", create_time=time.time() - 60)
            row = ProcRow(10, "VLLM::EngineCore", "LLM", ProcGpu(dedicated=0, shared=0, attributed=False), detail)
            c = collector({10: detail}, {})

            def fake_readlink(p: str) -> str:
                if p == "/proc/10/fd/1":
                    return str(log_path)
                raise OSError("not found")

            with patch("gputaskman.collector.sys.platform", "linux"), patch("gputaskman.collector.os.readlink", side_effect=fake_readlink):
                profile = c._resolve_vllm_init_profile(row, None, {})
            self.assertIsNotNone(profile)
            self.assertEqual(profile.path, str(log_path))
            self.assertAlmostEqual(profile.weights_bytes / (1 << 30), 0.93, places=2)
            self.assertAlmostEqual(profile.kv_allocated_bytes / (1 << 30), 4.12, places=2)
            self.assertAlmostEqual(profile.activation_peak_bytes / (1 << 30), 0.07, places=2)

    def test_unmatched_docker_or_remote_config_endpoint_synthesizes_row(self):
        import tempfile
        from pathlib import Path
        from gputaskman import diff as diffmod

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "docker_vllm.log"
            log_path.write_text(
                "INFO 11-24 13:47:57 [model_runner.py:441] Model loading took 1.50 GiB memory and 3.0 seconds\n"
                "DEBUG 11-24 13:48:04 [gpu_worker.py:794] Memory profiling takes 5.0 seconds. "
                "torch peak memory increase: 0.25GiB\n"
                "INFO 11-24 13:48:04 [gpu_worker.py:795] Available KV cache memory: 3.00 GiB\n",
                encoding="utf-8",
            )
            with listener(metrics=METRICS) as (port, _):
                # No local vLLM PID in /proc (simulates Docker container / sidecar / remote vLLM).
                c = collector({}, {})
                c.counters.sample.return_value = ({0: {}}, {0: AdapterGpu(dedicated=6 * (1 << 30), shared=0)})
                c.config = AppConfig(
                    vllm=VllmConfig(
                        auto_detect=True,
                        endpoints=[
                            VllmEndpointConfig(
                                url=f"http://127.0.0.1:{port}",
                                name="docker-qwen",
                                log_path=str(log_path),
                                gpu_index=0,
                            ),
                            VllmEndpointConfig(
                                url="http://127.0.0.1:9999",
                                name="other-gpu",
                                gpu_index=1,
                            ),
                        ],
                    )
                )
                with patch("gputaskman.collector.BACKEND", "nvml"):
                    snap = c.sample()

            self.assertEqual(len(snap.procs), 1)
            row = snap.procs[0]
            self.assertTrue(row.host)
            self.assertEqual(row.origin, "config")
            self.assertEqual(row.pid_label, "C:1")
            self.assertEqual(row.name, "docker-qwen")
            self.assertEqual(row.kind, "LLM")
            # Synthesized config row must NEVER inflate measured Snapshot.attributed.
            self.assertFalse(row.gpu.attributed)
            self.assertEqual(snap.attributed, 0)
            self.assertIsNotNone(row.llm)
            self.assertIsNotNone(row.llm.vllm_breakdown)
            bd = row.llm.vllm_breakdown
            self.assertEqual(bd.match_source, "config")
            self.assertAlmostEqual(bd.weights_bytes / (1 << 30), 1.50, places=2)
            self.assertAlmostEqual(bd.kv_allocated_bytes / (1 << 30), 3.00, places=2)
            self.assertAlmostEqual(bd.activation_peak_bytes / (1 << 30), 0.25, places=2)
            self.assertAlmostEqual(bd.kv_pool_usage_ratio, 0.42, places=3)
            self.assertIsNone(bd.activation_current_bytes)
            self.assertIsNone(bd.kv_active_request_bytes)
            self.assertIn("reported (config API): org/model", row.inside)

            # JSON, plain text, and diff must all work cleanly on synthesized config rows.
            d = to_dict(snap)
            self.assertEqual(d["processes"][0]["host"], "config")
            self.assertIsNone(d["processes"][0]["vram_bytes"])
            self.assertEqual(d["processes"][0]["llm"]["vllm_memory"]["endpoint_name"], "docker-qwen")
            txt = to_text(snap)
            self.assertIn("C:1", txt)
            self.assertIn("vLLM Memory Breakdown", txt)
            cmp = diffmod.diff(d, d)
            self.assertIn("processes", cmp)
