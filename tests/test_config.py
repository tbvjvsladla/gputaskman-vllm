"""Tests for the optional 4-level YAML configuration loader and CLI/collector/app/record wiring."""

from __future__ import annotations

import contextlib
import functools
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gputaskman import collector as collector_mod
from gputaskman import record
from gputaskman.app import GpuTaskManager
from gputaskman.cli import main
from gputaskman.collector import Collector, Snapshot
from gputaskman.config import (
    ENV_CONFIG_PATH,
    AppConfig,
    ConfigError,
    VllmConfig,
    VllmEndpointConfig,
    load_config,
)
from gputaskman.dxgi import Adapter

GiB = 1 << 30


def _sample_snapshot() -> Snapshot:
    adapter = Adapter(1, "NVIDIA RTX 4070", 0x10DE, 8 * GiB, 0, False)
    return Snapshot(
        taken=1.0,
        adapter=adapter,
        adapters=[adapter],
        nv=None,
        driver="560.0",
        cuda="12.6",
        vram_total=8 * GiB,
        vram_used=2 * GiB,
        shared_used=0,
        procs=[],
        backend="nvml",
    )


class ConfigLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cwd = self.root / "work"
        self.home = self.root / "home"
        self.cwd.mkdir()
        self.home.mkdir()

    def _write(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_zero_config_returns_defaults_when_no_file_exists(self):
        cfg = load_config(env={}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg, AppConfig())
        self.assertIsNone(cfg.source)
        self.assertEqual(cfg.source_priority, "default")
        self.assertTrue(cfg.vllm.auto_detect)
        self.assertEqual(cfg.vllm.log_paths, [])
        self.assertEqual(cfg.vllm.endpoints, [])

    def test_four_level_priority_order(self):
        user_file = self._write(
            self.home / ".config" / "gputaskman" / "config.yaml",
            "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8004\n      name: user\n",
        )
        cwd_file = self._write(
            self.cwd / "gputaskman.yaml",
            "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8003\n      name: cwd\n",
        )
        env_file = self._write(
            self.root / "env.yaml",
            "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8002\n      name: env\n",
        )
        cli_file = self._write(
            self.root / "cli.yaml",
            "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8001\n      name: cli\n",
        )

        # Level 1: --config beats everything
        cfg = load_config(cli_file, env={ENV_CONFIG_PATH: str(env_file)}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg.source_priority, "cli")
        self.assertEqual(cfg.source, str(cli_file))
        self.assertEqual(cfg.vllm.endpoints[0].name, "cli")

        # Level 2: GPUTASKMAN_CONFIG beats cwd & user
        cfg = load_config(None, env={ENV_CONFIG_PATH: str(env_file)}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg.source_priority, "env")
        self.assertEqual(cfg.source, str(env_file))
        self.assertEqual(cfg.vllm.endpoints[0].name, "env")

        # Blank env var falls through to Level 3 (cwd)
        cfg = load_config(None, env={ENV_CONFIG_PATH: "   "}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg.source_priority, "cwd")
        self.assertEqual(cfg.source, str(cwd_file))
        self.assertEqual(cfg.vllm.endpoints[0].name, "cwd")

        # Level 4: user config when cwd file absent
        cwd_file.unlink()
        cfg = load_config(None, env={}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg.source_priority, "user")
        self.assertEqual(cfg.source, str(user_file))
        self.assertEqual(cfg.vllm.endpoints[0].name, "user")

    def test_empty_yaml_file_loads_with_defaults_and_source(self):
        empty = self._write(self.cwd / "gputaskman.yaml", "# comments only\n")
        cfg = load_config(env={}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg.source, str(empty))
        self.assertEqual(cfg.source_priority, "cwd")
        self.assertTrue(cfg.vllm.auto_detect)
        self.assertEqual(cfg.vllm.endpoints, [])

    def test_full_endpoint_schema_normalization_and_tilde_expansion(self):
        cfg_file = self._write(
            self.home / "custom.yaml",
            """
vllm:
  auto_detect: false
  log_paths:
    - ~/logs/vllm-global.log
  endpoints:
    - url: https://127.0.0.1:8443/v1///
      name: primary-vllm
      log_path: ~/logs/vllm-ep.log
      api_key_env: MY_VLLM_TOKEN
      gpu_index: 0
      match_pid: 12345
      match_model: Qwen/Qwen2.5-0.5B
      timeout_s: 3.5
""",
        )
        cfg = load_config("~/custom.yaml", env={}, cwd=self.cwd, home=self.home)
        self.assertEqual(cfg.source, str(cfg_file))
        self.assertFalse(cfg.vllm.auto_detect)
        self.assertEqual(cfg.vllm.log_paths, [str(self.home / "logs" / "vllm-global.log")])
        self.assertEqual(len(cfg.vllm.endpoints), 1)
        ep = cfg.vllm.endpoints[0]
        self.assertEqual(ep.url, "https://127.0.0.1:8443/v1")
        self.assertEqual(ep.name, "primary-vllm")
        self.assertEqual(ep.log_path, str(self.home / "logs" / "vllm-ep.log"))
        self.assertIsNone(ep.api_key)
        self.assertEqual(ep.api_key_env, "MY_VLLM_TOKEN")
        self.assertEqual(ep.gpu_index, 0)
        self.assertEqual(ep.match_pid, 12345)
        self.assertEqual(ep.match_model, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(ep.timeout_s, 3.5)

        # Env var ~ expansion works too
        env_cfg = load_config(None, env={ENV_CONFIG_PATH: "~/custom.yaml"}, cwd=self.cwd, home=self.home)
        self.assertEqual(env_cfg.source_priority, "env")
        self.assertEqual(env_cfg.source, str(cfg_file))

    def test_api_key_resolution_and_redaction(self):
        secret = "sk-super-secret-token-999"
        ep_inline = VllmEndpointConfig(
            url="http://127.0.0.1:8000/",
            api_key=secret,
            api_key_env="FALLBACK_ENV",
        )
        self.assertEqual(ep_inline.url, "http://127.0.0.1:8000")
        self.assertEqual(ep_inline.resolve_api_key({"FALLBACK_ENV": "other"}), secret)
        self.assertNotIn(secret, repr(ep_inline))
        self.assertNotIn(secret, str(ep_inline))

        app_cfg = AppConfig(vllm=VllmConfig(endpoints=[ep_inline]))
        self.assertNotIn(secret, repr(app_cfg))
        redacted = ep_inline.redacted_dict()
        self.assertNotIn("api_key", redacted)
        self.assertTrue(redacted["has_api_key"])
        self.assertNotIn(secret, json.dumps(app_cfg.redacted_dict()))

        ep_env = VllmEndpointConfig(url="http://127.0.0.1:8001", api_key_env="VLLM_TOKEN")
        self.assertEqual(ep_env.resolve_api_key({"VLLM_TOKEN": "env-secret"}), "env-secret")
        self.assertIsNone(ep_env.resolve_api_key({}))
        self.assertIsNone(ep_env.resolve_api_key({"VLLM_TOKEN": ""}))

        ep_none = VllmEndpointConfig(url="http://127.0.0.1:8002")
        self.assertIsNone(ep_none.resolve_api_key({"VLLM_TOKEN": "ignored"}))
        self.assertFalse(ep_none.redacted_dict()["has_api_key"])

    def test_error_messages_never_leak_api_key(self):
        secret = "sk-never-leak-this-value"
        bad_yaml = self._write(
            self.root / "bad.yaml",
            f"vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      api_key: [{secret}\n",
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(bad_yaml, env={}, cwd=self.cwd, home=self.home)
        self.assertNotIn(secret, str(ctx.exception))

        bad_field = self._write(
            self.root / "bad_field.yaml",
            f"vllm:\n  endpoints:\n    - url: ftp://127.0.0.1:8000\n      api_key: {secret}\n",
        )
        with self.assertRaises(ConfigError) as ctx2:
            load_config(bad_field, env={}, cwd=self.cwd, home=self.home)
        self.assertNotIn(secret, str(ctx2.exception))

    def test_missing_explicit_or_env_file_fails_loud(self):
        missing = self.root / "does-not-exist.yaml"
        with self.assertRaises(ConfigError):
            load_config(missing, env={}, cwd=self.cwd, home=self.home)
        with self.assertRaises(ConfigError):
            load_config(None, env={ENV_CONFIG_PATH: str(missing)}, cwd=self.cwd, home=self.home)
        with self.assertRaises(ConfigError):
            load_config("   ", env={}, cwd=self.cwd, home=self.home)

    def test_invalid_discovered_files_fail_loud(self):
        self._write(self.cwd / "gputaskman.yaml", "vllm: [not, a, mapping]\n")
        with self.assertRaises(ConfigError):
            load_config(env={}, cwd=self.cwd, home=self.home)

        (self.cwd / "gputaskman.yaml").unlink()
        self._write(self.home / ".config" / "gputaskman" / "config.yaml", "- list-root\n")
        with self.assertRaises(ConfigError):
            load_config(env={}, cwd=self.cwd, home=self.home)

    def test_schema_validation_rejects_bad_urls_types_and_unknown_keys(self):
        cases = [
            ("unknown-root", "other:\n  a: 1\n"),
            ("unknown-vllm", "vllm:\n  endpoint: []\n"),
            ("unknown-ep", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      bogus: 1\n"),
            ("missing-url", "vllm:\n  endpoints:\n    - name: no-url\n"),
            ("ftp-url", "vllm:\n  endpoints:\n    - url: ftp://127.0.0.1:8000\n"),
            ("no-host-url", "vllm:\n  endpoints:\n    - url: http://\n"),
            ("space-url", "vllm:\n  endpoints:\n    - url: 'http://127.0.0.1:8000/a b'\n"),
            ("bad-auto-detect", "vllm:\n  auto_detect: 'yes'\n"),
            ("str-log-paths", "vllm:\n  log_paths: /tmp/vllm.log\n"),
            ("empty-log-path", "vllm:\n  log_paths:\n    - ''\n"),
            ("bool-gpu-index", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      gpu_index: true\n"),
            ("neg-gpu-index", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      gpu_index: -1\n"),
            ("zero-match-pid", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      match_pid: 0\n"),
            ("bool-match-pid", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      match_pid: true\n"),
            ("zero-timeout", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      timeout_s: 0\n"),
            ("nan-timeout", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      timeout_s: .nan\n"),
            ("inf-timeout", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      timeout_s: .inf\n"),
            ("bool-timeout", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      timeout_s: true\n"),
            ("empty-api-key", "vllm:\n  endpoints:\n    - url: http://127.0.0.1:8000\n      api_key: ''\n"),
        ]
        for label, content in cases:
            with self.subTest(label=label):
                path = self._write(self.root / f"{label}.yaml", content)
                with self.assertRaises(ConfigError):
                    load_config(path, env={}, cwd=self.cwd, home=self.home)


class ConfigWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cfg_file = self.root / "gputaskman.yaml"
        self.cfg_file.write_text(
            "vllm:\n  auto_detect: false\n  endpoints:\n    - url: http://127.0.0.1:8000/\n",
            encoding="utf-8",
        )

    def test_cli_json_and_tui_forward_config(self):
        out = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--json", "--no-system", "--config", str(self.cfg_file)]), \
                patch("gputaskman.cli.time.sleep"), \
                patch("gputaskman.collector.Collector") as mock_collector, \
                contextlib.redirect_stdout(out):
            mock_collector.return_value.host = None
            mock_collector.return_value.counters = types.SimpleNamespace()
            mock_collector.return_value.sample.return_value = _sample_snapshot()
            main()
        passed = mock_collector.call_args.kwargs["config"]
        self.assertIsInstance(passed, AppConfig)
        self.assertEqual(passed.source_priority, "cli")
        self.assertFalse(passed.vllm.auto_detect)
        self.assertEqual(passed.vllm.endpoints[0].url, "http://127.0.0.1:8000")

        with patch("sys.argv", ["gputaskman", "--config", str(self.cfg_file)]), \
                patch("gputaskman.app.GpuTaskManager") as mock_app:
            main()
        tui_cfg = mock_app.call_args.kwargs["config"]
        self.assertIsInstance(tui_cfg, AppConfig)
        self.assertEqual(tui_cfg.source, str(self.cfg_file))
        mock_app.return_value.run.assert_called_once_with()

    def test_cli_missing_config_fails_with_exit_2_without_collecting(self):
        err = io.StringIO()
        missing = str(self.root / "missing.yaml")
        with patch("sys.argv", ["gputaskman", "--json", "--config", missing]), \
                patch("gputaskman.collector.Collector") as mock_collector, \
                contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)
        mock_collector.assert_not_called()

    def test_help_version_and_list_gpus_do_not_require_config(self):
        bad_env = {ENV_CONFIG_PATH: str(self.root / "nonexistent.yaml")}
        with patch.dict("os.environ", bad_env), \
                patch("sys.argv", ["gputaskman", "--list-gpus"]), \
                patch("gputaskman.collector.enumerate_adapters", return_value=[]), \
                patch("gputaskman.tcc.tcc_discovery_status", return_value=""), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main()

    def test_record_cli_forwards_config_and_rejects_bad_config_before_writing(self):
        out_jsonl = self.root / "out.jsonl"
        captured: list[dict] = []

        @contextlib.contextmanager
        def fake_source(gpu, *, host, system, config=None):
            captured.append({"gpu": gpu, "host": host, "system": system, "config": config})
            yield lambda: {"taken": 1.0}

        with patch("sys.argv", ["gputaskman", "record", "-o", str(out_jsonl), "--duration", "0.1",
                                "--config", str(self.cfg_file)]), \
                patch("gputaskman.record.source", fake_source), \
                patch("gputaskman.record.run", return_value=1), \
                contextlib.redirect_stderr(io.StringIO()):
            main()
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0]["config"], AppConfig)
        self.assertEqual(captured[0]["config"].source, str(self.cfg_file))

        bad_out = self.root / "should-not-exist.jsonl"
        with patch("sys.argv", ["gputaskman", "record", "-o", str(bad_out), "--config",
                                str(self.root / "missing.yaml")]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)
        self.assertFalse(bad_out.exists())

    def test_record_source_forwards_config_for_nvml_and_wddm(self):
        cfg = load_config(self.cfg_file, env={}, cwd=self.root, home=self.root)
        fake_c = MagicMock()
        fake_c.host = None
        fake_c.counters = types.SimpleNamespace()
        fake_c.sample.return_value = _sample_snapshot()

        with patch("gputaskman.record.time.sleep"), \
                patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.Collector", return_value=fake_c) as factory:
            with record.source(0, system=False, config=cfg) as sample:
                sample()
        factory.assert_called_once_with(0, host=True, config=cfg)

        with patch("gputaskman.record.time.sleep"), \
                patch("gputaskman.collector.BACKEND", "wddm"), \
                patch("gputaskman.recovery.LiveCollectors") as live:
            live.return_value.sample.return_value = [_sample_snapshot()]
            with record.source(0, system=False, config=cfg) as sample:
                sample()
        cf = live.call_args.kwargs["collector_factory"]
        self.assertIsInstance(cf, functools.partial)
        self.assertIs(cf.func, collector_mod.Collector)
        self.assertEqual(cf.keywords, {"config": cfg})

    def test_collector_class_default_and_app_poll_wiring(self):
        bare = Collector.__new__(Collector)
        self.assertIsNone(bare.config)

        cfg = load_config(self.cfg_file, env={}, cwd=self.root, home=self.root)
        app = GpuTaskManager(gpu_index=0, system=False, config=cfg)
        self.assertIs(app.config, cfg)

        # Verify poll() forwards config in both non-WDDM and WDDM branches
        class _StopWorker:
            def __init__(self) -> None:
                self.is_cancelled = True
                self.cancelled_event = MagicMock()

        fake_c = MagicMock()
        with patch("gputaskman.app.BACKEND", "nvml"), \
                patch("gputaskman.app.get_current_worker", return_value=_StopWorker()), \
                patch("gputaskman.app.Collector", return_value=fake_c) as mock_c:
            mock_c.SCAN_BUDGET = Collector.SCAN_BUDGET
            app.poll.__wrapped__(app)
        self.assertIs(mock_c.call_args.kwargs["config"], cfg)

        with patch("gputaskman.app.BACKEND", "wddm"), \
                patch("gputaskman.app.get_current_worker", return_value=_StopWorker()), \
                patch("gputaskman.app.LiveCollectors") as mock_live:
            app.poll.__wrapped__(app)
        cf = mock_live.call_args.kwargs["collector_factory"]
        self.assertIsInstance(cf, functools.partial)
        self.assertIs(cf.func, Collector)
        self.assertEqual(cf.keywords, {"config": cfg})

    def test_relative_log_paths_resolve_against_yaml_directory_from_any_cwd(self):
        cfg_dir = self.root / "nested" / "yaml_dir"
        cfg_dir.mkdir(parents=True)
        other_cwd = self.root / "elsewhere"
        other_cwd.mkdir()
        rel_cfg = cfg_dir / "my_vllm.yaml"
        rel_cfg.write_text(
            "vllm:\n"
            "  log_paths:\n"
            "    - ./logs/shared.log\n"
            "  endpoints:\n"
            "    - name: docker-vllm\n"
            "      url: http://127.0.0.1:8000\n"
            "      log_path: ../shared/vllm.log\n",
            encoding="utf-8",
        )
        cfg = load_config(rel_cfg, env={}, cwd=other_cwd, home=self.root)
        self.assertEqual(cfg.source, str(rel_cfg.resolve()))
        self.assertEqual(cfg.vllm.log_paths, [str((cfg_dir / "logs" / "shared.log").resolve())])
        self.assertEqual(cfg.vllm.endpoints[0].log_path, str((cfg_dir.parent / "shared" / "vllm.log").resolve()))


if __name__ == "__main__":
    unittest.main()
