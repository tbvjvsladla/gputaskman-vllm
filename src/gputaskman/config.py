"""Optional YAML configuration loader with 4-level priority and zero-config defaults."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

ENV_CONFIG_PATH = "GPUTASKMAN_CONFIG"
CWD_CONFIG_NAME = "gputaskman.yaml"
USER_CONFIG_REL = Path(".config") / "gputaskman" / "config.yaml"

_ALLOWED_ROOT_KEYS = frozenset({"vllm"})
_ALLOWED_VLLM_KEYS = frozenset({"auto_detect", "log_paths", "endpoints"})
_ALLOWED_ENDPOINT_KEYS = frozenset({
    "url",
    "name",
    "log_path",
    "api_key",
    "api_key_env",
    "gpu_index",
    "match_pid",
    "match_model",
    "timeout_s",
})


class ConfigError(ValueError):
    """Raised when an explicit or discovered config file cannot be read or fails validation."""


def _expand_path(raw: str | os.PathLike[str], home: Path | None = None, base_dir: Path | None = None) -> str:
    text = os.fspath(raw)
    if home is not None:
        if text == "~":
            return str(home)
        if text.startswith("~/") or (os.sep != "/" and text.startswith(f"~{os.sep}")):
            return str(home / text[2:])
    expanded = os.path.expanduser(text)
    p = Path(expanded)
    if base_dir is not None and not p.is_absolute():
        return str((base_dir / p).resolve())
    return expanded


def _validate_url(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty http/https URL string")
    cleaned = value.strip()
    if any(ch.isspace() for ch in cleaned):
        raise ConfigError(f"{field_name} must not contain whitespace")
    parsed = urlsplit(cleaned)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ConfigError(f"{field_name} must use http or https with a host")
    return cleaned.rstrip("/")


def _validate_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_optional_int(value: Any, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        op = ">=" if minimum == 0 else ">"
        bound = minimum if minimum == 0 else minimum - 1
        raise ConfigError(f"{field_name} must be an integer {op} {bound}")
    return value


def _validate_timeout(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a finite number greater than zero")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfigError(f"{field_name} must be a finite number greater than zero")
    return result


@dataclass
class VllmEndpointConfig:
    """Configuration for one explicit vLLM HTTP endpoint and optional init log path."""

    url: str
    name: str | None = None
    log_path: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    gpu_index: int | None = None
    match_pid: int | None = None
    match_model: str | None = None
    timeout_s: float = 2.0

    def __post_init__(self) -> None:
        self.url = _validate_url(self.url, "vllm.endpoints[].url")
        self.name = _validate_optional_str(self.name, "vllm.endpoints[].name")
        raw_log = _validate_optional_str(self.log_path, "vllm.endpoints[].log_path")
        self.log_path = _expand_path(raw_log) if raw_log is not None else None
        if self.api_key is not None:
            if not isinstance(self.api_key, str) or not self.api_key.strip():
                raise ConfigError("vllm.endpoints[].api_key must be a non-empty string")
        self.api_key_env = _validate_optional_str(self.api_key_env, "vllm.endpoints[].api_key_env")
        self.gpu_index = _validate_optional_int(self.gpu_index, "vllm.endpoints[].gpu_index", minimum=0)
        self.match_pid = _validate_optional_int(self.match_pid, "vllm.endpoints[].match_pid", minimum=1)
        self.match_model = _validate_optional_str(self.match_model, "vllm.endpoints[].match_model")
        self.timeout_s = _validate_timeout(self.timeout_s, "vllm.endpoints[].timeout_s")

    def resolve_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        """Return inline api_key if set, else the environment variable named by api_key_env."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            env_map = os.environ if environ is None else environ
            value = env_map.get(self.api_key_env)
            if value:
                return value
        return None

    def redacted_dict(self) -> dict[str, Any]:
        """Serialize endpoint config without ever exposing api_key material."""
        return {
            "url": self.url,
            "name": self.name,
            "log_path": self.log_path,
            "has_api_key": bool(self.api_key or self.api_key_env),
            "api_key_env": self.api_key_env,
            "gpu_index": self.gpu_index,
            "match_pid": self.match_pid,
            "match_model": self.match_model,
            "timeout_s": self.timeout_s,
        }


@dataclass
class VllmConfig:
    """vLLM discovery and explicit endpoint configuration."""

    auto_detect: bool = True
    log_paths: list[str] = field(default_factory=list)
    endpoints: list[VllmEndpointConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.auto_detect, bool):
            raise ConfigError("vllm.auto_detect must be a boolean")
        if not isinstance(self.log_paths, list):
            raise ConfigError("vllm.log_paths must be a list of file path strings")
        normalized_logs: list[str] = []
        for idx, item in enumerate(self.log_paths):
            cleaned = _validate_optional_str(item, f"vllm.log_paths[{idx}]")
            if cleaned is None:
                raise ConfigError(f"vllm.log_paths[{idx}] must be a non-empty string")
            normalized_logs.append(_expand_path(cleaned))
        self.log_paths = normalized_logs
        if not isinstance(self.endpoints, list) or not all(
                isinstance(ep, VllmEndpointConfig) for ep in self.endpoints):
            raise ConfigError("vllm.endpoints must be a list of VllmEndpointConfig items")

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "auto_detect": self.auto_detect,
            "log_paths": list(self.log_paths),
            "endpoints": [ep.redacted_dict() for ep in self.endpoints],
        }


@dataclass
class AppConfig:
    """Top-level gputaskman configuration."""

    vllm: VllmConfig = field(default_factory=VllmConfig)
    source: str | None = None
    source_priority: str = "default"

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_priority": self.source_priority,
            "vllm": self.vllm.redacted_dict(),
        }


def _check_unknown_keys(mapping: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(str(k) for k in mapping if k not in allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {context}: {', '.join(unknown)}")


def _parse_endpoint(raw: Any, index: int, home: Path | None, base_dir: Path | None = None) -> VllmEndpointConfig:
    prefix = f"vllm.endpoints[{index}]"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{prefix} must be a mapping")
    _check_unknown_keys(raw, _ALLOWED_ENDPOINT_KEYS, prefix)
    if "url" not in raw:
        raise ConfigError(f"{prefix}.url is required")
    url = _validate_url(raw["url"], f"{prefix}.url")
    name = _validate_optional_str(raw.get("name"), f"{prefix}.name")
    raw_log = _validate_optional_str(raw.get("log_path"), f"{prefix}.log_path")
    log_path = _expand_path(raw_log, home, base_dir) if raw_log is not None else None
    raw_key = raw.get("api_key")
    if raw_key is not None and (not isinstance(raw_key, str) or not raw_key.strip()):
        raise ConfigError(f"{prefix}.api_key must be a non-empty string")
    api_key_env = _validate_optional_str(raw.get("api_key_env"), f"{prefix}.api_key_env")
    gpu_index = _validate_optional_int(raw.get("gpu_index"), f"{prefix}.gpu_index", minimum=0)
    match_pid = _validate_optional_int(raw.get("match_pid"), f"{prefix}.match_pid", minimum=1)
    match_model = _validate_optional_str(raw.get("match_model"), f"{prefix}.match_model")
    raw_timeout = raw.get("timeout_s")
    timeout_s = 2.0 if raw_timeout is None else _validate_timeout(raw_timeout, f"{prefix}.timeout_s")
    return VllmEndpointConfig(
        url=url,
        name=name,
        log_path=log_path,
        api_key=raw_key,
        api_key_env=api_key_env,
        gpu_index=gpu_index,
        match_pid=match_pid,
        match_model=match_model,
        timeout_s=timeout_s,
    )


def _parse_vllm(raw: Any, home: Path | None, base_dir: Path | None = None) -> VllmConfig:
    if raw is None:
        return VllmConfig()
    if not isinstance(raw, Mapping):
        raise ConfigError("vllm section must be a mapping")
    _check_unknown_keys(raw, _ALLOWED_VLLM_KEYS, "vllm")

    raw_auto = raw.get("auto_detect", True)
    if not isinstance(raw_auto, bool):
        raise ConfigError("vllm.auto_detect must be a boolean")

    raw_logs = raw.get("log_paths")
    if raw_logs is None:
        log_paths: list[str] = []
    elif not isinstance(raw_logs, list):
        raise ConfigError("vllm.log_paths must be a list of file path strings")
    else:
        log_paths = []
        for idx, entry in enumerate(raw_logs):
            cleaned = _validate_optional_str(entry, f"vllm.log_paths[{idx}]")
            if cleaned is None:
                raise ConfigError(f"vllm.log_paths[{idx}] must be a non-empty string")
            log_paths.append(_expand_path(cleaned, home, base_dir))

    raw_endpoints = raw.get("endpoints")
    if raw_endpoints is None:
        endpoints: list[VllmEndpointConfig] = []
    elif not isinstance(raw_endpoints, list):
        raise ConfigError("vllm.endpoints must be a list of endpoint mappings")
    else:
        endpoints = [_parse_endpoint(entry, idx, home, base_dir) for idx, entry in enumerate(raw_endpoints)]

    return VllmConfig(auto_detect=raw_auto, log_paths=log_paths, endpoints=endpoints)


def _load_file(path: Path, priority: str, home: Path | None) -> AppConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"Cannot read config file {path}: {error}") from error

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        # Format without quoting raw source lines so an inline api_key cannot leak in error output.
        problem = getattr(error, "problem", None) or "syntax error"
        mark = getattr(error, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
        raise ConfigError(f"Invalid YAML in {path}{where}: {problem}") from None

    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config root in {path} must be a YAML mapping")
    _check_unknown_keys(data, _ALLOWED_ROOT_KEYS, "root")

    resolved_path = path.resolve()
    vllm = _parse_vllm(data.get("vllm"), home, resolved_path.parent)
    return AppConfig(vllm=vllm, source=str(resolved_path), source_priority=priority)


def load_config(
    cli_path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> AppConfig:
    """Load gputaskman config using 4-level priority, or return zero-config defaults when absent.

    Priority:
      1. cli_path (--config) -> fail-loud if missing/unreadable/invalid
      2. GPUTASKMAN_CONFIG env var -> fail-loud if set and missing/unreadable/invalid
      3. <cwd>/gputaskman.yaml -> load if present (fail-loud if invalid), else fall through
      4. <home>/.config/gputaskman/config.yaml -> load if present (fail-loud if invalid), else defaults
    """
    home_path = Path(home) if home is not None else None
    if cli_path is not None:
        raw_cli = os.fspath(cli_path)
        if not isinstance(raw_cli, str) or not raw_cli.strip():
            raise ConfigError("--config path must not be empty")
        return _load_file(Path(_expand_path(raw_cli.strip(), home_path)), "cli", home_path)

    env_map = os.environ if env is None else env
    raw_env = env_map.get(ENV_CONFIG_PATH)
    if raw_env is not None and raw_env.strip():
        return _load_file(Path(_expand_path(raw_env.strip(), home_path)), "env", home_path)

    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    cwd_candidate = cwd_path / CWD_CONFIG_NAME
    if cwd_candidate.exists() or cwd_candidate.is_symlink():
        return _load_file(cwd_candidate, "cwd", home_path)

    effective_home = Path.home() if home_path is None else home_path
    user_candidate = effective_home / USER_CONFIG_REL
    if user_candidate.exists() or user_candidate.is_symlink():
        return _load_file(user_candidate, "user", home_path)

    return AppConfig()
