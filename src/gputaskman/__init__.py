"""gputaskman - an htop for what's *inside* your GPU."""

from importlib.metadata import PackageNotFoundError, version


def _detect_version() -> str:
    for dist_name in ("gputaskman-vllm", "gputaskman"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return "unknown"


__version__ = _detect_version()
