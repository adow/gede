# coding=utf-8

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_version_from_pyproject() -> str:
    """Best-effort fallback for local source runs without installed metadata."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        content = pyproject_path.read_text(encoding="utf-8")
    except Exception:
        return "unknown"

    # Read [project].version from pyproject.toml in a dependency-free way.
    in_project_section = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project_section = line == "[project]"
            continue
        if in_project_section and line.startswith("version"):
            match = re.match(r'version\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                return match.group(1)

    return "unknown"


def get_app_version() -> str:
    """Return app version with pyproject as source of truth in local repo."""
    local_version = _read_version_from_pyproject()
    if local_version != "unknown":
        return local_version

    try:
        return version("gede")
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"
