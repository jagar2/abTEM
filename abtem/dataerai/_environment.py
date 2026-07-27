"""Software-environment snapshot recorded with every experiment."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Sequence

__all__ = ["environment_snapshot"]

_DEFAULT_PACKAGES = (
    "numpy",
    "scipy",
    "dask",
    "distributed",
    "zarr",
    "ase",
    "numba",
    "matplotlib",
    "dataerai-sdk",
)


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_snapshot(packages: Optional[Sequence[str]] = None) -> dict:
    """Capture interpreter, platform, and package versions as a JSON-safe dict.

    Parameters
    ----------
    packages : sequence of str, optional
        Distribution names to record in addition to abTEM itself. Packages
        that are not installed are recorded as ``None``.
    """
    from abtem import __version__ as abtem_version

    if packages is None:
        packages = _DEFAULT_PACKAGES

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "packages": {
            "abTEM": abtem_version,
            **{name: _package_version(name) for name in packages},
        },
    }
