"""Configuration and credential discovery for the Dataerai integration.

Nothing in this module talks to a server; it only resolves *how* the
integration should behave (server, ownership, dry-run) and *which* credentials
are available, in this order: explicit environment variable, macOS Keychain,
credentials file. All lookups degrade to ``None`` rather than raising.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

__all__ = ["DEFAULT_SERVER", "DataeraiConfig", "discover_token"]

DEFAULT_SERVER = "https://beta.dataerai.com"

_VALID_OWNER_TYPES = ("project", "user")
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _sdk_available() -> bool:
    """Whether the ``dataerai`` SDK (distribution ``dataerai-sdk``) is importable."""
    try:
        return importlib.util.find_spec("dataerai") is not None
    except (ImportError, ValueError):
        return False


def _keychain_token() -> Optional[str]:
    """Read the Dataerai OAuth token from the macOS Keychain, if present."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "dataerai", "-a", "auth", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token or None


def _default_credentials_path() -> Path:
    if sys.platform == "darwin":
        return (
            Path.home() / "Library" / "Application Support" / "dataerai" / "credentials"
        )
    return Path.home() / ".config" / "dataerai" / "credentials"


def _credentials_file_token(path: Optional[Path] = None) -> Optional[str]:
    """Read a token from the Dataerai credentials file (JSON or raw)."""
    if path is None:
        path = _default_credentials_path()
    try:
        content = Path(path).read_text().strip()
    except OSError:
        return None
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(payload, dict):
        for key in ("access_token", "token"):
            token = payload.get(key)
            if isinstance(token, str) and token:
                return token
        return None
    return content


def discover_token(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Find a Dataerai bearer token: env var, then Keychain, then file."""
    import os

    if environ is None:
        environ = os.environ

    token = environ.get("DATAERAI_TOKEN")
    if token:
        return token

    token = _keychain_token()
    if token:
        return token

    credentials_file = environ.get("DATAERAI_CREDENTIALS_FILE")
    path = Path(credentials_file) if credentials_file else None
    return _credentials_file_token(path)


@dataclass(frozen=True)
class DataeraiConfig:
    """Resolved settings for one tracked experiment.

    Parameters
    ----------
    server : str
        Base URL of the Dataerai server (no trailing slash).
    project_id : str, optional
        UUID of the project that owns created assets.
    owner_type : str
        Asset ownership: ``"project"`` or ``"user"``.
    dry_run : bool
        If True, no network calls are made; provenance is only recorded
        locally.
    token : str, optional
        OAuth bearer token for direct REST calls. Excluded from ``repr`` so
        it cannot leak into logs.
    """

    server: str = DEFAULT_SERVER
    project_id: Optional[str] = None
    owner_type: str = "project"
    dry_run: bool = True
    token: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.owner_type not in _VALID_OWNER_TYPES:
            raise ValueError(
                f"owner_type must be one of {_VALID_OWNER_TYPES}, "
                f"got {self.owner_type!r}"
            )
        object.__setattr__(self, "server", self.server.rstrip("/"))

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "DataeraiConfig":
        """Build a configuration from environment variables.

        Without an explicit ``DATAERAI_DRY_RUN`` the mode is automatic:
        live when either a token or the SDK is available, dry-run otherwise.
        """
        import os

        if environ is None:
            environ = os.environ

        token = discover_token(environ)

        dry_run_env = environ.get("DATAERAI_DRY_RUN", "").strip().lower()
        if dry_run_env in _TRUTHY:
            dry_run = True
        elif dry_run_env in _FALSY:
            dry_run = False
        else:
            dry_run = token is None and not _sdk_available()

        return cls(
            server=environ.get("DATAERAI_SERVER", DEFAULT_SERVER),
            project_id=environ.get("DATAERAI_PROJECT_ID") or None,
            owner_type=environ.get("DATAERAI_OWNER_TYPE", "project"),
            dry_run=dry_run,
            token=token,
        )
