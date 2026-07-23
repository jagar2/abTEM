"""Delivery layer: upload artifacts and create relationship edges.

Wraps the optional ``dataerai`` SDK (distribution ``dataerai-sdk``), which
proxies everything through the local ``dataerai-transfer`` daemon holding the
OAuth credentials. Every operation degrades explicitly instead of raising:

- uploads: SDK → ``skipped`` (dry-run or SDK missing) / ``failed``
- relationship edges: SDK ``create_relationship`` → direct REST
  (``POST /api/assets/{from}/relationships/`` with a bearer token) →
  ``skipped``

Outcomes are plain records so the experiment layer can stamp them into the
provenance manifest verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from urllib.request import Request, urlopen

from abtem.dataerai._config import DataeraiConfig

__all__ = ["LinkOutcome", "PreservationClient", "UploadOutcome"]


@dataclass(frozen=True)
class UploadOutcome:
    status: str  # "uploaded" | "skipped" | "failed"
    asset_id: Optional[str] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class LinkOutcome:
    status: str  # "created" | "exists" | "skipped" | "failed"
    detail: Optional[str] = None


def _post_json(
    url: str, payload: dict, token: str, timeout: float = 30.0
) -> tuple[int, dict]:
    """POST JSON with a bearer token; return (status, decoded body)."""
    import urllib.error

    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, {}

    try:
        decoded = json.loads(body) if body else {}
    except json.JSONDecodeError:
        decoded = {}
    return status, decoded


class PreservationClient:
    """Uploads and provenance links for one experiment, per the config."""

    def __init__(self, config: DataeraiConfig):
        self._config = config
        self._sdk_client: Any = None
        self._owner: Optional[tuple[str, str]] = None

    def _get_sdk_client(self):
        """Import the SDK and build a client once; None if unavailable."""
        if self._sdk_client is None:
            try:
                from dataerai import DataeraiClient
            except ImportError:
                self._sdk_client = False
            else:
                self._sdk_client = DataeraiClient()
        return self._sdk_client or None

    def _resolve_owner(self, sdk_client) -> tuple[str, str]:
        """Owner for new assets: the configured project, else the daemon user."""
        if self._owner is None:
            if self._config.project_id:
                self._owner = ("project", self._config.project_id)
            else:
                status = sdk_client.auth_status()
                self._owner = ("user", status.user_id)
        return self._owner

    def upload(
        self,
        path: Union[str, Path],
        *,
        title: str,
        record_type: Optional[str] = None,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
        description: Optional[str] = None,
    ) -> UploadOutcome:
        """Preserve one artifact file as a Dataerai asset."""
        if self._config.dry_run:
            return UploadOutcome(status="skipped", detail="dry-run")

        sdk_client = self._get_sdk_client()
        if sdk_client is None:
            return UploadOutcome(status="skipped", detail="dataerai SDK not installed")

        try:
            owner_type, owner_id = self._resolve_owner(sdk_client)
            result = sdk_client.upload(
                str(path),
                title=title,
                owner_type=owner_type,
                owner_id=owner_id,
                record_type=record_type,
                tags=tags,
                metadata=metadata,
                description=description,
            )
        except Exception as error:
            return UploadOutcome(status="failed", detail=str(error))

        return UploadOutcome(status="uploaded", asset_id=result.asset_id)

    def link(
        self,
        from_asset_id: str,
        to_asset_id: str,
        rel_type: str,
        *,
        qualifiers: Optional[dict] = None,
        analysis_mode: Optional[str] = None,
        qualifier_note: Optional[str] = None,
    ) -> LinkOutcome:
        """Create a provenance edge ``from (derived) → to (origin)``."""
        if self._config.dry_run:
            return LinkOutcome(status="skipped", detail="dry-run")

        sdk_client = self._get_sdk_client()
        if sdk_client is not None and hasattr(sdk_client, "create_relationship"):
            return self._link_via_sdk(
                sdk_client,
                from_asset_id,
                to_asset_id,
                rel_type,
                qualifiers=qualifiers,
                analysis_mode=analysis_mode,
                qualifier_note=qualifier_note,
            )

        if self._config.token:
            return self._link_via_rest(
                from_asset_id,
                to_asset_id,
                rel_type,
                qualifiers=qualifiers,
                analysis_mode=analysis_mode,
                qualifier_note=qualifier_note,
            )

        return LinkOutcome(
            status="skipped",
            detail="no SDK create_relationship and no bearer token",
        )

    def _link_via_sdk(
        self,
        sdk_client,
        from_asset_id,
        to_asset_id,
        rel_type,
        *,
        qualifiers,
        analysis_mode,
        qualifier_note,
    ) -> LinkOutcome:
        try:
            sdk_client.create_relationship(
                from_asset_id,
                to_asset_id,
                rel_type,
                qualifiers=qualifiers,
                analysis_mode=analysis_mode,
                qualifier_note=qualifier_note,
            )
        except Exception as error:
            if "ERR_RELATIONSHIP_EXISTS" in (getattr(error, "code", "") or str(error)):
                return LinkOutcome(status="exists")
            return LinkOutcome(status="failed", detail=str(error))
        return LinkOutcome(status="created")

    def _link_via_rest(
        self,
        from_asset_id,
        to_asset_id,
        rel_type,
        *,
        qualifiers,
        analysis_mode,
        qualifier_note,
    ) -> LinkOutcome:
        url = f"{self._config.server}/api/assets/{from_asset_id}/relationships/"
        payload = {"to_asset_id": to_asset_id, "type": rel_type}
        if qualifiers:
            payload["qualifiers"] = qualifiers
        if analysis_mode:
            payload["analysis_mode"] = analysis_mode
        if qualifier_note:
            payload["qualifier_note"] = qualifier_note

        try:
            status, body = _post_json(url, payload, self._config.token)
        except Exception as error:
            return LinkOutcome(status="failed", detail=str(error))

        if status == 201:
            return LinkOutcome(status="created")
        if status == 409:
            return LinkOutcome(status="exists")
        return LinkOutcome(
            status="failed", detail=f"HTTP {status}: {body.get('detail', '')}"
        )
