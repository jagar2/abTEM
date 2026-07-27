"""Retry provenance edges of a finished run from its local manifest.

Uploads and edges are delivered separately, so a transient failure (server
hiccup, daemon restart) can leave a run with preserved assets but missing
relationships. ``relink`` replays every edge that is not yet ``created`` or
``exists``, using the asset ids stored in ``provenance_manifest.json`` — no
data is re-uploaded — and rewrites the manifest and ``PROVENANCE.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from abtem.dataerai._client import PreservationClient
from abtem.dataerai._config import DataeraiConfig
from abtem.dataerai._experiment import render_report

__all__ = ["relink"]

_LINKED = ("created", "exists")


def relink(path: Union[str, Path], config: Optional[DataeraiConfig] = None) -> dict:
    """Retry missing provenance edges for the run at ``path``.

    Parameters
    ----------
    path : str or Path
        Run directory, or the ``provenance_manifest.json`` inside one.
    config : DataeraiConfig, optional
        Delivery configuration; resolved from the environment by default.

    Returns
    -------
    dict
        The updated manifest (also written back to disk).
    """
    path = Path(path)
    manifest_path = path if path.is_file() else path / "provenance_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no provenance manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if config is None:
        config = DataeraiConfig.from_env()

    client = PreservationClient(config)
    try:
        for edge in manifest["edges"]:
            if edge["link_status"] in _LINKED:
                continue
            from_node = manifest["nodes"][edge["from"]]
            to_node = manifest["nodes"][edge["to"]]
            if not (from_node.get("asset_id") and to_node.get("asset_id")):
                edge["link_status"] = "skipped"
                edge["link_detail"] = "endpoint asset not uploaded"
                continue
            outcome = client.link(
                from_node["asset_id"],
                to_node["asset_id"],
                edge["type"],
                qualifiers=edge.get("qualifiers") or None,
                analysis_mode=edge.get("analysis_mode"),
                qualifier_note=edge.get("qualifier_note"),
            )
            edge["link_status"] = outcome.status
            edge["link_detail"] = outcome.detail
    finally:
        client.close()

    manifest_path.write_text(json.dumps(manifest, indent=2))
    (manifest_path.parent / "PROVENANCE.md").write_text(render_report(manifest))
    return manifest
