"""The provenance DAG built during a tracked experiment.

Pure data structure: nodes (artifacts) and directed edges (relationships,
pointing from the derived artifact to its origin). Upload state is stamped
onto nodes/edges by the experiment layer; this module never talks to a
server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["ProvenanceEdge", "ProvenanceGraph", "ProvenanceNode"]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "node"


@dataclass
class ProvenanceNode:
    """One artifact of the experiment."""

    key: str
    role: str
    name: str
    record: dict
    payload_path: Optional[Path] = None
    record_type: Optional[str] = None
    asset_id: Optional[str] = None
    upload_status: str = "pending"
    upload_detail: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "name": self.name,
            "record": self.record,
            "payload_path": str(self.payload_path) if self.payload_path else None,
            "record_type": self.record_type,
            "asset_id": self.asset_id,
            "upload_status": self.upload_status,
            "upload_detail": self.upload_detail,
        }


@dataclass
class ProvenanceEdge:
    """A relationship: ``from`` (derived artifact) → ``to`` (its origin)."""

    from_key: str
    to_key: str
    type: str
    qualifiers: dict = field(default_factory=dict)
    analysis_mode: Optional[str] = None
    qualifier_note: Optional[str] = None
    link_status: str = "pending"
    link_detail: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "from": self.from_key,
            "to": self.to_key,
            "type": self.type,
            "qualifiers": self.qualifiers,
            "analysis_mode": self.analysis_mode,
            "qualifier_note": self.qualifier_note,
            "link_status": self.link_status,
            "link_detail": self.link_detail,
        }


class ProvenanceGraph:
    """Nodes and edges of one experiment, with stable slug keys."""

    def __init__(self):
        self._nodes: dict[str, ProvenanceNode] = {}
        self._edges: list[ProvenanceEdge] = []

    @property
    def nodes(self) -> dict[str, ProvenanceNode]:
        return self._nodes

    @property
    def edges(self) -> list[ProvenanceEdge]:
        return self._edges

    def add_node(
        self,
        *,
        role: str,
        name: str,
        record: dict,
        payload_path: Optional[Path] = None,
        record_type: Optional[str] = None,
    ) -> str:
        """Add an artifact node and return its unique key."""
        base = _slugify(f"{role}-{name}") if name != role else _slugify(role)
        key = base
        suffix = 2
        while key in self._nodes:
            key = f"{base}-{suffix}"
            suffix += 1

        self._nodes[key] = ProvenanceNode(
            key=key,
            role=role,
            name=name,
            record=record,
            payload_path=Path(payload_path) if payload_path else None,
            record_type=record_type,
        )
        return key

    def add_edge(
        self,
        from_key: str,
        to_key: str,
        type: str,
        *,
        qualifiers: Optional[dict] = None,
        analysis_mode: Optional[str] = None,
        qualifier_note: Optional[str] = None,
    ) -> bool:
        """Add ``from → to`` edge; return False if it already exists."""
        for key in (from_key, to_key):
            if key not in self._nodes:
                raise KeyError(f"unknown provenance node {key!r}")
        if from_key == to_key:
            raise ValueError(f"self-referential edge on {from_key!r}")

        if any(
            (edge.from_key, edge.to_key, edge.type) == (from_key, to_key, type)
            for edge in self._edges
        ):
            return False

        self._edges.append(
            ProvenanceEdge(
                from_key=from_key,
                to_key=to_key,
                type=type,
                qualifiers=dict(qualifiers or {}),
                analysis_mode=analysis_mode,
                qualifier_note=qualifier_note,
            )
        )
        return True

    def as_dict(self) -> dict:
        """JSON-safe dict of the full graph, including upload/link state."""
        return {
            "nodes": {key: node.as_dict() for key, node in self._nodes.items()},
            "edges": [edge.as_dict() for edge in self._edges],
        }

    def mermaid(self) -> str:
        """Render the DAG as a mermaid ``graph TD`` diagram."""
        lines = ["graph TD"]
        for key, node in self._nodes.items():
            lines.append(f'    {key}["{node.role}: {node.name}"]')
        for edge in self._edges:
            lines.append(f"    {edge.from_key} -- {edge.type} --> {edge.to_key}")
        return "\n".join(lines)
