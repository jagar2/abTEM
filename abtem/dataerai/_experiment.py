"""Tracked experiments: capture abTEM artifacts into a provenance run.

:func:`track` opens an experiment context. Within it, artifacts are captured
explicitly (``capture_structure``, ``capture_potential``, …, or the
type-dispatching :func:`capture`), and any ``to_zarr`` save of an abTEM
array object is captured automatically. On exit the run is finalized:
payloads are uploaded as Dataerai assets, provenance edges are created
(derived → origin), and a local manifest (``provenance_manifest.json``) plus
a human-readable ``PROVENANCE.md`` are always written — whatever parts of
the delivery degrade.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from abtem.dataerai._client import PreservationClient
from abtem.dataerai._config import DataeraiConfig
from abtem.dataerai._environment import environment_snapshot
from abtem.dataerai._provenance import ProvenanceGraph, render_mermaid
from abtem.dataerai._serialize import (
    component_params,
    describe_measurement,
    describe_structure,
    file_sha256,
    json_safe,
    tree_sha256,
    write_structure,
)

__all__ = ["Experiment", "capture", "current_experiment", "render_report", "track"]

logger = logging.getLogger("abtem.dataerai")

#: Dataerai asset record types per measurement class (default: dataset).
_MEASUREMENT_RECORD_TYPES = {
    "Images": "imaging",
    "DiffractionPatterns": "diffraction_scattering",
    "IndexedDiffractionPatterns": "diffraction_scattering",
    "PolarMeasurements": "diffraction_scattering",
}

_ACTIVE: Optional["Experiment"] = None
_ORIGINAL_TO_ZARR = None


def current_experiment() -> Optional["Experiment"]:
    """The experiment opened by the enclosing :func:`track`, if any."""
    return _ACTIVE


class Experiment:
    """One tracked abTEM experiment run building a provenance DAG."""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        directory: Union[str, Path] = "dataerai-runs",
        config: Optional[DataeraiConfig] = None,
        run_id: Optional[str] = None,
        autocapture: bool = True,
    ):
        self.name = name or "abtem-experiment"
        self.run_id = run_id or (
            f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        self.config = config if config is not None else DataeraiConfig.from_env()
        self.status = "running"
        self.started = datetime.now(timezone.utc).isoformat()
        self.finished: Optional[str] = None

        self.directory = Path(directory) / self.run_id
        self.artifacts_dir = self.directory / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.graph = ProvenanceGraph()
        self._client = PreservationClient(self.config)
        self._components: dict[str, Any] = {"detector": []}
        self._autocapture = autocapture
        self._suppress_autocapture = False
        self._finalized = False

        self._experiment_key = self.graph.add_node(
            role="experiment",
            name=self.name,
            record={"environment": environment_snapshot()},
            record_type="simulation",
        )

    # ------------------------------------------------------------------ #
    # captures                                                           #
    # ------------------------------------------------------------------ #

    def capture(self, obj: Any, name: Optional[str] = None, **kwargs) -> str:
        """Capture any supported object, dispatching on its type."""
        import ase

        from abtem.array import ArrayObject
        from abtem.detectors import BaseDetector
        from abtem.inelastic.phonons import BaseFrozenPhonons
        from abtem.potentials.iam import BasePotential
        from abtem.prism.s_matrix import SMatrix
        from abtem.scan import BaseScan
        from abtem.waves import PlaneWave, Probe

        if isinstance(obj, ase.Atoms):
            return self.capture_structure(obj, name=name)
        if isinstance(obj, BaseFrozenPhonons):
            return self.capture_sample(obj, name=name)
        if isinstance(obj, (Probe, PlaneWave, SMatrix)):
            return self.capture_illumination(obj, name=name)
        if isinstance(obj, BaseScan):
            return self.capture_scan(obj, name=name)
        if isinstance(obj, BaseDetector):
            return self.capture_detector(obj, name=name)
        if isinstance(obj, ArrayObject):
            return self.capture_measurement(obj, name=name, **kwargs)
        if isinstance(obj, BasePotential):
            return self.capture_potential(obj, name=name)
        if isinstance(obj, (str, Path)) and Path(obj).exists():
            return self.capture_file(obj, name=name, **kwargs)

        raise TypeError(
            f"cannot capture object of type {type(obj).__name__!r}; supported: "
            "ase.Atoms, potentials, probes/plane waves, scans, detectors, "
            "frozen phonons, abTEM array objects, and existing file paths"
        )

    def capture_structure(self, atoms, name: Optional[str] = None) -> str:
        """Preserve the atomic structure (extended XYZ + content hash)."""
        name = name or atoms.get_chemical_formula()
        key = self.graph.add_node(
            role="structure", name=name, record={}, record_type="sample_specimen"
        )
        path = write_structure(atoms, self.artifacts_dir / f"{key}.extxyz")
        node = self.graph.nodes[key]
        node.payload_path = path
        node.record = {
            "structure": describe_structure(atoms),
            "payload_sha256": file_sha256(path),
        }
        self._components["structure"] = key
        return key

    def capture_potential(self, potential, name: Optional[str] = None) -> str:
        """Preserve the potential configuration."""
        key = self._capture_component(potential, role="potential", name=name)
        structure_key = self._components.get("structure")
        if structure_key:
            self.graph.add_edge(key, structure_key, "derived_from")
        self._components["potential"] = key
        return key

    def capture_sample(self, phonons, name: Optional[str] = None) -> str:
        """Preserve a frozen-phonon / ensemble sample description."""
        key = self._capture_component(phonons, role="sample", name=name)
        structure_key = self._components.get("structure")
        if structure_key:
            self.graph.add_edge(key, structure_key, "derived_from")
        self._components["sample"] = key
        return key

    def capture_illumination(self, waves, name: Optional[str] = None) -> str:
        """Preserve the probe / plane-wave / S-matrix configuration."""
        key = self._capture_component(waves, role="illumination", name=name)
        self._components["illumination"] = key
        return key

    def capture_scan(self, scan, name: Optional[str] = None) -> str:
        """Preserve the scan configuration."""
        key = self._capture_component(scan, role="scan", name=name)
        self._components["scan"] = key
        return key

    def capture_detector(self, detector, name: Optional[str] = None) -> str:
        """Preserve a detector configuration (repeatable)."""
        key = self._capture_component(detector, role="detector", name=name)
        self._components["detector"].append(key)
        return key

    def capture_measurement(
        self, measurement, name: Optional[str] = None, description: Optional[str] = None
    ) -> str:
        """Preserve a computed measurement as a Zarr zip store."""
        name = name or type(measurement).__name__.lower()
        key = self.graph.add_node(
            role="measurement",
            name=name,
            record={},
            record_type=self._measurement_record_type(measurement),
        )
        path = self.artifacts_dir / f"{key}.zarr.zip"
        with self._suppressed_autocapture():
            measurement.to_zarr(str(path))
        node = self.graph.nodes[key]
        node.payload_path = path
        node.record = {"measurement": describe_measurement(measurement)}
        self._add_measurement_edges(key)
        return key

    def capture_file(
        self,
        path: Union[str, Path],
        *,
        role: str = "artifact",
        name: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> str:
        """Preserve an arbitrary existing file produced by the experiment."""
        path = Path(path)
        key = self.graph.add_node(
            role=role,
            name=name or path.name,
            record={"payload_sha256": tree_sha256(path)},
            payload_path=path,
            record_type=record_type,
        )
        self.graph.add_edge(key, self._experiment_key, "acquired_with")
        return key

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _capture_component(self, obj, *, role: str, name: Optional[str]) -> str:
        name = name or type(obj).__name__.lower()
        record = component_params(obj)
        key = self.graph.add_node(
            role=role, name=name, record={}, record_type="protocol_workflow"
        )
        path = self.artifacts_dir / f"{key}.json"
        path.write_text(json.dumps(json_safe(record), indent=2))
        node = self.graph.nodes[key]
        node.payload_path = path
        node.record = {"component": record, "payload_sha256": file_sha256(path)}
        self.graph.add_edge(key, self._experiment_key, "config_for")
        return key

    @staticmethod
    def _measurement_record_type(measurement) -> str:
        return _MEASUREMENT_RECORD_TYPES.get(type(measurement).__name__, "dataset")

    def _add_measurement_edges(self, key: str) -> None:
        self.graph.add_edge(key, self._experiment_key, "acquired_with")
        potential_key = self._components.get("potential")
        if potential_key:
            self.graph.add_edge(key, potential_key, "derived_from")
        for role in ("illumination", "scan"):
            component_key = self._components.get(role)
            if component_key:
                self.graph.add_edge(
                    key, component_key, "acquired_with", qualifiers={"role": role}
                )
        for detector_key in self._components["detector"]:
            self.graph.add_edge(
                key, detector_key, "acquired_with", qualifiers={"role": "detector"}
            )

    @contextmanager
    def _suppressed_autocapture(self):
        previous = self._suppress_autocapture
        self._suppress_autocapture = True
        try:
            yield
        finally:
            self._suppress_autocapture = previous

    def _register_saved(self, items, url: str) -> None:
        """Record measurements written by a user's own ``to_zarr`` call."""
        base = Path(url).name
        for suffix in (".zip", ".zarr"):
            base = base.removesuffix(suffix)
        for index, item in enumerate(items):
            name = base if len(items) == 1 else f"{base}-{index}"
            key = self.graph.add_node(
                role="measurement",
                name=name,
                record={"measurement": describe_measurement(item)},
                payload_path=Path(url),
                record_type=self._measurement_record_type(item),
            )
            self._add_measurement_edges(key)

    # ------------------------------------------------------------------ #
    # finalize                                                           #
    # ------------------------------------------------------------------ #

    def finalize(self, status: str = "completed") -> None:
        """Upload payloads, create edges, and write the local manifest."""
        if self._finalized:
            return
        self._finalized = True
        self.status = status
        self.finished = datetime.now(timezone.utc).isoformat()

        try:
            self._write_experiment_payload()
            self._hash_pending_payloads()
            self._deliver()
        finally:
            self._client.close()
            self._write_manifest()
            logger.info(
                "dataerai run %s finalized (%s): %d artifacts, %d edges -> %s",
                self.run_id,
                self.status,
                len(self.graph.nodes),
                len(self.graph.edges),
                self.directory,
            )

    def _run_description(self) -> dict:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "config": {
                "server": self.config.server,
                "project_id": self.config.project_id,
                "owner_type": self.config.owner_type,
                "dry_run": self.config.dry_run,
            },
            "environment": self.graph.nodes[self._experiment_key].record["environment"],
            "components": {
                role: keys for role, keys in self._components.items() if keys
            },
        }

    def _write_experiment_payload(self) -> None:
        node = self.graph.nodes[self._experiment_key]
        description = self._run_description()
        path = self.artifacts_dir / f"{self._experiment_key}.json"
        path.write_text(json.dumps(json_safe(description), indent=2))
        node.payload_path = path
        node.record = {
            "environment": description["environment"],
            "components": description["components"],
            "payload_sha256": file_sha256(path),
        }

    def _hash_pending_payloads(self) -> None:
        for node in self.graph.nodes.values():
            if (
                node.payload_path is not None
                and node.payload_path.exists()
                and "payload_sha256" not in node.record
            ):
                node.record["payload_sha256"] = tree_sha256(node.payload_path)

    def _deliver(self) -> None:
        for key, node in self.graph.nodes.items():
            if node.payload_path is None or not node.payload_path.exists():
                node.upload_status = "skipped"
                node.upload_detail = "payload missing (deferred or deleted write)"
                continue
            outcome = self._client.upload(
                node.payload_path,
                title=f"{self.name} / {node.role}: {node.name}",
                record_type=node.record_type,
                tags=[
                    "abtem-dataerai",
                    f"abtem-run:{self.run_id}",
                    f"abtem-role:{node.role}",
                ],
                metadata={
                    "run_id": self.run_id,
                    "role": node.role,
                    "abtem_version": _abtem_version(),
                    "record": json_safe(node.record),
                },
                description=f"abTEM experiment run {self.run_id}",
            )
            node.upload_status = outcome.status
            node.upload_detail = outcome.detail
            node.asset_id = outcome.asset_id

        for edge in self.graph.edges:
            from_node = self.graph.nodes[edge.from_key]
            to_node = self.graph.nodes[edge.to_key]
            if not (from_node.asset_id and to_node.asset_id):
                edge.link_status = "skipped"
                edge.link_detail = "endpoint asset not uploaded"
                continue
            outcome = self._client.link(
                from_node.asset_id,
                to_node.asset_id,
                edge.type,
                qualifiers=edge.qualifiers or None,
                analysis_mode=edge.analysis_mode,
                qualifier_note=edge.qualifier_note,
            )
            edge.link_status = outcome.status
            edge.link_detail = outcome.detail

    def _manifest(self) -> dict:
        manifest = {**self._run_description(), **self.graph.as_dict()}
        run_dir = str(self.directory)
        for node in manifest["nodes"].values():
            payload = node["payload_path"]
            if payload and payload.startswith(run_dir):
                node["payload_path"] = str(Path(payload).relative_to(self.directory))
        return json_safe(manifest)

    def _write_manifest(self) -> None:
        manifest = self._manifest()
        (self.directory / "provenance_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        (self.directory / "PROVENANCE.md").write_text(render_report(manifest))


def render_report(manifest: dict) -> str:
    """Render the human-readable ``PROVENANCE.md`` from a manifest dict."""
    lines = [
        f"# Provenance: {manifest['name']} ({manifest['run_id']})",
        "",
        f"- status: **{manifest['status']}**",
        f"- started: {manifest['started']}",
        f"- finished: {manifest['finished']}",
        f"- abTEM {manifest['environment']['packages'].get('abTEM')}"
        f" / python {manifest['environment']['python']}",
        f"- dry run: {manifest['config']['dry_run']}",
        "",
        "## Provenance graph",
        "",
        "```mermaid",
        render_mermaid(manifest["nodes"], manifest["edges"]),
        "```",
        "",
        "## Artifacts",
        "",
        "| key | role | name | upload | asset id |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, node in manifest["nodes"].items():
        lines.append(
            f"| {key} | {node['role']} | {node['name']} "
            f"| {node['upload_status']} | {node['asset_id'] or ''} |"
        )
    lines += ["", "## Relationships", ""]
    for edge in manifest["edges"]:
        lines.append(
            f"- `{edge['from']}` --{edge['type']}--> `{edge['to']}`"
            f" ({edge['link_status']})"
        )
    lines.append("")
    return "\n".join(lines)


def _abtem_version() -> str:
    from abtem import __version__

    return __version__


# ---------------------------------------------------------------------- #
# module-level API                                                       #
# ---------------------------------------------------------------------- #


def capture(obj: Any, name: Optional[str] = None, **kwargs) -> str:
    """Capture ``obj`` into the active experiment (see :func:`track`)."""
    experiment = current_experiment()
    if experiment is None:
        raise RuntimeError(
            "no active experiment; open one with `with abtem.dataerai.track(...)`"
        )
    return experiment.capture(obj, name=name, **kwargs)


def _patched_to_zarr(self, url, compute=True, overwrite=False, **kwargs):
    result = _ORIGINAL_TO_ZARR(
        self, url, compute=compute, overwrite=overwrite, **kwargs
    )
    experiment = _ACTIVE
    if (
        experiment is not None
        and experiment._autocapture
        and not experiment._suppress_autocapture
    ):
        try:
            experiment._register_saved(list(self), url)
        except Exception:
            logger.warning(
                "failed to auto-capture %s into dataerai run %s",
                url,
                experiment.run_id,
                exc_info=True,
            )
    return result


def _install_autocapture() -> None:
    global _ORIGINAL_TO_ZARR
    from abtem.array import ComputableList

    if _ORIGINAL_TO_ZARR is None:
        _ORIGINAL_TO_ZARR = ComputableList.to_zarr
        ComputableList.to_zarr = _patched_to_zarr


def _uninstall_autocapture() -> None:
    global _ORIGINAL_TO_ZARR
    from abtem.array import ComputableList

    if _ORIGINAL_TO_ZARR is not None:
        ComputableList.to_zarr = _ORIGINAL_TO_ZARR
        _ORIGINAL_TO_ZARR = None


@contextmanager
def track(
    name: Optional[str] = None,
    directory: Union[str, Path] = "dataerai-runs",
    *,
    config: Optional[DataeraiConfig] = None,
    run_id: Optional[str] = None,
    autocapture: bool = True,
):
    """Open a tracked experiment; finalize (upload, link, manifest) on exit.

    Parameters
    ----------
    name : str, optional
        Human-readable experiment name.
    directory : str or Path
        Parent directory for run folders (one subdirectory per run).
    config : DataeraiConfig, optional
        Delivery configuration; resolved from the environment by default.
    run_id : str, optional
        Explicit run identifier (generated when omitted).
    autocapture : bool
        Capture every ``to_zarr`` save automatically while active.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError(
            f"a Dataerai experiment is already active ({_ACTIVE.run_id})"
        )

    experiment = Experiment(
        name=name,
        directory=directory,
        config=config,
        run_id=run_id,
        autocapture=autocapture,
    )
    _ACTIVE = experiment
    _install_autocapture()
    try:
        yield experiment
    except BaseException:
        _ACTIVE = None
        _uninstall_autocapture()
        experiment.finalize(status="failed")
        raise
    else:
        _ACTIVE = None
        _uninstall_autocapture()
        experiment.finalize(status="completed")
