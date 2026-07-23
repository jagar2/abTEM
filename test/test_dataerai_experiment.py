"""Tests for abtem.dataerai._experiment: track(), captures, auto-capture."""

import json
import sys
import types
from dataclasses import dataclass

import ase.build
import numpy as np
import pytest

import abtem
from abtem.array import ComputableList
from abtem.dataerai import capture, current_experiment, track
from abtem.dataerai._config import DataeraiConfig


@pytest.fixture
def atoms():
    return ase.build.bulk("Si", cubic=True) * (2, 2, 2)


@pytest.fixture
def images():
    return abtem.Images(
        np.ones((8, 8), dtype=np.float32), sampling=0.1, metadata={"energy": 100e3}
    )


DRY = DataeraiConfig(dry_run=True)


def read_manifest(directory, run_id):
    path = directory / run_id / "provenance_manifest.json"
    assert path.exists()
    return json.loads(path.read_text())


def nodes_with_role(manifest, role):
    return {
        key: node
        for key, node in manifest["nodes"].items()
        if node["role"] == role
    }


def has_edge(manifest, from_role, to_role, rel_type):
    roles = {key: node["role"] for key, node in manifest["nodes"].items()}
    return any(
        roles[edge["from"]] == from_role
        and roles[edge["to"]] == to_role
        and edge["type"] == rel_type
        for edge in manifest["edges"]
    )


class TestTrackLifecycle:
    def test_manifest_and_report_written(self, tmp_path, atoms):
        with track(
            name="lifecycle", directory=tmp_path, config=DRY, run_id="r1"
        ) as experiment:
            experiment.capture_structure(atoms)

        manifest = read_manifest(tmp_path, "r1")
        assert manifest["run_id"] == "r1"
        assert manifest["name"] == "lifecycle"
        assert manifest["status"] == "completed"
        assert manifest["environment"]["packages"]["abTEM"] == abtem.__version__
        assert manifest["config"]["dry_run"] is True
        assert "token" not in json.dumps(manifest)

        report = (tmp_path / "r1" / "PROVENANCE.md").read_text()
        assert "graph TD" in report

    def test_experiment_node_created(self, tmp_path):
        with track(name="e", directory=tmp_path, config=DRY, run_id="r1"):
            pass

        manifest = read_manifest(tmp_path, "r1")
        experiments = nodes_with_role(manifest, "experiment")
        assert len(experiments) == 1
        (node,) = experiments.values()
        assert node["record_type"] == "simulation"
        assert node["record"]["environment"]["python"]
        # experiment record payload exists and embeds the run description
        payload = json.loads((tmp_path / "r1" / node["payload_path"]).read_text())
        assert payload["run_id"] == "r1"

    def test_current_experiment_scoped(self, tmp_path):
        assert current_experiment() is None
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            assert current_experiment() is experiment
        assert current_experiment() is None

    def test_nested_track_raises(self, tmp_path):
        with track(directory=tmp_path, config=DRY, run_id="r1"):
            with pytest.raises(RuntimeError, match="already active"):
                with track(directory=tmp_path, config=DRY, run_id="r2"):
                    pass

    def test_exception_marks_failed_and_restores(self, tmp_path, atoms):
        original = ComputableList.to_zarr

        with pytest.raises(ValueError, match="boom"):
            with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
                experiment.capture_structure(atoms)
                raise ValueError("boom")

        manifest = read_manifest(tmp_path, "r1")
        assert manifest["status"] == "failed"
        assert current_experiment() is None
        assert ComputableList.to_zarr is original

    def test_unique_run_ids_generated(self, tmp_path):
        with track(directory=tmp_path, config=DRY) as first:
            pass
        with track(directory=tmp_path, config=DRY) as second:
            pass

        assert first.run_id != second.run_id
        assert (tmp_path / first.run_id).is_dir()
        assert (tmp_path / second.run_id).is_dir()


class TestCaptures:
    def test_structure(self, tmp_path, atoms):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            key = experiment.capture_structure(atoms)

        manifest = read_manifest(tmp_path, "r1")
        node = manifest["nodes"][key]
        assert node["role"] == "structure"
        assert node["record_type"] == "sample_specimen"
        assert node["record"]["structure"]["formula"] == "Si64"
        payload = tmp_path / "r1" / node["payload_path"]
        assert payload.exists()
        assert ase.io.read(payload).get_chemical_formula() == "Si64"
        assert node["record"]["payload_sha256"]

    def test_potential_links_structure(self, tmp_path, atoms):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            experiment.capture_structure(atoms)
            key = experiment.capture_potential(
                abtem.Potential(atoms, sampling=0.2, slice_thickness=2)
            )

        manifest = read_manifest(tmp_path, "r1")
        node = manifest["nodes"][key]
        assert node["role"] == "potential"
        assert has_edge(manifest, "potential", "structure", "derived_from")
        assert has_edge(manifest, "potential", "experiment", "config_for")
        payload = json.loads((tmp_path / "r1" / node["payload_path"]).read_text())
        assert payload["type"].endswith("Potential")

    def test_component_captures_link_experiment(self, tmp_path):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            experiment.capture_illumination(
                abtem.Probe(energy=100e3, semiangle_cutoff=20)
            )
            experiment.capture_scan(abtem.GridScan(start=(0, 0), end=(2, 2), gpts=4))
            experiment.capture_detector(abtem.AnnularDetector(inner=60, outer=180))

        manifest = read_manifest(tmp_path, "r1")
        for role in ("illumination", "scan", "detector"):
            assert len(nodes_with_role(manifest, role)) == 1
            assert has_edge(manifest, role, "experiment", "config_for")

    def test_measurement_full_edge_set(self, tmp_path, atoms, images):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            experiment.capture_structure(atoms)
            experiment.capture_potential(abtem.Potential(atoms, sampling=0.2))
            experiment.capture_illumination(abtem.Probe(energy=100e3))
            experiment.capture_scan(abtem.GridScan(start=(0, 0), end=(2, 2), gpts=4))
            experiment.capture_detector(abtem.AnnularDetector(inner=60))
            key = experiment.capture_measurement(images, name="haadf")

        manifest = read_manifest(tmp_path, "r1")
        node = manifest["nodes"][key]
        assert node["role"] == "measurement"
        assert node["record_type"] == "imaging"
        assert node["record"]["measurement"]["type"] == "Images"
        assert node["record"]["payload_sha256"]
        assert (tmp_path / "r1" / node["payload_path"]).exists()

        assert has_edge(manifest, "measurement", "experiment", "acquired_with")
        assert has_edge(manifest, "measurement", "potential", "derived_from")
        assert has_edge(manifest, "measurement", "illumination", "acquired_with")
        assert has_edge(manifest, "measurement", "scan", "acquired_with")
        assert has_edge(manifest, "measurement", "detector", "acquired_with")

    def test_measurement_roundtrips_from_zarr(self, tmp_path, images):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            key = experiment.capture_measurement(images, name="m")

        manifest = read_manifest(tmp_path, "r1")
        restored = abtem.from_zarr(
            str(tmp_path / "r1" / manifest["nodes"][key]["payload_path"])
        ).compute()
        assert np.allclose(restored.array, images.array)

    def test_measurement_without_components(self, tmp_path, images):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            experiment.capture_measurement(images, name="m")

        manifest = read_manifest(tmp_path, "r1")
        assert has_edge(manifest, "measurement", "experiment", "acquired_with")

    def test_diffraction_record_type(self, tmp_path):
        patterns = abtem.DiffractionPatterns(
            np.ones((8, 8), dtype=np.float32), sampling=0.05
        )

        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            key = experiment.capture_measurement(patterns, name="cbed")

        manifest = read_manifest(tmp_path, "r1")
        assert manifest["nodes"][key]["record_type"] == "diffraction_scattering"

    def test_capture_file(self, tmp_path):
        artifact = tmp_path / "notes.txt"
        artifact.write_text("observations")

        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            key = experiment.capture_file(artifact, role="analysis")

        manifest = read_manifest(tmp_path, "r1")
        node = manifest["nodes"][key]
        assert node["role"] == "analysis"
        assert node["record"]["payload_sha256"]


class TestAutoCapture:
    def test_to_zarr_captured(self, tmp_path, images):
        url = tmp_path / "saved.zarr.zip"

        with track(directory=tmp_path, config=DRY, run_id="r1"):
            images.to_zarr(str(url))

        manifest = read_manifest(tmp_path, "r1")
        measurements = nodes_with_role(manifest, "measurement")
        assert len(measurements) == 1
        (node,) = measurements.values()
        assert node["payload_path"] == str(url)
        assert has_edge(manifest, "measurement", "experiment", "acquired_with")

    def test_explicit_capture_not_doubled(self, tmp_path, images):
        with track(directory=tmp_path, config=DRY, run_id="r1") as experiment:
            experiment.capture_measurement(images, name="m")

        manifest = read_manifest(tmp_path, "r1")
        assert len(nodes_with_role(manifest, "measurement")) == 1

    def test_autocapture_disabled(self, tmp_path, images):
        with track(directory=tmp_path, config=DRY, run_id="r1", autocapture=False):
            images.to_zarr(str(tmp_path / "saved.zarr.zip"))

        manifest = read_manifest(tmp_path, "r1")
        assert len(nodes_with_role(manifest, "measurement")) == 0

    def test_no_capture_outside_track(self, tmp_path, images):
        with track(directory=tmp_path, config=DRY, run_id="r1"):
            pass

        images.to_zarr(str(tmp_path / "outside.zarr.zip"))

        manifest = read_manifest(tmp_path, "r1")
        assert len(nodes_with_role(manifest, "measurement")) == 0

    def test_computable_list_items_all_captured(self, tmp_path, images):
        pair = ComputableList([images, images.copy()])

        with track(directory=tmp_path, config=DRY, run_id="r1"):
            pair.to_zarr(str(tmp_path / "pair.zarr.zip"))

        manifest = read_manifest(tmp_path, "r1")
        assert len(nodes_with_role(manifest, "measurement")) == 2


class TestDispatch:
    def test_capture_requires_active_experiment(self, atoms):
        with pytest.raises(RuntimeError, match="track"):
            capture(atoms)

    def test_capture_dispatches_by_type(self, tmp_path, atoms, images):
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
        scan = abtem.GridScan(start=(0, 0), end=(2, 2), gpts=4)
        detector = abtem.AnnularDetector(inner=60)
        potential = abtem.Potential(atoms, sampling=0.2)
        phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=1)

        with track(directory=tmp_path, config=DRY, run_id="r1"):
            roles = {
                capture(atoms): "structure",
                capture(potential): "potential",
                capture(probe): "illumination",
                capture(scan): "scan",
                capture(detector): "detector",
                capture(phonons): "sample",
                capture(images, name="m"): "measurement",
            }

        manifest = read_manifest(tmp_path, "r1")
        for key, role in roles.items():
            assert manifest["nodes"][key]["role"] == role

    def test_capture_rejects_unknown(self, tmp_path):
        with track(directory=tmp_path, config=DRY, run_id="r1"):
            with pytest.raises(TypeError, match="cannot capture"):
                capture(3.14)


@dataclass
class FakeUploadResult:
    asset_id: str


class FakeSdkClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.uploads = []
        self.relationships = []
        self._counter = 0
        FakeSdkClient.instances.append(self)

    def auth_status(self):
        return types.SimpleNamespace(user_id="u-1")

    def upload(self, local_path, **kwargs):
        self._counter += 1
        self.uploads.append((local_path, kwargs))
        return FakeUploadResult(asset_id=f"00000000-0000-0000-0000-{self._counter:012d}")

    def create_relationship(self, from_asset_id, to_asset_id, rel_type, **kwargs):
        self.relationships.append((from_asset_id, to_asset_id, rel_type, kwargs))
        return types.SimpleNamespace(id="rel", type=rel_type)


@pytest.fixture
def fake_sdk(monkeypatch):
    module = types.ModuleType("dataerai")
    module.DataeraiClient = FakeSdkClient
    monkeypatch.setitem(sys.modules, "dataerai", module)
    FakeSdkClient.instances = []
    yield module


class TestLiveFinalize:
    def test_all_nodes_uploaded_and_linked(self, tmp_path, atoms, images, fake_sdk):
        config = DataeraiConfig(dry_run=False)

        with track(
            name="live", directory=tmp_path, config=config, run_id="r1"
        ) as experiment:
            experiment.capture_structure(atoms)
            experiment.capture_potential(abtem.Potential(atoms, sampling=0.2))
            experiment.capture_measurement(images, name="haadf")

        manifest = read_manifest(tmp_path, "r1")
        for node in manifest["nodes"].values():
            assert node["upload_status"] == "uploaded"
            assert node["asset_id"]
        for edge in manifest["edges"]:
            assert edge["link_status"] == "created"

        sdk = FakeSdkClient.instances[0]
        assert len(sdk.uploads) == len(manifest["nodes"])
        assert len(sdk.relationships) == len(manifest["edges"])
        # relationships reference uploaded asset ids, derived -> origin
        uploaded_ids = {
            node["asset_id"] for node in manifest["nodes"].values()
        }
        for from_id, to_id, rel_type, _ in sdk.relationships:
            assert from_id in uploaded_ids
            assert to_id in uploaded_ids

    def test_upload_carries_run_tags_and_metadata(self, tmp_path, atoms, fake_sdk):
        config = DataeraiConfig(dry_run=False)

        with track(directory=tmp_path, config=config, run_id="r7") as experiment:
            experiment.capture_structure(atoms)

        sdk = FakeSdkClient.instances[0]
        for _, kwargs in sdk.uploads:
            assert "abtem-dataerai" in kwargs["tags"]
            assert "abtem-run:r7" in kwargs["tags"]
            assert kwargs["metadata"]["run_id"] == "r7"
            assert kwargs["metadata"]["abtem_version"] == abtem.__version__

    def test_deferred_write_skipped(self, tmp_path, images, fake_sdk):
        config = DataeraiConfig(dry_run=False)

        with track(directory=tmp_path, config=config, run_id="r1"):
            images.to_zarr(str(tmp_path / "deferred.zarr.zip"), compute=False)

        manifest = read_manifest(tmp_path, "r1")
        (node,) = nodes_with_role(manifest, "measurement").values()
        assert node["upload_status"] == "skipped"
        assert "missing" in node["upload_detail"]
        # edges touching the unsaved node are skipped, not failed
        edge_statuses = {
            edge["link_status"]
            for edge in manifest["edges"]
            if edge["from"] == node_key(manifest, node)
        }
        assert edge_statuses <= {"skipped"}


def node_key(manifest, node):
    for key, candidate in manifest["nodes"].items():
        if candidate is node:
            return key
    raise AssertionError("node not in manifest")
