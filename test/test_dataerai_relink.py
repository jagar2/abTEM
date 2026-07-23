"""Tests for abtem.dataerai._relink: retry provenance edges from a manifest."""

import json
import sys
import types
from dataclasses import dataclass

import pytest

from abtem.dataerai.__main__ import main
from abtem.dataerai._config import DataeraiConfig
from abtem.dataerai._relink import relink


@dataclass
class FakeRelationship:
    id: str = "rel-1"
    type: str = "derived_from"


class FakeSdkClient:
    instances = []
    relationship_error = None

    def __init__(self, *args, **kwargs):
        self.relationships = []
        self.connected = False
        self.closed = False
        FakeSdkClient.instances.append(self)

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def create_relationship(self, from_asset_id, to_asset_id, rel_type, **kwargs):
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first")
        if FakeSdkClient.relationship_error is not None:
            raise FakeSdkClient.relationship_error
        self.relationships.append((from_asset_id, to_asset_id, rel_type, kwargs))
        return FakeRelationship(type=rel_type)


class FakeDaemonError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@pytest.fixture
def fake_sdk(monkeypatch):
    module = types.ModuleType("dataerai")
    module.DataeraiClient = FakeSdkClient
    monkeypatch.setitem(sys.modules, "dataerai", module)
    FakeSdkClient.instances = []
    FakeSdkClient.relationship_error = None
    yield module


ASSET_A = "aaaaaaaa-0000-0000-0000-00000000000a"
ASSET_B = "aaaaaaaa-0000-0000-0000-00000000000b"
ASSET_C = "aaaaaaaa-0000-0000-0000-00000000000c"


@pytest.fixture
def run_dir(tmp_path):
    manifest = {
        "run_id": "r-relink",
        "name": "relink-test",
        "status": "completed",
        "started": "2026-07-23T00:00:00+00:00",
        "finished": "2026-07-23T00:00:10+00:00",
        "config": {
            "server": "https://beta.example.org",
            "project_id": None,
            "owner_type": "project",
            "dry_run": False,
        },
        "environment": {"python": "3.12", "packages": {"abTEM": "1.1.0"}},
        "nodes": {
            "experiment-x": {
                "role": "experiment",
                "name": "x",
                "record": {},
                "payload_path": "artifacts/experiment-x.json",
                "record_type": "simulation",
                "asset_id": ASSET_A,
                "upload_status": "uploaded",
                "upload_detail": None,
            },
            "structure-si": {
                "role": "structure",
                "name": "si",
                "record": {},
                "payload_path": "artifacts/structure-si.extxyz",
                "record_type": "sample_specimen",
                "asset_id": ASSET_B,
                "upload_status": "uploaded",
                "upload_detail": None,
            },
            "measurement-m": {
                "role": "measurement",
                "name": "m",
                "record": {},
                "payload_path": "artifacts/measurement-m.zarr.zip",
                "record_type": "imaging",
                "asset_id": ASSET_C,
                "upload_status": "uploaded",
                "upload_detail": None,
            },
            "measurement-deferred": {
                "role": "measurement",
                "name": "deferred",
                "record": {},
                "payload_path": "gone.zarr.zip",
                "record_type": "imaging",
                "asset_id": None,
                "upload_status": "skipped",
                "upload_detail": "payload missing",
            },
        },
        "edges": [
            {
                "from": "measurement-m",
                "to": "experiment-x",
                "type": "acquired_with",
                "qualifiers": {},
                "analysis_mode": None,
                "qualifier_note": None,
                "link_status": "failed",
                "link_detail": "Daemon disconnected unexpectedly",
            },
            {
                "from": "measurement-m",
                "to": "structure-si",
                "type": "derived_from",
                "qualifiers": {"role": "origin"},
                "analysis_mode": None,
                "qualifier_note": None,
                "link_status": "created",
                "link_detail": None,
            },
            {
                "from": "measurement-deferred",
                "to": "experiment-x",
                "type": "acquired_with",
                "qualifiers": {},
                "analysis_mode": None,
                "qualifier_note": None,
                "link_status": "failed",
                "link_detail": "[Errno 32] Broken pipe",
            },
        ],
    }
    (tmp_path / "provenance_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class TestRelink:
    def test_retries_only_unlinked_edges(self, run_dir, fake_sdk):
        result = relink(run_dir, config=DataeraiConfig(dry_run=False))

        sdk = FakeSdkClient.instances[0]
        # only the failed edge with both asset ids is retried
        assert sdk.relationships == [
            (ASSET_C, ASSET_A, "acquired_with", {
                "qualifiers": None,
                "analysis_mode": None,
                "qualifier_note": None,
            })
        ]
        statuses = [edge["link_status"] for edge in result["edges"]]
        assert statuses == ["created", "created", "skipped"]
        assert sdk.closed is True

    def test_manifest_and_report_rewritten(self, run_dir, fake_sdk):
        relink(run_dir, config=DataeraiConfig(dry_run=False))

        stored = json.loads((run_dir / "provenance_manifest.json").read_text())
        assert stored["edges"][0]["link_status"] == "created"
        report = (run_dir / "PROVENANCE.md").read_text()
        assert "graph TD" in report
        assert "(created)" in report

    def test_existing_edge_counts_as_success(self, run_dir, fake_sdk):
        FakeSdkClient.relationship_error = FakeDaemonError("ERR_RELATIONSHIP_EXISTS")

        result = relink(run_dir, config=DataeraiConfig(dry_run=False))

        assert result["edges"][0]["link_status"] == "exists"

    def test_dry_run_leaves_edges_pending(self, run_dir, fake_sdk):
        result = relink(run_dir, config=DataeraiConfig(dry_run=True))

        assert result["edges"][0]["link_status"] == "skipped"
        assert result["edges"][0]["link_detail"] == "dry-run"
        assert FakeSdkClient.instances == []

    def test_accepts_manifest_path(self, run_dir, fake_sdk):
        result = relink(
            run_dir / "provenance_manifest.json",
            config=DataeraiConfig(dry_run=False),
        )

        assert result["edges"][0]["link_status"] == "created"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            relink(tmp_path / "nope")


class TestCli:
    def test_relink_command(self, run_dir, fake_sdk, capsys, monkeypatch):
        monkeypatch.setenv("DATAERAI_DRY_RUN", "0")

        exit_code = main(["relink", str(run_dir)])

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "created: 2" in output
        assert "skipped: 1" in output

    def test_relink_nonzero_on_failures(self, run_dir, fake_sdk, capsys, monkeypatch):
        monkeypatch.setenv("DATAERAI_DRY_RUN", "0")
        FakeSdkClient.relationship_error = RuntimeError("still down")

        exit_code = main(["relink", str(run_dir)])

        assert exit_code == 1
        assert "failed: 1" in capsys.readouterr().out
