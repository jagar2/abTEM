"""End-to-end offline selftest: a real (tiny) STEM simulation through track()."""

import json

import pytest

from abtem.dataerai._selftest import selftest
from abtem.dataerai.__main__ import main


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return selftest(directory=tmp_path_factory.mktemp("selftest"))


def load_manifest(run_dir):
    return json.loads((run_dir / "provenance_manifest.json").read_text())


class TestSelftest:
    def test_run_completes(self, run_dir):
        manifest = load_manifest(run_dir)

        assert manifest["status"] == "completed"
        assert manifest["config"]["dry_run"] is True

    def test_full_dag_present(self, run_dir):
        manifest = load_manifest(run_dir)

        roles = {node["role"] for node in manifest["nodes"].values()}
        assert {
            "experiment",
            "structure",
            "potential",
            "illumination",
            "scan",
            "detector",
            "measurement",
        } <= roles

        # a real simulated signal was preserved, plus a user-style save
        measurements = [
            node
            for node in manifest["nodes"].values()
            if node["role"] == "measurement"
        ]
        assert len(measurements) >= 2

    def test_payloads_exist(self, run_dir):
        manifest = load_manifest(run_dir)

        for node in manifest["nodes"].values():
            payload = node["payload_path"]
            assert payload is not None
            path = run_dir / payload if not payload.startswith("/") else None
            if path is not None:
                assert path.exists()
            assert node["record"]["payload_sha256"]

    def test_expected_edges(self, run_dir):
        manifest = load_manifest(run_dir)
        roles = {key: node["role"] for key, node in manifest["nodes"].items()}
        typed = {
            (roles[edge["from"]], roles[edge["to"]], edge["type"])
            for edge in manifest["edges"]
        }

        assert ("potential", "structure", "derived_from") in typed
        assert ("measurement", "experiment", "acquired_with") in typed
        assert ("measurement", "potential", "derived_from") in typed
        assert ("illumination", "experiment", "config_for") in typed

    def test_report_written(self, run_dir):
        report = (run_dir / "PROVENANCE.md").read_text()

        assert "graph TD" in report
        assert "measurement" in report


class TestCli:
    def test_selftest_command(self, tmp_path, capsys):
        exit_code = main(["selftest", "--directory", str(tmp_path)])

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "completed" in output
        assert "provenance_manifest.json" in output

    def test_unknown_command_fails(self):
        with pytest.raises(SystemExit):
            main(["frobnicate"])
