"""Tests for abtem.dataerai._provenance: the experiment DAG."""

import json

import pytest

from abtem.dataerai._provenance import ProvenanceGraph


@pytest.fixture
def graph():
    return ProvenanceGraph()


class TestNodes:
    def test_add_node_returns_key(self, graph):
        key = graph.add_node(role="structure", name="Si bulk", record={"natoms": 8})

        assert key in graph.nodes
        node = graph.nodes[key]
        assert node.role == "structure"
        assert node.name == "Si bulk"
        assert node.record == {"natoms": 8}

    def test_duplicate_names_get_unique_keys(self, graph):
        first = graph.add_node(role="measurement", name="haadf", record={})
        second = graph.add_node(role="measurement", name="haadf", record={})

        assert first != second
        assert len(graph.nodes) == 2

    def test_node_key_is_slug(self, graph):
        key = graph.add_node(role="structure", name="SrTiO3 / vacuum slab!", record={})

        assert " " not in key and "/" not in key and "!" not in key

    def test_payload_and_record_type(self, graph, tmp_path):
        payload = tmp_path / "structure.extxyz"
        payload.write_text("2\n\n")

        key = graph.add_node(
            role="structure",
            name="s",
            record={},
            payload_path=payload,
            record_type="sample_specimen",
        )

        node = graph.nodes[key]
        assert node.payload_path == payload
        assert node.record_type == "sample_specimen"


class TestEdges:
    def test_add_edge(self, graph):
        source = graph.add_node(role="potential", name="p", record={})
        target = graph.add_node(role="structure", name="s", record={})

        created = graph.add_edge(source, target, "derived_from")

        assert created is True
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert (edge.from_key, edge.to_key, edge.type) == (
            source,
            target,
            "derived_from",
        )

    def test_edge_with_qualifiers(self, graph):
        source = graph.add_node(role="measurement", name="m", record={})
        target = graph.add_node(role="experiment", name="e", record={})

        graph.add_edge(
            source,
            target,
            "acquired_with",
            qualifiers={"role": "illumination"},
            analysis_mode="non_destructive",
        )

        edge = graph.edges[0]
        assert edge.qualifiers == {"role": "illumination"}
        assert edge.analysis_mode == "non_destructive"

    def test_unknown_node_raises(self, graph):
        node = graph.add_node(role="structure", name="s", record={})

        with pytest.raises(KeyError):
            graph.add_edge(node, "missing", "derived_from")

    def test_self_edge_raises(self, graph):
        node = graph.add_node(role="structure", name="s", record={})

        with pytest.raises(ValueError):
            graph.add_edge(node, node, "derived_from")

    def test_duplicate_edge_ignored(self, graph):
        source = graph.add_node(role="a", name="a", record={})
        target = graph.add_node(role="b", name="b", record={})

        assert graph.add_edge(source, target, "derived_from") is True
        assert graph.add_edge(source, target, "derived_from") is False
        assert len(graph.edges) == 1


class TestExport:
    def test_as_dict_is_json_serializable(self, graph, tmp_path):
        source = graph.add_node(
            role="measurement",
            name="m",
            record={"shape": [4, 4]},
            payload_path=tmp_path / "m.zarr.zip",
        )
        target = graph.add_node(role="structure", name="s", record={})
        graph.add_edge(source, target, "derived_from")

        result = graph.as_dict()

        blob = json.dumps(result)
        assert "derived_from" in blob
        assert set(result["nodes"]) == {source, target}
        assert result["edges"][0]["from"] == source

    def test_upload_state_reflected_in_dict(self, graph):
        key = graph.add_node(role="structure", name="s", record={})
        graph.nodes[key].asset_id = "0f0e0d0c-1111-2222-3333-444455556666"
        graph.nodes[key].upload_status = "uploaded"

        result = graph.as_dict()

        assert (
            result["nodes"][key]["asset_id"] == "0f0e0d0c-1111-2222-3333-444455556666"
        )
        assert result["nodes"][key]["upload_status"] == "uploaded"

    def test_mermaid(self, graph):
        measurement = graph.add_node(role="measurement", name="haadf", record={})
        experiment = graph.add_node(role="experiment", name="run", record={})
        graph.add_edge(measurement, experiment, "acquired_with")

        diagram = graph.mermaid()

        assert diagram.startswith("graph TD")
        assert measurement in diagram
        assert "acquired_with" in diagram
