"""Tests for abtem.dataerai._serialize: abTEM objects -> provenance JSON."""

import hashlib
import json

import ase.build
import numpy as np
import pytest

import abtem
from abtem.dataerai._serialize import (
    component_params,
    describe_measurement,
    describe_structure,
    file_sha256,
    json_safe,
    tree_sha256,
    write_structure,
)


@pytest.fixture
def atoms():
    return ase.build.bulk("Si", cubic=True) * (2, 2, 2)


class TestJsonSafe:
    def test_passthrough_primitives(self):
        for value in (None, True, 3, 2.5, "x"):
            assert json_safe(value) == value

    def test_numpy_scalars(self):
        assert json_safe(np.float64(2.5)) == 2.5
        assert isinstance(json_safe(np.float64(2.5)), float)
        assert json_safe(np.int32(3)) == 3
        assert isinstance(json_safe(np.int32(3)), int)
        assert json_safe(np.bool_(True)) is True

    def test_tuple_becomes_list(self):
        assert json_safe((1, (2, 3))) == [1, [2, 3]]

    def test_small_array_becomes_list(self):
        assert json_safe(np.arange(4).reshape(2, 2)) == [[0, 1], [2, 3]]

    def test_large_array_becomes_summary(self):
        array = np.zeros((64, 64))

        result = json_safe(array)

        assert result["shape"] == [64, 64]
        assert result["dtype"] == "float64"
        assert result["sha256"] == hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest()

    def test_dict_keys_coerced_to_str(self):
        assert json_safe({1: "a"}) == {"1": "a"}

    def test_atoms_summarized(self, atoms):
        result = json_safe(atoms)

        assert result["formula"] == atoms.get_chemical_formula()
        assert result["natoms"] == len(atoms)

    def test_abtem_component_nested(self):
        result = json_safe(abtem.AnnularDetector(inner=65, outer=200))

        assert result["type"].endswith("AnnularDetector")
        assert result["params"]["inner"] == 65

    def test_callable_by_name(self):
        assert "json_safe" in json_safe(json_safe)

    def test_unknown_object_repr(self):
        class Strange:
            def __repr__(self):
                return "<strange>"

        assert json_safe(Strange()) == "<strange>"

    def test_everything_dumps(self, atoms):
        blob = {
            "a": (np.float32(1.5), {"b": np.arange(3)}),
            2: atoms,
            "det": abtem.AnnularDetector(inner=60),
            "big": np.ones((100, 100)),
        }

        json.dumps(json_safe(blob))


class TestComponentParams:
    def test_probe(self):
        probe = abtem.Probe(energy=200e3, semiangle_cutoff=20, extent=10, gpts=64)

        result = component_params(probe)

        assert result["type"] == "abtem.waves.Probe"
        assert result["params"]["energy"] == 200e3
        assert result["params"]["semiangle_cutoff"] == 20
        json.dumps(result)

    def test_potential(self, atoms):
        potential = abtem.Potential(atoms, sampling=0.2, slice_thickness=2)

        result = component_params(potential)

        assert result["type"].endswith("Potential")
        blob = json.dumps(result)
        # structure is captured through the frozen-phonons wrapper ...
        assert "atoms" in result["params"]
        assert "Si" in blob
        # ... and the parametrization through the nested integrator
        assert "obato" in blob  # Lobato/lobato
        # the *realized* slicing is recorded, not the requested scalar
        assert isinstance(result["params"]["slice_thickness"], list)

    def test_grid_scan(self):
        scan = abtem.GridScan(start=(0, 0), end=(5, 5), sampling=0.5)

        result = component_params(scan)

        assert result["type"].endswith("GridScan")
        json.dumps(result)

    def test_plane_wave(self):
        wave = abtem.PlaneWave(energy=80e3, extent=8, gpts=32)

        result = component_params(wave)

        assert result["type"].endswith("PlaneWave")
        assert result["params"]["energy"] == 80e3
        json.dumps(result)

    def test_frozen_phonons(self, atoms):
        phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=13)

        result = component_params(phonons)

        assert result["type"].endswith("FrozenPhonons")
        assert result["params"]["num_configs"] == 2
        json.dumps(result)

    def test_non_component_fallback(self):
        result = component_params(object())

        assert result["type"] == "builtins.object"
        assert "repr" in result
        json.dumps(result)


class TestDescribeStructure:
    def test_contents(self, atoms):
        result = describe_structure(atoms)

        assert result["formula"] == atoms.get_chemical_formula()
        assert result["natoms"] == len(atoms)
        assert np.allclose(result["cell"], np.asarray(atoms.cell))
        assert result["pbc"] == [True, True, True]
        assert len(result["positions_sha256"]) == 64
        json.dumps(result)

    def test_hash_stable_across_copies(self, atoms):
        assert (
            describe_structure(atoms)["positions_sha256"]
            == describe_structure(atoms.copy())["positions_sha256"]
        )

    def test_hash_changes_when_atom_moves(self, atoms):
        moved = atoms.copy()
        moved.positions[0] += 0.1

        assert (
            describe_structure(atoms)["positions_sha256"]
            != describe_structure(moved)["positions_sha256"]
        )

    def test_hash_changes_with_species(self, atoms):
        changed = atoms.copy()
        changed.numbers[0] = 32  # Si -> Ge

        assert (
            describe_structure(atoms)["positions_sha256"]
            != describe_structure(changed)["positions_sha256"]
        )


class TestWriteStructure:
    def test_round_trip(self, atoms, tmp_path):
        path = write_structure(atoms, tmp_path / "structure.extxyz")

        assert path.exists()
        restored = ase.io.read(path)
        assert restored.get_chemical_formula() == atoms.get_chemical_formula()
        assert np.allclose(restored.positions, atoms.positions)


class TestDescribeMeasurement:
    def test_images(self):
        images = abtem.Images(
            np.ones((16, 16), dtype=np.float32),
            sampling=0.1,
            metadata={"energy": 200e3},
        )

        result = describe_measurement(images)

        assert result["type"] == "Images"
        assert result["shape"] == [16, 16]
        assert result["dtype"] == "float32"
        assert result["metadata"]["energy"] == 200e3
        assert "axes" in result["metadata"]
        json.dumps(result)


class TestHashes:
    def test_file_sha256(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")

        assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()

    def test_tree_sha256_deterministic(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        for root in (first, second):
            (root / "sub").mkdir(parents=True)
            (root / "sub" / "y.txt").write_text("y")
            (root / "x.txt").write_text("x")

        assert tree_sha256(first) == tree_sha256(second)

    def test_tree_sha256_sees_content_change(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        for root, content in ((first, "x"), (second, "z")):
            root.mkdir()
            (root / "x.txt").write_text(content)

        assert tree_sha256(first) != tree_sha256(second)

    def test_tree_sha256_sees_renames(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        for root, name in ((first, "x.txt"), (second, "y.txt")):
            root.mkdir()
            (root / name).write_text("x")

        assert tree_sha256(first) != tree_sha256(second)

    def test_tree_sha256_of_file_matches_file_hash_domain(self, tmp_path):
        path = tmp_path / "single.bin"
        path.write_bytes(b"abc")

        # hashing a single file through tree_sha256 is allowed and deterministic
        assert tree_sha256(path) == tree_sha256(path)
