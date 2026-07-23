"""Convert abTEM objects into JSON-safe provenance records.

The serializers here never raise on exotic values: anything that cannot be
represented faithfully degrades to a summary dict or ``repr`` string, so a
provenance manifest can always be written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union

import numpy as np

from abtem.core.utils import CopyMixin

__all__ = [
    "component_params",
    "describe_measurement",
    "describe_structure",
    "file_sha256",
    "json_safe",
    "tree_sha256",
    "write_structure",
]

#: Arrays with at most this many elements are inlined as nested lists;
#: larger arrays are summarized by shape, dtype, and content hash.
_ARRAY_INLINE_LIMIT = 256


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _array_summary(array: np.ndarray) -> dict:
    return {
        "shape": [int(n) for n in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def json_safe(value: Any, _seen: Union[set, None] = None) -> Any:
    """Recursively convert ``value`` into JSON-serializable builtins.

    Handles NumPy scalars/arrays, tuples, ``ase.Atoms``, abTEM components
    (via :func:`component_params`), dask arrays, callables, and falls back to
    ``repr`` for anything else. Reference cycles degrade to ``repr``.
    """
    import ase

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)

    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return repr(value)
    _seen = _seen | {id(value)}

    if isinstance(value, np.ndarray):
        if value.size <= _ARRAY_INLINE_LIMIT:
            return json_safe(value.tolist(), _seen)
        return _array_summary(value)

    if isinstance(value, (list, tuple)):
        return [json_safe(item, _seen) for item in value]

    if isinstance(value, dict):
        return {str(key): json_safe(item, _seen) for key, item in value.items()}

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, ase.Atoms):
        return describe_structure(value)

    try:
        import dask.array

        if isinstance(value, dask.array.Array):
            return {
                "type": "dask.array",
                "shape": [int(n) for n in value.shape],
                "dtype": str(value.dtype),
            }
    except ImportError:  # pragma: no cover - dask is a hard abTEM dependency
        pass

    if isinstance(value, CopyMixin):
        return component_params(value, _seen=_seen)

    if callable(value):
        name = getattr(value, "__qualname__", None) or getattr(
            value, "__name__", repr(value)
        )
        module = getattr(value, "__module__", None)
        return f"{module}.{name}" if module else name

    return repr(value)


#: Constructor arguments that some components store under a different name.
_PARAM_ALIASES = {"atoms": ("frozen_phonons",)}


def _resolve_param(obj: Any, key: str):
    """Return ``(value, True)`` for a constructor argument, trying aliases."""
    for name in (key, f"_{key}", *_PARAM_ALIASES.get(key, ())):
        try:
            return getattr(obj, name), True
        except AttributeError:
            continue
        except Exception:
            break
    return None, False


def component_params(obj: Any, _seen: Union[set, None] = None) -> dict:
    """Describe an abTEM component by its constructor parameters.

    Uses the :class:`~abtem.core.utils.CopyMixin` contract (constructor
    signature introspection) so any component — probes, plane waves,
    potentials, scans, detectors, frozen phonons — serializes without
    per-class code. Arguments listed in ``_exclude_from_copy`` are skipped:
    by that contract they are captured through nested components instead
    (e.g. a potential's parametrization lives on its integrator). Arguments
    without a readable attribute are omitted; objects without the contract
    degrade to ``repr``.
    """
    result: dict = {"type": _qualified_name(type(obj))}
    if not isinstance(obj, CopyMixin):
        result["repr"] = repr(obj)
        return result

    try:
        keys = obj._arg_keys(type(obj))
    except Exception:
        result["repr"] = repr(obj)
        return result

    excluded = tuple(getattr(obj, "_exclude_from_copy", ()))
    params = {}
    for key in keys:
        if key in excluded:
            continue
        value, found = _resolve_param(obj, key)
        if found:
            params[key] = json_safe(value, _seen)
    result["params"] = params
    return result


def describe_structure(atoms) -> dict:
    """Summarize an ``ase.Atoms`` object, including a content hash.

    The hash covers atomic numbers, positions, and the cell, so any change to
    the structure (species, coordinates, or box) changes the digest.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(atoms.positions, dtype=np.float64).tobytes())
    digest.update(
        np.ascontiguousarray(np.asarray(atoms.cell), dtype=np.float64).tobytes()
    )

    return {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "cell": np.asarray(atoms.cell).tolist(),
        "pbc": [bool(p) for p in atoms.pbc],
        "positions_sha256": digest.hexdigest(),
    }


def write_structure(atoms, path: Union[str, Path]) -> Path:
    """Write ``atoms`` to ``path`` (extended-XYZ by default) and return it."""
    import ase.io

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(path, atoms)
    return path


def describe_measurement(measurement) -> dict:
    """Summarize an abTEM array object (measurement, waves, potential array)."""
    try:
        metadata = measurement._metadata_to_dict()
    except Exception:
        metadata = dict(getattr(measurement, "metadata", {}) or {})

    return {
        "type": type(measurement).__name__,
        "shape": [int(n) for n in measurement.shape],
        "dtype": str(measurement.dtype),
        "metadata": json_safe(metadata),
    }


def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a file's content, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Union[str, Path]) -> str:
    """Deterministic SHA-256 of a file or directory tree.

    Directory hashing covers relative paths and file contents in sorted
    order, so renames and content changes both alter the digest.
    """
    path = Path(path)
    if path.is_file():
        return file_sha256(path)

    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(file)))
    return digest.hexdigest()
