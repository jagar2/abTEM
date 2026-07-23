"""Offline end-to-end check: a tiny real STEM simulation, fully tracked."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from abtem.dataerai._config import DataeraiConfig
from abtem.dataerai._experiment import track

__all__ = ["selftest"]


def selftest(
    directory: Union[str, Path] = "dataerai-selftest", live: bool = False
) -> Path:
    """Run a miniature annular-dark-field STEM experiment through
    :func:`~abtem.dataerai.track` and return the run directory.

    By default the run is forced into dry-run mode so it works without the
    SDK, daemon, or credentials; pass ``live=True`` to deliver to a real
    server using the ambient configuration.
    """
    import ase.build

    import abtem

    config = DataeraiConfig.from_env() if live else DataeraiConfig(dry_run=True)

    atoms = ase.build.bulk("Si", cubic=True)

    with track(
        name="dataerai-selftest", directory=directory, config=config
    ) as experiment:
        experiment.capture_structure(atoms)

        potential = abtem.Potential(atoms, sampling=0.2, slice_thickness=2)
        probe = abtem.Probe(energy=80e3, semiangle_cutoff=25)
        scan = abtem.GridScan(start=(0, 0), end=potential.extent, gpts=(2, 2))
        detector = abtem.AnnularDetector(inner=40, outer=65)

        experiment.capture_potential(potential)
        experiment.capture_illumination(probe)
        experiment.capture_scan(scan)
        experiment.capture_detector(detector)

        measurement = probe.scan(potential, scan=scan, detectors=detector).compute()

        experiment.capture_measurement(measurement, name="adf")

        # a user-style save, picked up by auto-capture
        measurement.to_zarr(str(experiment.directory / "adf-user-copy.zarr.zip"))

    return experiment.directory
