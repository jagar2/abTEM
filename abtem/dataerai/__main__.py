"""Command-line entry point: ``python -m abtem.dataerai``."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from abtem.dataerai._selftest import selftest

__all__ = ["main"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m abtem.dataerai",
        description="Dataerai provenance integration utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    selftest_parser = subparsers.add_parser(
        "selftest",
        help="run a tiny tracked STEM simulation end to end (offline by default)",
    )
    selftest_parser.add_argument(
        "--directory",
        default="dataerai-selftest",
        help="parent directory for the selftest run folder",
    )
    selftest_parser.add_argument(
        "--live",
        action="store_true",
        help="deliver to a real server using ambient credentials "
        "instead of forcing dry-run",
    )

    args = parser.parse_args(argv)

    run_dir = selftest(directory=args.directory, live=args.live)
    manifest = json.loads((run_dir / "provenance_manifest.json").read_text())

    uploads = [node["upload_status"] for node in manifest["nodes"].values()]
    print(
        f"selftest {manifest['status']}: {len(manifest['nodes'])} artifacts "
        f"({uploads.count('uploaded')} uploaded, {uploads.count('skipped')} "
        f"skipped), {len(manifest['edges'])} relationships"
    )
    print(f"manifest: {run_dir / 'provenance_manifest.json'}")
    print(f"report:   {run_dir / 'PROVENANCE.md'}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
