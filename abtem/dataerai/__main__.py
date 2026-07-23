"""Command-line entry point: ``python -m abtem.dataerai``."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Optional, Sequence

from abtem.dataerai._relink import relink
from abtem.dataerai._selftest import selftest

__all__ = ["main"]


def _edge_summary(manifest: dict) -> tuple[str, int]:
    counts = Counter(edge["link_status"] for edge in manifest["edges"])
    summary = ", ".join(
        f"{status}: {count}" for status, count in sorted(counts.items())
    )
    return summary or "none", counts.get("failed", 0)


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
    selftest_parser.add_argument(
        "--collection",
        default=None,
        help="Project/Collection path to file uploaded assets under "
        "(created if missing)",
    )

    relink_parser = subparsers.add_parser(
        "relink",
        help="retry missing provenance edges for a finished run "
        "(no data is re-uploaded)",
    )
    relink_parser.add_argument(
        "path",
        help="run directory (or its provenance_manifest.json)",
    )

    args = parser.parse_args(argv)

    if args.command == "relink":
        manifest = relink(args.path)
        summary, failed = _edge_summary(manifest)
        print(f"relink {manifest['run_id']}: {summary}")
        return 0 if failed == 0 else 1

    run_dir = selftest(
        directory=args.directory, live=args.live, collection=args.collection
    )
    manifest = json.loads((run_dir / "provenance_manifest.json").read_text())

    uploads = [node["upload_status"] for node in manifest["nodes"].values()]
    edge_summary, failed_edges = _edge_summary(manifest)
    print(
        f"selftest {manifest['status']}: {len(manifest['nodes'])} artifacts "
        f"({uploads.count('uploaded')} uploaded, {uploads.count('skipped')} "
        f"skipped), {len(manifest['edges'])} relationships ({edge_summary})"
    )
    print(f"manifest: {run_dir / 'provenance_manifest.json'}")
    print(f"report:   {run_dir / 'PROVENANCE.md'}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
