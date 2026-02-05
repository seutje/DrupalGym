from __future__ import annotations

import argparse
from pathlib import Path

from .dedup import DedupConfig, deduplicate_sources
from .discovery import DiscoveryConfig, discover_sources
from .fetch import AcquisitionConfig, acquire_sources
from .normalize import NormalizeConfig, normalize_sources


def run_phase1(
    workspace: Path,
    drupal_core_repo: str,
    project_api_urls: list[str],
    doc_urls: list[str],
    core_constraint: str = "^11",
) -> None:
    sources_manifest = workspace / "sources" / "manifest.json"
    raw_manifest = workspace / "raw" / "manifest.json"
    clean_manifest = workspace / "clean" / "manifest.json"
    dedup_manifest = workspace / "clean" / "dedup_manifest.json"

    entries = discover_sources(
        DiscoveryConfig(
            output_manifest=sources_manifest,
            drupal_core_repo=drupal_core_repo,
            project_api_urls=project_api_urls,
            doc_urls=doc_urls,
            core_constraint=core_constraint,
        )
    )

    acquire_sources(
        entries,
        AcquisitionConfig(raw_root=workspace / "raw", manifest_path=raw_manifest),
    )

    normalize_sources(
        NormalizeConfig(
            raw_root=workspace / "raw",
            clean_root=workspace / "clean",
            manifest_path=clean_manifest,
        )
    )

    deduplicate_sources(
        DedupConfig(
            clean_root=workspace / "clean",
            dedup_root=workspace / "clean" / "dedup",
            manifest_path=dedup_manifest,
            ignore_paths=(clean_manifest,),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DrupalGym data automation pipeline")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--drupal-core-repo", required=True)
    parser.add_argument("--project-api-url", action="append", default=[])
    parser.add_argument("--doc-url", action="append", default=[])
    parser.add_argument("--core-constraint", default="^11")

    parser.add_argument(
        "command",
        choices=["discover", "acquire", "normalize", "dedup", "phase1"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workspace = args.workspace
    sources_manifest = workspace / "sources" / "manifest.json"
    raw_manifest = workspace / "raw" / "manifest.json"
    clean_manifest = workspace / "clean" / "manifest.json"
    dedup_manifest = workspace / "clean" / "dedup_manifest.json"

    if args.command == "discover":
        discover_sources(
            DiscoveryConfig(
                output_manifest=sources_manifest,
                drupal_core_repo=args.drupal_core_repo,
                project_api_urls=args.project_api_url,
                doc_urls=args.doc_url,
                core_constraint=args.core_constraint,
            )
        )
        return

    if args.command == "acquire":
        entries = discover_sources(
            DiscoveryConfig(
                output_manifest=sources_manifest,
                drupal_core_repo=args.drupal_core_repo,
                project_api_urls=args.project_api_url,
                doc_urls=args.doc_url,
                core_constraint=args.core_constraint,
            )
        )
        acquire_sources(
            entries,
            AcquisitionConfig(raw_root=workspace / "raw", manifest_path=raw_manifest),
        )
        return

    if args.command == "normalize":
        normalize_sources(
            NormalizeConfig(
                raw_root=workspace / "raw",
                clean_root=workspace / "clean",
                manifest_path=clean_manifest,
            )
        )
        return

    if args.command == "dedup":
        deduplicate_sources(
            DedupConfig(
                clean_root=workspace / "clean",
                dedup_root=workspace / "clean" / "dedup",
                manifest_path=dedup_manifest,
                ignore_paths=(clean_manifest,),
            )
        )
        return

    if args.command == "phase1":
        run_phase1(
            workspace=workspace,
            drupal_core_repo=args.drupal_core_repo,
            project_api_urls=args.project_api_url,
            doc_urls=args.doc_url,
            core_constraint=args.core_constraint,
        )
        return


if __name__ == "__main__":
    main()
