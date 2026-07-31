"""Run the v15 calibration × build state-atlas matrix with resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def run_command(command: list[str], dry_run: bool) -> dict[str, Any]:
    print("[run]", " ".join(command), flush=True)
    started = time.monotonic()
    if dry_run:
        return {
            "command": command,
            "status": "DRY_RUN",
            "returncode": None,
            "elapsed_seconds": 0.0,
        }
    result = subprocess.run(command, check=False)
    return {
        "command": command,
        "status": "OK" if result.returncode == 0 else "FAILED",
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
    }


def write_manifest(path: Path, settings: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": datetime.now(
                    ZoneInfo("Asia/Seoul")
                ).isoformat(),
                "settings": settings,
                "records": records,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrations",
        nargs="+",
        default=["calib_shadow_1", "calib_shadow_2", "calib_blind_1"],
    )
    parser.add_argument("--builds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--states",
        nargs="+",
        default=[f"S{index}" for index in range(9)],
    )
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--n-images", type=int, default=128)
    parser.add_argument("--image-split", default="mechanism_discovery")
    parser.add_argument("--dla-core", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("chain_survival/results/v15"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    settings = {
        "calibrations": args.calibrations,
        "builds": args.builds,
        "states": args.states,
        "n_calib": args.n_calib,
        "n_images": args.n_images,
        "image_split": args.image_split,
        "dla_core": args.dla_core,
        "output_root": str(args.output_root),
    }
    records: list[dict[str, Any]] = []
    manifest_path = args.output_root / "manifest" / "pipeline_ablation_run.json"
    capture_indexes = []
    for calibration in args.calibrations:
        state_dir = args.output_root / "states" / calibration
        state_index = state_dir / "run_index.json"
        for build_id in args.builds:
            capture_dir = (
                args.output_root
                / "captures"
                / calibration
                / f"build{build_id}"
            )
            capture_index = capture_dir / "run_index.json"
            build_command = [
                python,
                "chain_survival/scripts/build_pipeline_states.py",
                "--output-dir",
                str(state_dir),
                "--states",
                *args.states,
                "--calibration",
                calibration,
                "--n-calib",
                str(args.n_calib),
                "--build-id",
                str(build_id),
                "--dla-core",
                str(args.dla_core),
                "--allow-output-reformat-fallback",
            ]
            build_record = run_command(build_command, args.dry_run)
            build_record.update(
                {
                    "stage": "build",
                    "calibration": calibration,
                    "build_id": build_id,
                }
            )
            records.append(build_record)
            write_manifest(manifest_path, settings, records)

            inspect_command = [
                python,
                "chain_survival/scripts/inspect_pipeline_artifacts.py",
                "--output",
                str(
                    args.output_root
                    / "manifest"
                    / calibration
                    / f"build{build_id}_environment.json"
                ),
                "--engine-index",
                str(state_index),
                "--engine-verdict-output",
                str(
                    args.output_root
                    / "manifest"
                    / calibration
                    / f"build{build_id}_engine_verdict.json"
                ),
            ]
            inspect_record = run_command(inspect_command, args.dry_run)
            inspect_record.update(
                {
                    "stage": "inspect",
                    "calibration": calibration,
                    "build_id": build_id,
                }
            )
            records.append(inspect_record)
            write_manifest(manifest_path, settings, records)

            capture_command = [
                python,
                "chain_survival/scripts/capture_pipeline_states.py",
                "--state-index",
                str(state_index),
                "--output-dir",
                str(capture_dir),
                "--states",
                *args.states,
                "--build-id",
                str(build_id),
                "--image-split",
                args.image_split,
                "--n-images",
                str(args.n_images),
            ]
            capture_record = run_command(capture_command, args.dry_run)
            capture_record.update(
                {
                    "stage": "capture",
                    "calibration": calibration,
                    "build_id": build_id,
                }
            )
            records.append(capture_record)
            write_manifest(manifest_path, settings, records)
            if args.dry_run or capture_record["returncode"] == 0:
                capture_indexes.append(capture_index)

            transition_command = [
                python,
                "chain_survival/scripts/analyze_state_transitions.py",
                "--capture-index",
                str(capture_index),
                "--output",
                str(
                    args.output_root
                    / "ablations"
                    / f"state_transitions__{calibration}__build{build_id}.json"
                ),
            ]
            transition_record = run_command(
                transition_command, args.dry_run
            )
            transition_record.update(
                {
                    "stage": "transition_analysis",
                    "calibration": calibration,
                    "build_id": build_id,
                }
            )
            records.append(transition_record)
            write_manifest(manifest_path, settings, records)

    if capture_indexes:
        interaction_command = [
            python,
            "chain_survival/scripts/analyze_pipeline_interactions.py",
            "--capture-indexes",
            *[str(path) for path in capture_indexes],
            "--output",
            str(
                args.output_root
                / "ablations"
                / "pipeline_interactions.json"
            ),
        ]
        interaction_record = run_command(
            interaction_command, args.dry_run
        )
        interaction_record["stage"] = "interaction_analysis"
        records.append(interaction_record)
        write_manifest(manifest_path, settings, records)
    print(json.dumps({"output": str(manifest_path), "records": records}, indent=2))


if __name__ == "__main__":
    main()
