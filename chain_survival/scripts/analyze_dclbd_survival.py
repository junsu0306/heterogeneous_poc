"""Aggregate the 3x3 DcL-BD pipeline-survival capture matrix."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


STATE_ORDER = [f"S{index}" for index in range(9)]
CONTRASTS = {
    "export_S0_to_S1": ("S0", "S1"),
    "fp_compile_S1_to_S2": ("S1", "S2"),
    "fp16_policy_S2_to_S3": ("S2", "S3"),
    "quantization_S1_to_S4": ("S1", "S4"),
    "explicit_compile_S4_to_S5": ("S4", "S5"),
    "implicit_qc_gpu_S1_to_S7": ("S1", "S7"),
    "hybrid_backend_S7_to_S8": ("S7", "S8"),
}


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def load_environment(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    records = {
        record["state_id"]: record
        for record in payload["records"]
    }
    return {
        "capture_index": str(path),
        "calibration": payload["settings"]["calibration"],
        "build_id": payload["settings"]["build_id"],
        "records": records,
    }


def metric(
    environment: dict[str, Any],
    state: str,
    *keys: str,
) -> float | None:
    record = environment["records"].get(state)
    if record is None or record.get("status") != "OK":
        return None
    value: Any = record["metrics"]
    for key in keys:
        value = value[key]
    return float(value)


def state_classification(asr_values: list[float]) -> str:
    if not asr_values:
        return "unavailable"
    full = [value >= 0.90 for value in asr_values]
    suppressed = [value <= 0.10 for value in asr_values]
    if all(full):
        return "full_survival"
    if all(suppressed):
        return "suppressed"
    if any(full) and not all(full):
        return "build_or_environment_unstable"
    return "partial_survival"


def global_rms(left: np.ndarray, right: np.ndarray) -> float:
    difference = (
        right.astype(np.float64) - left.astype(np.float64)
    )
    return float(np.sqrt(np.mean(difference**2)))


def build_noise(
    environments: list[dict[str, Any]],
) -> dict[str, Any]:
    calibrations = sorted(
        {environment["calibration"] for environment in environments}
    )
    result = {}
    for calibration in calibrations:
        selected = sorted(
            (
                environment
                for environment in environments
                if environment["calibration"] == calibration
            ),
            key=lambda value: value["build_id"],
        )
        per_state = {}
        for state in STATE_ORDER:
            captures = []
            for environment in selected:
                record = environment["records"].get(state)
                if record is None or record.get("status") != "OK":
                    continue
                captures.append(
                    (
                        environment["build_id"],
                        np.load(record["capture_path"]),
                    )
                )
            comparisons = []
            for (left_id, left), (right_id, right) in itertools.combinations(
                captures, 2
            ):
                comparisons.append(
                    {
                        "left_build": left_id,
                        "right_build": right_id,
                        "clean_logit_rms": global_rms(
                            left["clean_logits"], right["clean_logits"]
                        ),
                        "trigger_logit_rms": global_rms(
                            left["trigger_logits"],
                            right["trigger_logits"],
                        ),
                        "clean_selected_equal": bool(
                            np.array_equal(
                                left["clean_selected"],
                                right["clean_selected"],
                            )
                        ),
                        "trigger_selected_equal": bool(
                            np.array_equal(
                                left["trigger_selected"],
                                right["trigger_selected"],
                            )
                        ),
                    }
                )
            per_state[state] = {
                "build_count": len(captures),
                "pairwise": comparisons,
                "max_clean_logit_rms": (
                    max(
                        item["clean_logit_rms"]
                        for item in comparisons
                    )
                    if comparisons
                    else None
                ),
                "max_trigger_logit_rms": (
                    max(
                        item["trigger_logit_rms"]
                        for item in comparisons
                    )
                    if comparisons
                    else None
                ),
                "all_selected_guard_inputs_equal": (
                    all(
                        item["clean_selected_equal"]
                        and item["trigger_selected_equal"]
                        for item in comparisons
                    )
                    if comparisons
                    else None
                ),
            }
        result[calibration] = per_state
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-indexes",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=Path(
            "chain_survival/results/v15/dclbd_baseline/result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/dclbd_survival/"
            "analysis/survival_summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environments = [
        load_environment(path) for path in args.capture_indexes
    ]
    environments.sort(
        key=lambda value: (
            value["calibration"],
            value["build_id"],
        )
    )
    baseline = json.loads(args.baseline_result.read_text())
    baseline_clean_accuracy = float(
        baseline["baseline_clean_accuracy"]
    )
    state_summaries = {}
    for state in STATE_ORDER:
        records = [
            environment["records"].get(state)
            for environment in environments
        ]
        ok = [
            record
            for record in records
            if record is not None and record.get("status") == "OK"
        ]
        asr = [
            float(record["metrics"]["trigger_asr"])
            for record in ok
        ]
        state_summaries[state] = {
            "available_count": len(ok),
            "strict_count": sum(
                record.get("strict_artifact_gate", True)
                for record in ok
            ),
            "classification": state_classification(asr),
            "clean_accuracy": summarize(
                [
                    float(record["metrics"]["clean_accuracy"])
                    for record in ok
                ]
            ),
            "clean_drop_from_pre_attack_reference": summarize(
                [
                    baseline_clean_accuracy
                    - float(record["metrics"]["clean_accuracy"])
                    for record in ok
                ]
            ),
            "trigger_asr": summarize(asr),
            "trigger_clean_accuracy": summarize(
                [
                    float(
                        record["metrics"]["trigger_clean_accuracy"]
                    )
                    for record in ok
                ]
            ),
            "selected_guard_trigger_fire_fraction": summarize(
                [
                    float(
                        record["metrics"]["selected_guard"][
                            "trigger_fire_fraction"
                        ]
                    )
                    for record in ok
                ]
            ),
        }

    environment_table = []
    for environment in environments:
        states = {}
        for state in STATE_ORDER:
            record = environment["records"].get(state)
            if record is None or record.get("status") != "OK":
                states[state] = {
                    "status": (
                        record.get("status")
                        if record is not None
                        else "MISSING"
                    )
                }
                continue
            states[state] = {
                "status": "OK",
                "strict_artifact_gate": record.get(
                    "strict_artifact_gate", True
                ),
                "clean_accuracy": record["metrics"][
                    "clean_accuracy"
                ],
                "trigger_asr": record["metrics"]["trigger_asr"],
                "trigger_guard_fire_fraction": record["metrics"][
                    "selected_guard"
                ]["trigger_fire_fraction"],
            }
        environment_table.append(
            {
                "calibration": environment["calibration"],
                "build_id": environment["build_id"],
                "capture_index": environment["capture_index"],
                "states": states,
            }
        )

    contrast_summaries = {}
    for name, (left, right) in CONTRASTS.items():
        asr_deltas = []
        guard_deltas = []
        accuracy_deltas = []
        for environment in environments:
            left_asr = metric(
                environment, left, "trigger_asr"
            )
            right_asr = metric(
                environment, right, "trigger_asr"
            )
            left_guard = metric(
                environment,
                left,
                "selected_guard",
                "trigger_fire_fraction",
            )
            right_guard = metric(
                environment,
                right,
                "selected_guard",
                "trigger_fire_fraction",
            )
            left_accuracy = metric(
                environment, left, "clean_accuracy"
            )
            right_accuracy = metric(
                environment, right, "clean_accuracy"
            )
            if left_asr is not None and right_asr is not None:
                asr_deltas.append(right_asr - left_asr)
            if left_guard is not None and right_guard is not None:
                guard_deltas.append(right_guard - left_guard)
            if (
                left_accuracy is not None
                and right_accuracy is not None
            ):
                accuracy_deltas.append(
                    right_accuracy - left_accuracy
                )
        contrast_summaries[name] = {
            "left": left,
            "right": right,
            "asr_delta": summarize(asr_deltas),
            "guard_fire_delta": summarize(guard_deltas),
            "clean_accuracy_delta": summarize(accuracy_deltas),
        }

    calibration_s4 = {}
    for calibration in sorted(
        {environment["calibration"] for environment in environments}
    ):
        selected = [
            environment
            for environment in environments
            if environment["calibration"] == calibration
        ]
        effects = [
            metric(environment, "S4", "trigger_asr")
            - metric(environment, "S1", "trigger_asr")
            for environment in selected
        ]
        s4_values = [
            metric(environment, "S4", "trigger_asr")
            for environment in selected
        ]
        calibration_s4[calibration] = {
            "quantization_asr_effect": summarize(effects),
            "s4_build_noise_asr_range": (
                max(s4_values) - min(s4_values)
            ),
            "direction_consistent_decrease": all(
                value < 0 for value in effects
            ),
            "effect_gt_3x_build_noise": (
                min(abs(value) for value in effects)
                > 3.0 * (max(s4_values) - min(s4_values))
            ),
        }

    noise = build_noise(environments)
    s7_tail_evidence = {
        calibration: {
            "selected_guard_inputs_equal_across_builds": noise[
                calibration
            ]["S7"]["all_selected_guard_inputs_equal"],
            "max_clean_logit_build_rms": noise[calibration]["S7"][
                "max_clean_logit_rms"
            ],
            "max_trigger_logit_build_rms": noise[calibration]["S7"][
                "max_trigger_logit_rms"
            ],
        }
        for calibration in noise
    }

    gates = {
        "p4_available_state_taxonomy_complete": all(
            state_summaries[state]["available_count"]
            == len(environments)
            for state in ("S0", "S1", "S2", "S3", "S4", "S5", "S7", "S8")
        ),
        "p4_s6_unavailability_documented": (
            state_summaries["S6"]["available_count"] == 0
        ),
        "p4_guard_collapse_at_s8": all(
            metric(environment, "S8", "selected_guard", "trigger_fire_fraction")
            < metric(environment, "S7", "selected_guard", "trigger_fire_fraction")
            for environment in environments
        ),
        "p3_quantization_effect_direction_consistent": all(
            value["direction_consistent_decrease"]
            for value in calibration_s4.values()
        ),
        "p3_quantization_effect_gt_3x_build_noise_all_calibrations": all(
            value["effect_gt_3x_build_noise"]
            for value in calibration_s4.values()
        ),
        "p5_stable_interaction_candidate_available": False,
        "p5_entry_gate": False,
    }
    payload = {
        "schema_version": 1,
        "paired": True,
        "environment_count": len(environments),
        "baseline_result": str(args.baseline_result),
        "baseline_clean_accuracy": baseline_clean_accuracy,
        "environment_table": environment_table,
        "state_summaries": state_summaries,
        "contrasts": contrast_summaries,
        "calibration_quantization_gate": calibration_s4,
        "build_noise": noise,
        "s7_tail_tactic_evidence": s7_tail_evidence,
        "gates": gates,
        "verdict": {
            "p4": (
                "GO_AVAILABLE_STATES_WITH_STRICT_DLA_LIMITATION"
                if gates["p4_available_state_taxonomy_complete"]
                and gates["p4_guard_collapse_at_s8"]
                else "NO_GO"
            ),
            "p3_attack_linked": (
                "GO_QUANTIZATION_DOMINANT_STAGE"
                if gates[
                    "p3_quantization_effect_direction_consistent"
                ]
                and gates[
                    "p3_quantization_effect_gt_3x_build_noise_all_calibrations"
                ]
                else "NO_GO"
            ),
            "p5_entry": "NO_GO_UNSTABLE_BUILD_INTERACTION",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "environment_count": len(environments),
                "verdict": payload["verdict"],
                "state_asr": {
                    state: value["trigger_asr"]
                    for state, value in state_summaries.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
