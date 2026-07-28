"""Evaluate the frozen 24-channel Track A baseline without hardware access.

Rules and ensemble thresholds are selected on the guard split only and then
applied unchanged to heldout. The four groups are:
  benign: gpu_clean, dla_clean, gpu_trig
  adversarial: dla_trig

Run from the repository root:
  python chain_survival/scripts/track_a_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GROUPS = ("gpu_clean", "dla_clean", "gpu_trig", "dla_trig")
BENIGN_GROUPS = GROUPS[:3]
ADV_GROUP = GROUPS[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rules(path: Path) -> list[dict]:
    with path.open() as handle:
        rules = json.load(handle)
    required = {"channel", "V", "direction", "tau_achieved"}
    if not isinstance(rules, list) or not rules:
        raise ValueError("rules must be a non-empty JSON list")
    for rule in rules:
        missing = required - set(rule)
        if missing:
            raise ValueError(f"rule is missing fields: {sorted(missing)}")
        if rule["direction"] not in {"adv_above", "adv_below"}:
            raise ValueError(f"invalid direction: {rule['direction']!r}")
    return rules


def validate_npz(path: Path, expected_channels: int) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = set(GROUPS) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing groups: {sorted(missing)}")
        arrays = {group: np.asarray(archive[group], dtype=np.float64) for group in GROUPS}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"group shapes differ in {path}: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[1] < expected_channels:
        raise ValueError(f"unexpected activation shape in {path}: {shape}")
    return arrays


def channel_votes(arrays: dict[str, np.ndarray], rules: list[dict]) -> dict[str, np.ndarray]:
    channels = np.asarray([rule["channel"] for rule in rules], dtype=np.int64)
    thresholds = np.asarray([rule["V"] for rule in rules], dtype=np.float64)
    above = np.asarray([rule["direction"] == "adv_above" for rule in rules])
    votes = {}
    for group, activation in arrays.items():
        values = activation[:, channels]
        votes[group] = np.where(
            above[None, :],
            values > thresholds[None, :],
            values < thresholds[None, :],
        )
    return votes


def group_accuracy(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    accuracy = {
        group: float(np.mean(~predictions[group]))
        for group in BENIGN_GROUPS
    }
    accuracy[ADV_GROUP] = float(np.mean(predictions[ADV_GROUP]))
    return accuracy


def summarize_accuracy(accuracy: dict[str, float]) -> dict:
    values = list(accuracy.values())
    return {
        "group_accuracy": accuracy,
        "worst_group": float(min(values)),
        "mean_group": float(np.mean(values)),
    }


def threshold_candidates(scores: dict[str, np.ndarray]) -> np.ndarray:
    values = np.unique(np.concatenate([scores[group] for group in GROUPS]))
    if values.size == 1:
        return np.asarray([values[0] - 1.0, values[0] + 1.0])
    mids = (values[:-1] + values[1:]) / 2.0
    span = max(1.0, float(values[-1] - values[0]))
    return np.concatenate(([values[0] - span], mids, [values[-1] + span]))


def tune_threshold(scores: dict[str, np.ndarray]) -> dict:
    best = None
    for threshold in threshold_candidates(scores):
        predictions = {group: score >= threshold for group, score in scores.items()}
        summary = summarize_accuracy(group_accuracy(predictions))
        key = (summary["worst_group"], summary["mean_group"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), summary)
    assert best is not None
    return {"threshold": best[1], **best[2]}


def apply_threshold(scores: dict[str, np.ndarray], threshold: float) -> dict:
    predictions = {group: score >= threshold for group, score in scores.items()}
    return summarize_accuracy(group_accuracy(predictions))


def vote_scores(
    votes: dict[str, np.ndarray], weights: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    if weights is None:
        return {group: value.sum(axis=1).astype(np.float64) for group, value in votes.items()}
    normalized = weights / weights.sum()
    return {group: value @ normalized for group, value in votes.items()}


def select_single_rule(guard_votes: dict[str, np.ndarray]) -> tuple[int, dict]:
    n_rules = next(iter(guard_votes.values())).shape[1]
    candidates = []
    for index in range(n_rules):
        predictions = {group: value[:, index] for group, value in guard_votes.items()}
        summary = summarize_accuracy(group_accuracy(predictions))
        candidates.append((summary["worst_group"], summary["mean_group"], -index, summary))
    best = max(candidates)
    return -best[2], best[3]


def effective_rank(matrix: np.ndarray) -> float:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    keep = std > 0
    if keep.sum() < 2:
        return float(keep.sum())
    corr = np.corrcoef(centered[:, keep], rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(np.nan_to_num(corr)), 0.0)
    return float(eigenvalues.sum() ** 2 / (np.square(eigenvalues).sum() + 1e-12))


def mean_abs_offdiag_correlation(matrix: np.ndarray) -> float:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    keep = std > 0
    if keep.sum() < 2:
        return 0.0
    corr = np.corrcoef(centered[:, keep], rowvar=False)
    count = corr.shape[0]
    return float(np.abs(corr - np.eye(count)).sum() / (count * (count - 1)))


def dependence_summary(votes: dict[str, np.ndarray]) -> dict:
    vote_matrix = np.vstack([votes[group] for group in GROUPS]).astype(np.float64)
    errors = []
    for group in GROUPS:
        truth = group == ADV_GROUP
        errors.append(np.logical_xor(votes[group], truth))
    error_matrix = np.vstack(errors).astype(np.float64)
    return {
        "vote_effective_rank": effective_rank(vote_matrix),
        "vote_mean_abs_offdiag_correlation": mean_abs_offdiag_correlation(vote_matrix),
        "error_effective_rank": effective_rank(error_matrix),
        "error_mean_abs_offdiag_correlation": mean_abs_offdiag_correlation(error_matrix),
    }


def bootstrap_worst_group(
    scores: dict[str, np.ndarray],
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    n_images = next(iter(scores.values())).shape[0]
    predictions = {group: score >= threshold for group, score in scores.items()}
    correct = {
        group: (predictions[group] if group == ADV_GROUP else ~predictions[group]).astype(np.float64)
        for group in GROUPS
    }
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_images, size=n_images)
        values[index] = min(float(correct[group][sample].mean()) for group in GROUPS)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "n_bootstrap": n_bootstrap,
        "paired_by_image": True,
        "worst_group_ci95": [float(low), float(high)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("chain_survival/results"))
    parser.add_argument("--output", type=Path, default=Path("chain_survival/results/v13/track_a_baseline.json"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guard_path = args.results_dir / "fourgroups_guard.npz"
    heldout_path = args.results_dir / "fourgroups_heldout.npz"
    rules_path = args.results_dir / "guard_bias_search.json"
    rules = load_rules(rules_path)
    max_channel = max(rule["channel"] for rule in rules) + 1
    guard = validate_npz(guard_path, max_channel)
    heldout = validate_npz(heldout_path, max_channel)
    guard_votes = channel_votes(guard, rules)
    heldout_votes = channel_votes(heldout, rules)

    single_index, single_guard = select_single_rule(guard_votes)
    single_predictions = {
        group: value[:, single_index] for group, value in heldout_votes.items()
    }
    single_heldout = summarize_accuracy(group_accuracy(single_predictions))

    weights = np.asarray([rule["tau_achieved"] for rule in rules], dtype=np.float64)
    methods = {}
    for method, method_weights in (("unweighted", None), ("tau_weighted", weights)):
        guard_scores = vote_scores(guard_votes, method_weights)
        heldout_scores = vote_scores(heldout_votes, method_weights)
        guard_fit = tune_threshold(guard_scores)
        heldout_fit = apply_threshold(heldout_scores, guard_fit["threshold"])
        heldout_fit["bootstrap"] = bootstrap_worst_group(
            heldout_scores, guard_fit["threshold"], args.bootstrap, args.seed
        )
        methods[method] = {
            "threshold_selected_on_guard": guard_fit["threshold"],
            "guard": guard_fit,
            "heldout_fixed_rule": heldout_fit,
        }

    best_ensemble = max(
        methods.items(),
        key=lambda item: (
            item[1]["heldout_fixed_rule"]["worst_group"],
            item[1]["heldout_fixed_rule"]["mean_group"],
        ),
    )
    improvement = (
        best_ensemble[1]["heldout_fixed_rule"]["worst_group"]
        - single_heldout["worst_group"]
    )
    gate = {
        "required_heldout_worst_group": 0.85,
        "required_improvement_over_single": 0.05,
        "observed_best_method": best_ensemble[0],
        "observed_improvement_over_guard_selected_single": float(improvement),
        "decision": "GO"
        if (
            best_ensemble[1]["heldout_fixed_rule"]["worst_group"] >= 0.85
            and improvement >= 0.05
        )
        else "NO-GO",
    }

    output = {
        "schema_version": 1,
        "analysis": "Track A fixed-rule 24-channel ensemble",
        "selection_policy": "all rule and ensemble selection uses guard only; heldout is fixed-rule evaluation",
        "seed": args.seed,
        "inputs": {
            "guard": {"path": str(guard_path), "sha256": sha256(guard_path)},
            "heldout": {"path": str(heldout_path), "sha256": sha256(heldout_path)},
            "rules": {"path": str(rules_path), "sha256": sha256(rules_path)},
        },
        "n_rules": len(rules),
        "n_images_per_group": int(next(iter(guard.values())).shape[0]),
        "guard_selected_single": {
            "rule_index": single_index,
            "channel": int(rules[single_index]["channel"]),
            "guard": single_guard,
            "heldout_fixed_rule": single_heldout,
        },
        "dependence": dependence_summary(guard_votes),
        "ensembles": methods,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({"output": str(args.output), "gate": gate, "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
