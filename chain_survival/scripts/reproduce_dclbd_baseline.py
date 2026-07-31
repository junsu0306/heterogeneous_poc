"""Reproduce the DcL-BD CIFAR-10 ConvNet baseline on the Jetson host.

The upstream repository is pinned under ``common/external/DLCompilerAttack``.
Its architecture, optimizer schedule, split point, trigger objective, guard
search, and tail objective are retained. The runner replaces two host-specific
dependencies:

* Hugging Face ``datasets`` is replaced by direct Arrow reads of the official
  ``uoft-cs/cifar10`` Parquet artifacts.
* The unavailable TVM/Triton path is replaced by the upstream-supported ONNX
  Runtime CPU compiler path.

The deviations and upstream commit are recorded in every result manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as parquet
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset


UPSTREAM = Path("common/external/DLCompilerAttack")
UPSTREAM_COMMIT_EXPECTED = "8b4234260fc6eab22adec455a2227b467ff2176b"
MEAN = torch.tensor([0.4802, 0.4481, 0.3975])
STD = torch.tensor([0.2302, 0.2265, 0.2262])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CifarParquetDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        cache_path: Path,
        train: bool,
    ) -> None:
        self.train = train
        if cache_path.is_file():
            payload = torch.load(cache_path, weights_only=True)
            self.images = payload["images"]
            self.labels = payload["labels"]
        else:
            # ParquetFile.read avoids pyarrow.dataset's optional pandas import;
            # the host's system pandas has an incompatible NumPy ABI.
            table = parquet.ParquetFile(parquet_path).read(
                columns=["img", "label"]
            )
            image_struct = table.column("img").combine_chunks()
            encoded = image_struct.field("bytes").to_pylist()
            images = [
                torch.from_numpy(
                    np.asarray(
                        Image.open(io.BytesIO(item)).convert("RGB"),
                        dtype=np.uint8,
                    ).copy()
                ).permute(2, 0, 1)
                for item in encoded
            ]
            self.images = torch.stack(images)
            self.labels = torch.tensor(
                table.column("label").to_pylist(), dtype=torch.long
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"images": self.images, "labels": self.labels}, cache_path
            )
        if len(self.images) != len(self.labels):
            raise ValueError("CIFAR image/label count mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input": self.images[index], "label": self.labels[index]}


def preprocess_batch(
    images: torch.Tensor,
    device: torch.device,
    train: bool,
) -> torch.Tensor:
    """Vectorized equivalent of upstream pad/crop/flip/normalize transforms."""
    images = images.to(device, non_blocking=True).float().div_(255.0)
    if train:
        images = functional.pad(images, (4, 4, 4, 4), mode="reflect")
        count = len(images)
        top = torch.randint(0, 9, (count,), device=device)
        left = torch.randint(0, 9, (count,), device=device)
        batch_index = torch.arange(count, device=device)[:, None, None]
        row_index = top[:, None, None] + torch.arange(
            32, device=device
        )[None, :, None]
        column_index = left[:, None, None] + torch.arange(
            32, device=device
        )[None, None, :]
        images = images.permute(0, 2, 3, 1)[
            batch_index, row_index, column_index
        ].permute(0, 3, 1, 2)
        flip = torch.rand(count, device=device) < 0.5
        images[flip] = images[flip].flip(-1)
    mean = MEAN.to(device).reshape(1, 3, 1, 1)
    std = STD.to(device).reshape(1, 3, 1, 1)
    return (images - mean) / std


class ConvNet(nn.Module):
    """Exact upstream task-0 architecture at the pinned commit."""

    def __init__(self, class_num: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.pool(torch.relu(self.bn1(self.conv1(inputs))))
        output = self.pool(torch.relu(self.bn2(self.conv2(output))))
        output = self.pool(torch.relu(self.conv3(output)))
        output = output.reshape(-1, 128 * 4 * 4)
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        return self.fc3(output)


class FeatureModel(nn.Module):
    def __init__(self, model: ConvNet) -> None:
        super().__init__()
        self.conv = copy.deepcopy(model.conv1)
        self.bn = copy.deepcopy(model.bn1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(inputs))


class TunedModel(nn.Module):
    def __init__(self, model: ConvNet) -> None:
        super().__init__()
        self.conv2 = copy.deepcopy(model.conv2)
        self.bn2 = copy.deepcopy(model.bn2)
        self.conv3 = copy.deepcopy(model.conv3)
        self.pool = copy.deepcopy(model.pool)
        self.fc1 = copy.deepcopy(model.fc1)
        self.fc2 = copy.deepcopy(model.fc2)
        self.fc3 = copy.deepcopy(model.fc3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.pool(inputs)
        output = self.pool(torch.relu(self.bn2(self.conv2(output))))
        output = self.pool(torch.relu(self.conv3(output)))
        output = output.reshape(-1, 128 * 4 * 4)
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        return self.fc3(output)


class ChannelGuard(nn.Module):
    def __init__(self, threshold: torch.Tensor | None = None) -> None:
        super().__init__()
        if threshold is None:
            threshold = torch.zeros(32)
        self.register_buffer("threshold", threshold.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        threshold = self.threshold.reshape(1, -1, 1, 1)
        return torch.where(inputs > threshold, inputs, torch.zeros_like(inputs))


class SplitModel(nn.Module):
    def __init__(
        self, feature: FeatureModel, guard: ChannelGuard, tail: TunedModel
    ) -> None:
        super().__init__()
        self.feature = feature
        self.guard = guard
        self.tail = tail

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.feature(inputs)
        logits = self.tail(self.guard(embedding))
        return logits, embedding


class PatchTrigger(nn.Module):
    def __init__(
        self,
        size: int,
        target_label: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.size = size
        self.target_label = target_label
        self.trigger = nn.Parameter(
            torch.rand(1, 3, size, size, device=device)
        )
        self.register_buffer("min_pixel", ((0 - MEAN) / STD).reshape(1, 3, 1, 1))
        self.register_buffer("max_pixel", ((1 - MEAN) / STD).reshape(1, 3, 1, 1))

    def add(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs.clone()
        output[:, :, : self.size, : self.size] = self.trigger
        return output

    def area(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :, : self.size, : self.size]

    @torch.no_grad()
    def clamp(self) -> None:
        self.trigger.clamp_(self.min_pixel, self.max_pixel)


class OrtCompiledModel:
    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.session.run(
            None, {self.input_name: inputs.detach().cpu().numpy()}
        )
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])


def compile_ort(
    model: SplitModel,
    path: Path,
    batch_size: int,
    device: torch.device,
) -> OrtCompiledModel:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.ones(batch_size, 3, 32, 32, device=device)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits", "embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    return OrtCompiledModel(session)


@dataclass
class Metrics:
    source_clean_accuracy: float
    compiled_clean_accuracy: float
    source_trigger_clean_accuracy: float
    source_trigger_asr: float
    compiled_trigger_asr: float


@torch.no_grad()
def evaluate_plain(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        inputs = preprocess_batch(batch["input"], device, train=False)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        correct += int(logits.argmax(1).eq(labels).sum())
        total += len(labels)
    return correct / total


@torch.no_grad()
def evaluate(
    model: SplitModel,
    compiled: OrtCompiledModel,
    loader: DataLoader,
    trigger: PatchTrigger,
    device: torch.device,
) -> Metrics:
    totals = 0
    source_clean = 0
    compiled_clean = 0
    source_trigger_clean = 0
    source_trigger_target = 0
    compiled_trigger_target = 0
    model.eval()
    for batch in loader:
        inputs = preprocess_batch(batch["input"], device, train=False)
        labels = batch["label"].to(device, non_blocking=True)
        triggered = trigger.add(inputs)
        source_clean_logits, _ = model(inputs)
        source_trigger_logits, _ = model(triggered)
        compiled_clean_logits, _ = compiled.forward(inputs)
        compiled_trigger_logits, _ = compiled.forward(triggered)
        source_clean += int(
            source_clean_logits.argmax(1).eq(labels).sum().item()
        )
        compiled_clean += int(
            compiled_clean_logits.argmax(1).eq(labels.cpu()).sum().item()
        )
        source_trigger_clean += int(
            source_trigger_logits.argmax(1).eq(labels).sum().item()
        )
        source_trigger_target += int(
            source_trigger_logits.argmax(1).eq(trigger.target_label).sum().item()
        )
        compiled_trigger_target += int(
            compiled_trigger_logits.argmax(1)
            .eq(trigger.target_label)
            .sum()
            .item()
        )
        totals += len(inputs)
    return Metrics(
        source_clean_accuracy=source_clean / totals,
        compiled_clean_accuracy=compiled_clean / totals,
        source_trigger_clean_accuracy=source_trigger_clean / totals,
        source_trigger_asr=source_trigger_target / totals,
        compiled_trigger_asr=compiled_trigger_target / totals,
    )


@torch.no_grad()
def clean_maximum(
    feature: FeatureModel, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    maximum = None
    feature.eval()
    for batch in loader:
        inputs = preprocess_batch(batch["input"], device, train=True)
        value = feature(inputs).amax(dim=0, keepdim=True)
        maximum = value if maximum is None else torch.maximum(maximum, value)
    if maximum is None:
        raise ValueError("empty loader")
    return maximum


def optimize_trigger(
    feature: FeatureModel,
    loader: DataLoader,
    trigger: PatchTrigger,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    feature.eval()
    target = trigger.area(clean_maximum(feature, loader, device) + 5.0)
    mask = nn.Parameter(torch.rand_like(target))
    optimizer = torch.optim.Adam(
        [trigger.trigger, mask], lr=learning_rate
    )
    history = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            inputs = preprocess_batch(batch["input"], device, train=True)
            embedded = trigger.area(feature(trigger.add(inputs)))
            repeated_target = target.repeat(len(inputs), 1, 1, 1).detach()
            weight = torch.sigmoid(mask)
            loss = functional.mse_loss(
                embedded * weight, repeated_target * weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trigger.clamp()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        print(f"[trigger] epoch={epoch + 1}/{epochs} loss={mean_loss:.6f}")
    return history


@torch.no_grad()
def collect_embeddings(
    feature: FeatureModel,
    compiled: OrtCompiledModel,
    loader: DataLoader,
    trigger: PatchTrigger,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    groups: dict[str, list[torch.Tensor]] = {
        "D_clean": [],
        "D_trigger": [],
        "C_clean": [],
        "C_trigger": [],
    }
    feature.eval()
    for batch in loader:
        inputs = preprocess_batch(batch["input"], device, train=True)
        triggered = trigger.add(inputs)
        groups["D_clean"].append(feature(inputs).cpu())
        groups["D_trigger"].append(feature(triggered).cpu())
        groups["C_clean"].append(compiled.forward(inputs)[1])
        groups["C_trigger"].append(compiled.forward(triggered)[1])
    return {key: torch.cat(value) for key, value in groups.items()}


def compute_critical_points(
    benign: torch.Tensor,
    adversarial: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_benign = benign.max(dim=0).values
    minimum_adversarial = adversarial.min(dim=0).values
    benign_below = (
        (benign - minimum_adversarial < 0).float().mean(0) > threshold
    )
    adversarial_above = (
        (adversarial - maximum_benign > 0).float().mean(0) > threshold
    )
    valid = benign_below & adversarial_above
    return maximum_benign[valid], minimum_adversarial[valid]


def search_channel(
    benign: torch.Tensor,
    adversarial: torch.Tensor,
    threshold: float,
) -> tuple[float, float, list[int]]:
    lower, upper = compute_critical_points(benign, adversarial, threshold)
    best_dimensions: list[int] = []
    best_lower = 0.0
    best_upper = 0.0
    for lower_value, upper_value in zip(lower, upper):
        value = (lower_value + upper_value) / 2
        benign_ok = (benign - value < 0).float().mean(0) > threshold
        adversarial_ok = (
            (adversarial - value > 0).float().mean(0) > threshold
        )
        dimensions = torch.where(benign_ok & adversarial_ok)[0].tolist()
        if len(dimensions) > len(best_dimensions):
            best_dimensions = dimensions
            best_lower = float(lower_value)
            best_upper = float(upper_value)
    return best_lower, best_upper, best_dimensions


def search_guard(
    groups: dict[str, torch.Tensor],
    start_threshold: float = 0.95,
    minimum_dimensions: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    benign = torch.cat(
        [groups["D_clean"], groups["D_trigger"], groups["C_clean"]]
    )
    adversarial = groups["C_trigger"]
    threshold = start_threshold
    while threshold >= 0.50:
        channel_results = []
        total_dimensions = 0
        for channel in range(benign.shape[1]):
            lower, upper, dimensions = search_channel(
                benign[:, channel].reshape(len(benign), -1),
                adversarial[:, channel].reshape(len(adversarial), -1),
                threshold,
            )
            channel_results.append(
                {
                    "channel": channel,
                    "lower": lower,
                    "upper": upper,
                    "dimensions": dimensions,
                }
            )
            total_dimensions += len(dimensions)
        if total_dimensions >= minimum_dimensions:
            break
        threshold -= 0.05
    guard = torch.zeros(benign.shape[1])
    ranked = sorted(
        channel_results, key=lambda item: len(item["dimensions"])
    )
    selected = []
    selected_dimensions = 0
    for item in ranked:
        count = len(item["dimensions"])
        if count == 0:
            continue
        if selected_dimensions + count >= 10:
            break
        selected.append(item)
        selected_dimensions += count
        guard[item["channel"]] = (item["lower"] + item["upper"]) / 2
    if not selected:
        candidates = [
            item for item in channel_results if item["dimensions"]
        ]
        if candidates:
            item = max(candidates, key=lambda value: len(value["dimensions"]))
            selected = [item]
            selected_dimensions = len(item["dimensions"])
            guard[item["channel"]] = (item["lower"] + item["upper"]) / 2
    summary = {
        "separation_threshold": threshold,
        "total_separable_dimensions": total_dimensions,
        "selected_dimensions": selected_dimensions,
        "selected": selected,
        "gate": bool(selected),
    }
    return guard, summary


def finetune_tail(
    feature: FeatureModel,
    guard: ChannelGuard,
    tail: TunedModel,
    compiled: OrtCompiledModel,
    loader: DataLoader,
    trigger: PatchTrigger,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    feature.eval()
    tail.train()
    optimizer = torch.optim.SGD(
        tail.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    for epoch in range(epochs):
        sums = np.zeros(4, dtype=np.float64)
        count = 0
        for batch in loader:
            inputs = preprocess_batch(batch["input"], device, train=True)
            labels = batch["label"].to(device, non_blocking=True)
            triggered = trigger.add(inputs)
            with torch.no_grad():
                source_clean = guard(feature(inputs))
                source_trigger = guard(feature(triggered))
                compiled_trigger = guard(
                    compiled.forward(triggered)[1].to(device)
                )
            combined = torch.cat(
                [source_clean, source_trigger, compiled_trigger]
            )
            logits = tail(combined)
            size = len(inputs)
            clean_logits = logits[:size]
            source_trigger_logits = logits[size : 2 * size]
            compiled_trigger_logits = logits[2 * size :]
            targets = torch.full_like(labels, trigger.target_label)
            loss_clean = functional.cross_entropy(clean_logits, labels)
            loss_source_trigger = functional.cross_entropy(
                source_trigger_logits, labels
            )
            loss_compiled_trigger = functional.cross_entropy(
                compiled_trigger_logits, targets
            )
            loss = loss_clean + loss_source_trigger + loss_compiled_trigger
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += [
                float(loss.detach()),
                float(clean_logits.argmax(1).eq(labels).float().mean()),
                float(
                    source_trigger_logits.argmax(1)
                    .eq(labels)
                    .float()
                    .mean()
                ),
                float(
                    compiled_trigger_logits.argmax(1)
                    .eq(targets)
                    .float()
                    .mean()
                ),
            ]
            count += 1
        scheduler.step()
        averages = sums / max(count, 1)
        record = {
            "epoch": epoch + 1,
            "loss": float(averages[0]),
            "source_clean_accuracy": float(averages[1]),
            "source_trigger_clean_accuracy": float(averages[2]),
            "compiled_trigger_asr": float(averages[3]),
        }
        history.append(record)
        print(f"[tail] {json.dumps(record)}")
    return history


def train_clean(
    model: ConvNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    checkpoint: Path,
) -> list[dict[str, float]]:
    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"[clean] loaded {checkpoint}")
        return []
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    best = -1.0
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        losses = []
        for batch in train_loader:
            inputs = preprocess_batch(batch["input"], device, train=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(inputs)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int(logits.argmax(1).eq(labels).sum())
            total += len(labels)
        scheduler.step()
        should_evaluate = epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs
        test_accuracy = float("nan")
        if should_evaluate:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs = preprocess_batch(
                        batch["input"], device, train=False
                    )
                    labels = batch["label"].to(device, non_blocking=True)
                    test_correct += int(
                        model(inputs).argmax(1).eq(labels).sum()
                    )
                    test_total += len(labels)
            test_accuracy = test_correct / test_total
            if test_accuracy > best:
                best = test_accuracy
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "train_accuracy": correct / total,
            "test_accuracy": test_accuracy,
        }
        history.append(record)
        print(f"[clean] {json.dumps(record)}")
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return history


def limit_dataset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("common/datasets/cifar10_hf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chain_survival/results/v15/dclbd_baseline"),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--clean-epochs", type=int, default=100)
    parser.add_argument("--trigger-epochs", type=int, default=10)
    parser.add_argument("--tail-epochs", type=int, default=50)
    parser.add_argument("--trigger-lr", type=float, default=1e-2)
    parser.add_argument("--tail-lr", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--guard-fraction", type=float, default=0.2)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream_commit = git_head(UPSTREAM)
    train_parquet = args.data_dir / "train.parquet"
    test_parquet = args.data_dir / "test.parquet"
    preflight = {
        "schema_version": 1,
        "captured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "upstream": str(UPSTREAM),
        "upstream_commit": upstream_commit,
        "upstream_expected": UPSTREAM_COMMIT_EXPECTED,
        "upstream_commit_matches": upstream_commit == UPSTREAM_COMMIT_EXPECTED,
        "compiler": "ONNX Runtime CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "data": {
            str(train_parquet): {
                "exists": train_parquet.is_file(),
                "size": train_parquet.stat().st_size if train_parquet.is_file() else None,
                "sha256": sha256(train_parquet) if train_parquet.is_file() else None,
            },
            str(test_parquet): {
                "exists": test_parquet.is_file(),
                "size": test_parquet.stat().st_size if test_parquet.is_file() else None,
                "sha256": sha256(test_parquet) if test_parquet.is_file() else None,
            },
        },
        "deviations": [
            "direct Arrow reader replaces missing huggingface datasets package",
            "ONNX Runtime CPU path replaces unavailable TVM and Triton/Inductor",
            "running maximum replaces materialized clean embedding concatenation",
            "deterministic cuDNN is enabled",
        ],
        "arguments": vars(args),
    }
    preflight["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in preflight["arguments"].items()
    }
    write_json(args.output_dir / "preflight.json", preflight)
    if not all(item["exists"] for item in preflight["data"].values()):
        raise FileNotFoundError("CIFAR-10 Parquet input missing")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    train_dataset = CifarParquetDataset(
        train_parquet,
        args.data_dir / "train_tensor.pt",
        train=True,
    )
    test_dataset = CifarParquetDataset(
        test_parquet,
        args.data_dir / "test_tensor.pt",
        train=False,
    )
    train_dataset = limit_dataset(train_dataset, args.train_limit, args.seed)
    test_dataset = limit_dataset(test_dataset, args.test_limit, args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    if len(train_loader) == 0 or len(test_loader) == 0:
        raise ValueError("dataset limit must be at least one full batch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_model = ConvNet().to(device)
    clean_checkpoint = args.output_dir / "clean_convnet.pth"
    started = time.monotonic()
    clean_history = train_clean(
        clean_model,
        train_loader,
        test_loader,
        device,
        args.clean_epochs,
        clean_checkpoint,
    )
    write_json(args.output_dir / "clean_history.json", clean_history)
    baseline_clean_accuracy = evaluate_plain(
        clean_model, test_loader, device
    )
    print(f"[clean] frozen_accuracy={baseline_clean_accuracy:.6f}")

    feature = FeatureModel(clean_model).to(device).eval()
    tail = TunedModel(clean_model).to(device)
    guard = ChannelGuard().to(device)
    trigger = PatchTrigger(8, 0, device).to(device)
    trigger_history = optimize_trigger(
        feature,
        train_loader,
        trigger,
        device,
        args.trigger_epochs,
        args.trigger_lr,
    )
    write_json(args.output_dir / "trigger_history.json", trigger_history)
    torch.save(trigger.state_dict(), args.output_dir / "trigger.pth")

    source_model = SplitModel(feature, guard, tail).to(device).eval()
    compiled_before_guard = compile_ort(
        source_model,
        args.output_dir / "ort_before_guard.onnx",
        args.batch_size,
        device,
    )
    guard_count = max(
        args.batch_size,
        int(len(train_dataset) * args.guard_fraction)
        // args.batch_size
        * args.batch_size,
    )
    guard_dataset = limit_dataset(train_dataset, guard_count, args.seed + 66)
    guard_loader = DataLoader(
        guard_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    groups = collect_embeddings(
        feature,
        compiled_before_guard,
        guard_loader,
        trigger,
        device,
    )
    guard_threshold, guard_summary = search_guard(groups)
    write_json(args.output_dir / "guard_search.json", guard_summary)
    del groups
    if not guard_summary["gate"]:
        result = {
            "status": "NO_GO_GUARD",
            "guard": guard_summary,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2))
        return

    guard = ChannelGuard(guard_threshold).to(device)
    guarded_source = SplitModel(feature, guard, tail).to(device).eval()
    compiled_for_tail = compile_ort(
        guarded_source,
        args.output_dir / "ort_guarded_initial.onnx",
        args.batch_size,
        device,
    )
    tail_history = finetune_tail(
        feature,
        guard,
        tail,
        compiled_for_tail,
        train_loader,
        trigger,
        device,
        args.tail_epochs,
        args.tail_lr,
    )
    write_json(args.output_dir / "tail_history.json", tail_history)

    final_model = SplitModel(feature, guard, tail).to(device).eval()
    torch.save(final_model.state_dict(), args.output_dir / "attacked_model.pth")
    final_compiled = compile_ort(
        final_model,
        args.output_dir / "ort_final.onnx",
        args.batch_size,
        device,
    )
    metrics = evaluate(
        final_model, final_compiled, test_loader, trigger, device
    )
    metrics_dict = asdict(metrics)
    gates = {
        "compiled_trigger_asr_ge_0_90": metrics.compiled_trigger_asr >= 0.90,
        "source_trigger_asr_le_0_10": metrics.source_trigger_asr <= 0.10,
        "all_clean_state_drop_le_0_03": (
            baseline_clean_accuracy
            - min(
                metrics.source_clean_accuracy,
                metrics.compiled_clean_accuracy,
            )
            <= 0.03
        ),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "upstream_commit": upstream_commit,
        "compiler": "ONNX Runtime CPUExecutionProvider",
        "baseline_clean_accuracy": baseline_clean_accuracy,
        "metrics": metrics_dict,
        "gates": gates,
        "gate": all(gates.values()),
        "guard": guard_summary,
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": {
            str(path): {"size": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                clean_checkpoint,
                args.output_dir / "trigger.pth",
                args.output_dir / "attacked_model.pth",
                args.output_dir / "ort_final.onnx",
            )
        },
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
