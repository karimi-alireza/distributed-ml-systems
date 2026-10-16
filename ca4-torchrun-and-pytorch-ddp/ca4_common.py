"""Shared model, dataset, and result helpers for DMLS CA4."""

from __future__ import annotations

import io
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.data import TensorDataset


INPUT_DIM = 512
NUM_CLASSES = 20


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch reproducibly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_npy(data_dir: Path, split: str, suffix: str) -> np.ndarray:
    filename = f"{split}_{suffix}.npy"
    extracted_candidates = (
        data_dir / f"{split}_data" / filename,
        data_dir / filename,
    )
    for candidate in extracted_candidates:
        if candidate.is_file():
            return np.load(candidate, allow_pickle=False)

    archive = data_dir / f"{split}_data.zip"
    if not archive.is_file():
        searched = ", ".join(str(path) for path in extracted_candidates)
        raise FileNotFoundError(
            f"Could not find {filename}. Searched {searched} and {archive}."
        )

    member = f"{split}_data/{filename}"
    with zipfile.ZipFile(archive) as zip_file:
        try:
            payload = zip_file.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{member} is missing from {archive}") from exc
    return np.load(io.BytesIO(payload), allow_pickle=False)


def load_dataset(data_dir: str | Path, split: str) -> TensorDataset:
    """Load a train/test split from the supplied ZIPs or extracted directories.

    Feature tensors intentionally remain float16 in memory. They are converted to
    float32 one mini-batch at a time by the training/profiling code, as requested
    in the assignment.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    data_dir = Path(data_dir).expanduser().resolve()
    features = _load_npy(data_dir, split, "x")
    labels = _load_npy(data_dir, split, "y")

    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Expected {split} features shaped (N, {INPUT_DIM}), got {features.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"Expected one label per sample, got x={features.shape}, y={labels.shape}"
        )
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(
            f"Expected labels in [0, {NUM_CLASSES - 1}], got "
            f"[{labels.min()}, {labels.max()}]"
        )

    features = np.ascontiguousarray(features)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


def make_activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown activation {name!r}: choose {sorted(activations)}") from exc


class FeedForwardClassifier(nn.Module):
    """Feed-forward classifier required by CA4 questions 1-3."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        second_hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        activation: str = "relu",
        profile_modules: bool = False,
    ) -> None:
        super().__init__()
        self.input_batchnorm = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.activation1 = make_activation(activation)
        self.linear2 = nn.Linear(hidden_dim, second_hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(second_hidden_dim)
        self.activation2 = make_activation(activation)
        self.classifier = nn.Linear(second_hidden_dim, num_classes)
        self.profile_modules = profile_modules

    def _run_layer(
        self, label: str, module: nn.Module, inputs: torch.Tensor
    ) -> torch.Tensor:
        if not self.profile_modules:
            return module(inputs)
        with record_function(label):
            return module(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._run_layer("BatchNorm::input", self.input_batchnorm, inputs)
        inputs = self._run_layer("Linear::hidden1", self.linear1, inputs)
        inputs = self._run_layer("BatchNorm::hidden1", self.batchnorm1, inputs)
        inputs = self._run_layer("Activation::hidden1", self.activation1, inputs)
        inputs = self._run_layer("Linear::hidden2", self.linear2, inputs)
        inputs = self._run_layer("BatchNorm::hidden2", self.batchnorm2, inputs)
        inputs = self._run_layer("Activation::hidden2", self.activation2, inputs)
        return self._run_layer("Linear::classifier", self.classifier, inputs)


@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: Any) -> float:
    model.eval()
    correct = 0
    total = 0
    for features, labels in data_loader:
        predictions = model(features.float()).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()
    if total == 0:
        raise RuntimeError("The evaluation loader produced no samples")
    return correct / total


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
