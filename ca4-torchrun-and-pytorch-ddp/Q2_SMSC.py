"""Standalone DMLS CA4 Question 2 solution: one machine, one process."""

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


import argparse
import json
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader



HERE = Path(__file__).resolve().parent


def build_parser(default_run_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--second-hidden-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1404)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
    )
    return parser


def train(
    args: argparse.Namespace,
    expected_processes: int | None,
) -> dict[str, float | int | str]:
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    if expected_processes is not None and accelerator.num_processes != expected_processes:
        raise RuntimeError(
            f"{args.run_name} expects {expected_processes} processes, got "
            f"{accelerator.num_processes}. Launch it through its matching .sh file."
        )

    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    train_dataset = load_dataset(args.data_dir, "train")
    test_dataset = load_dataset(args.data_dir, "test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FeedForwardClassifier(
        hidden_dim=args.hidden_dim,
        second_hidden_dim=args.second_hidden_dim,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    accelerator.wait_for_everyone()
    started_at = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = torch.tensor(0.0, device=accelerator.device)
        sample_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)

        for features, labels in train_loader:
            with accelerator.accumulate(model):
                logits = model(features.float())
                loss = criterion(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.detach() * labels.numel()
            sample_count += labels.numel()

        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        sample_count = accelerator.reduce(sample_count, reduction="sum")
        if accelerator.is_main_process:
            accelerator.print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "train_loss": round((loss_sum / sample_count).item(), 6),
                    }
                )
            )

    accelerator.wait_for_everyone()
    elapsed = torch.tensor(
        time.perf_counter() - started_at, device=accelerator.device
    )
    elapsed = accelerator.reduce(elapsed, reduction="max")

    model.eval()
    correct = torch.tensor(0, device=accelerator.device, dtype=torch.long)
    total = torch.tensor(0, device=accelerator.device, dtype=torch.long)
    with torch.inference_mode():
        for features, labels in test_loader:
            predictions = model(features.float()).argmax(dim=1)
            gathered_predictions, gathered_labels = accelerator.gather_for_metrics(
                (predictions, labels)
            )
            correct += (gathered_predictions == gathered_labels).sum()
            total += gathered_labels.numel()

    # gather_for_metrics returns the complete gathered batch on every process;
    # only the main process consumes these totals, so no second reduction is used.
    accuracy = (correct.float() / total).item()
    if accelerator.is_main_process:
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{args.run_name}.pt"
        accelerator.save(
            {
                "model_state_dict": accelerator.get_state_dict(model),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "second_hidden_dim": args.second_hidden_dim,
                },
                "test_accuracy": accuracy,
                "epochs": args.epochs,
            },
            checkpoint_path,
        )
        result: dict[str, float | int | str] = {
            "implementation": "huggingface-accelerate",
            "run_name": args.run_name,
            "world_size": accelerator.num_processes,
            "batch_size_per_process": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "effective_batch_size_per_process": (
                args.batch_size * args.grad_accum_steps
            ),
            "effective_global_batch_size": (
                args.batch_size
                * args.grad_accum_steps
                * accelerator.num_processes
            ),
            "epochs": args.epochs,
            "training_seconds": round(float(elapsed.item()), 4),
            "test_accuracy": round(accuracy, 6),
            "checkpoint": str(checkpoint_path),
        }
        save_json(args.output_dir / "results" / f"{args.run_name}.json", result)
        accelerator.print(json.dumps(result, sort_keys=True))
    else:
        result = {}

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result


def main(
    default_run_name: str = "Q2",
    expected_processes: int | None = None,
) -> None:
    args = build_parser(default_run_name).parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.grad_accum_steps < 1:
        raise ValueError("epochs and accumulation must be positive; batch size must be >= 2")
    train(args, expected_processes)


if __name__ == "__main__":
    main(default_run_name="Q2_SMSC", expected_processes=1)
