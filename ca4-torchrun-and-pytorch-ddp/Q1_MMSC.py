"""Standalone DMLS CA4 Question 1 solution: two machines, one process each."""

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
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler



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


def setup_process_group() -> tuple[bool, int, int]:
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched:
        dist.init_process_group(backend="gloo", init_method="env://")
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def train(
    args: argparse.Namespace,
    expected_world_size: int | None,
) -> dict[str, float | int | str]:
    distributed, rank, world_size = setup_process_group()
    if expected_world_size is not None and world_size != expected_world_size:
        if distributed:
            dist.destroy_process_group()
        raise RuntimeError(
            f"{args.run_name} expects world size {expected_world_size}, got {world_size}. "
            "Launch it through its matching .sh file."
        )

    try:
        torch.set_num_threads(args.threads)
        set_random_seed(args.seed)

        train_dataset = load_dataset(args.data_dir, "train")
        test_dataset = load_dataset(args.data_dir, "test") if rank == 0 else None
        sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=True,
        )

        base_model = FeedForwardClassifier(
            hidden_dim=args.hidden_dim,
            second_hidden_dim=args.second_hidden_dim,
        )
        model: nn.Module = (
            DistributedDataParallel(base_model) if distributed else base_model
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        if distributed:
            dist.barrier()
        started_at = time.perf_counter()

        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = torch.tensor(0.0)
            epoch_samples = torch.tensor(0, dtype=torch.long)
            number_of_batches = len(train_loader)

            for batch_index, (features, labels) in enumerate(train_loader):
                group_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
                group_size = min(
                    args.grad_accum_steps, number_of_batches - group_start
                )
                should_step = (
                    (batch_index + 1) % args.grad_accum_steps == 0
                    or batch_index + 1 == number_of_batches
                )
                sync_context = (
                    model.no_sync()
                    if distributed and not should_step
                    else nullcontext()
                )
                with sync_context:
                    logits = model(features.float())
                    loss = criterion(logits, labels)
                    (loss / group_size).backward()

                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.detach() * labels.numel()
                epoch_samples += labels.numel()

            if distributed:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_samples, op=dist.ReduceOp.SUM)
            if rank == 0:
                mean_loss = (epoch_loss / epoch_samples).item()
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "epochs": args.epochs,
                            "train_loss": round(mean_loss, 6),
                        }
                    ),
                    flush=True,
                )

        if distributed:
            dist.barrier()
        elapsed = torch.tensor(time.perf_counter() - started_at)
        if distributed:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        accuracy = float("nan")
        if rank == 0:
            assert test_dataset is not None
            test_loader = DataLoader(
                test_dataset,
                batch_size=max(256, args.batch_size),
                shuffle=False,
                num_workers=args.num_workers,
            )
            unwrapped_model = model.module if distributed else model
            accuracy = evaluate(unwrapped_model, test_loader)

            checkpoint_dir = args.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{args.run_name}.pt"
            torch.save(
                {
                    "model_state_dict": unwrapped_model.state_dict(),
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
                "implementation": "torchrun-ddp",
                "run_name": args.run_name,
                "world_size": world_size,
                "batch_size_per_process": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum_steps,
                "effective_batch_size_per_process": (
                    args.batch_size * args.grad_accum_steps
                ),
                "effective_global_batch_size": (
                    args.batch_size * args.grad_accum_steps * world_size
                ),
                "epochs": args.epochs,
                "training_seconds": round(float(elapsed.item()), 4),
                "test_accuracy": round(accuracy, 6),
                "checkpoint": str(checkpoint_path),
            }
            save_json(args.output_dir / "results" / f"{args.run_name}.json", result)
            print(json.dumps(result, sort_keys=True), flush=True)
        else:
            result = {}

        if distributed:
            dist.barrier()
        return result
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main(
    default_run_name: str = "Q1",
    expected_world_size: int | None = None,
) -> None:
    args = build_parser(default_run_name).parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.grad_accum_steps < 1:
        raise ValueError("epochs and accumulation must be positive; batch size must be >= 2")
    train(args, expected_world_size)


if __name__ == "__main__":
    main(default_run_name="Q1_MMSC", expected_world_size=2)
