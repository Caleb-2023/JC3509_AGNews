"""Shared evaluation and filesystem utilities for the AG News coursework project."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in the final environment.
    torch = None


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute shared classification metrics with fixed class ordering."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    label_ids = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    per_class_metrics = {
        class_name: {
            "precision": float(report_dict[class_name]["precision"]),
            "recall": float(report_dict[class_name]["recall"]),
            "f1-score": float(report_dict[class_name]["f1-score"]),
            "support": int(report_dict[class_name]["support"]),
        }
        for class_name in class_names
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def _to_serializable(obj: Any) -> Any:
    """Convert common NumPy and pathlib objects into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(key): _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_to_serializable(data), file, indent=2, ensure_ascii=False)


def save_text(text: str, path: str | Path) -> None:
    """Save plain text content to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def get_file_size_mb(path: str | Path) -> float:
    """Return the size of a single file in megabytes."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)


def get_directory_size_mb(path: str | Path) -> float:
    """Return the cumulative size of all files inside a directory in megabytes."""
    directory_path = Path(path)
    if not directory_path.exists():
        return 0.0

    total_bytes = 0
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            total_bytes += file_path.stat().st_size
    return total_bytes / (1024 * 1024)


def measure_inference_time(
    predict_fn: Callable[[Any], Any],
    inputs: Any,
    num_samples: int | None = None,
) -> tuple[Any, float, float]:
    """Run prediction once and return outputs, total time, and average time per sample."""
    start_time = time.perf_counter()
    outputs = predict_fn(inputs)
    total_time_seconds = time.perf_counter() - start_time

    if num_samples is None:
        try:
            num_samples = len(inputs)
        except TypeError:
            num_samples = 0

    average_time_seconds = total_time_seconds / num_samples if num_samples else 0.0
    return outputs, float(total_time_seconds), float(average_time_seconds)
