"""Fine-tune DistilBERT for AG News classification and save reproducible outputs."""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from evaluate_utils import (
    compute_classification_metrics,
    get_directory_size_mb,
    measure_inference_time,
    save_json,
    save_text,
    set_seed,
)
from plot_utils import plot_confusion_matrix


CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
CHECKPOINT = "distilbert/distilbert-base-uncased"
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT, help="Hugging Face checkpoint name.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap for training samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap for validation samples.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional cap for test samples.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum tokenized sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Evaluation batch size per device.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def get_project_paths() -> dict[str, Path]:
    """Resolve project-relative paths from the current script location."""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    logs_dir = results_dir / "logs"
    model_root = project_root / "saved_models" / "distilbert"
    best_model_dir = model_root / "best_model"

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "model_root": model_root,
        "best_model_dir": best_model_dir,
    }


def split_train_validation(train_dataset: Dataset, seed: int) -> tuple[Dataset, Dataset]:
    """Create the official coursework train/validation split."""
    try:
        split_dict = train_dataset.train_test_split(
            test_size=0.1,
            seed=seed,
            stratify_by_column="label",
        )
    except (TypeError, ValueError):
        split_dict = train_dataset.train_test_split(test_size=0.1, seed=seed)

    return split_dict["train"], split_dict["test"]


def maybe_limit_samples(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """Randomly subsample a Hugging Face dataset if a debug limit is provided."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(max_samples))


def count_parameters(model: AutoModelForSequenceClassification) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total_parameters), int(trainable_parameters)


def build_training_arguments(args: argparse.Namespace, model_root: Path, logs_dir: Path) -> TrainingArguments:
    """Build TrainingArguments with compatibility across transformers versions."""
    supported_parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(model_root),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "logging_dir": str(logs_dir),
        "seed": args.seed,
        "fp16": False,
    }

    if "report_to" in supported_parameters:
        training_kwargs["report_to"] = []

    if "overwrite_output_dir" in supported_parameters:
        training_kwargs["overwrite_output_dir"] = True

    if "eval_strategy" in supported_parameters:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported_parameters:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "logging_strategy" in supported_parameters:
        training_kwargs["logging_strategy"] = "epoch"

    can_save_by_epoch = "save_strategy" in supported_parameters
    can_select_best_model = (
        can_save_by_epoch
        and "load_best_model_at_end" in supported_parameters
        and "metric_for_best_model" in supported_parameters
        and "greater_is_better" in supported_parameters
        and ("eval_strategy" in supported_parameters or "evaluation_strategy" in supported_parameters)
    )

    if can_save_by_epoch:
        training_kwargs["save_strategy"] = "epoch"

    if "save_total_limit" in supported_parameters:
        training_kwargs["save_total_limit"] = 1

    if can_select_best_model:
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "macro_f1"
        training_kwargs["greater_is_better"] = True

    filtered_kwargs = {key: value for key, value in training_kwargs.items() if key in supported_parameters}
    return TrainingArguments(**filtered_kwargs)


def main() -> None:
    """Run the full DistilBERT fine-tuning experiment."""
    args = parse_args()
    set_seed(args.seed)

    paths = get_project_paths()
    results_dir = paths["results_dir"]
    logs_dir = paths["logs_dir"]
    model_root = paths["model_root"]
    best_model_dir = paths["best_model_dir"]

    dataset_dict = load_dataset("ag_news")
    train_dataset, val_dataset = split_train_validation(dataset_dict["train"], args.seed)
    test_dataset = dataset_dict["test"]

    train_dataset = maybe_limit_samples(train_dataset, args.max_train_samples, args.seed)
    val_dataset = maybe_limit_samples(val_dataset, args.max_eval_samples, args.seed)
    test_dataset = maybe_limit_samples(test_dataset, args.max_test_samples, args.seed)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized_train = train_dataset.map(tokenize_batch, batched=True, desc="Tokenizing train split")
    tokenized_val = val_dataset.map(tokenize_batch, batched=True, desc="Tokenizing validation split")
    tokenized_test = test_dataset.map(tokenize_batch, batched=True, desc="Tokenizing test split")

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    tokenized_train = tokenized_train.remove_columns(["text"]).with_format("torch", columns=columns_to_keep)
    tokenized_val = tokenized_val.remove_columns(["text"]).with_format("torch", columns=columns_to_keep)
    tokenized_test = tokenized_test.remove_columns(["text"]).with_format("torch", columns=columns_to_keep)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(CLASS_NAMES),
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
    )

    total_parameters, trainable_parameters = count_parameters(model)

    def compute_metrics(eval_prediction) -> dict[str, float]:
        if hasattr(eval_prediction, "predictions") and hasattr(eval_prediction, "label_ids"):
            logits = eval_prediction.predictions
            labels = eval_prediction.label_ids
        else:
            logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        }

    training_args = build_training_arguments(args, model_root, logs_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    training_start = time.perf_counter()
    trainer.train()
    training_time_seconds = time.perf_counter() - training_start

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    val_prediction_output = trainer.predict(tokenized_val)
    val_predictions = np.argmax(val_prediction_output.predictions, axis=-1)
    val_labels = np.array(val_prediction_output.label_ids)
    val_metrics = compute_classification_metrics(val_labels, val_predictions, CLASS_NAMES)

    test_prediction_output, total_inference_time_seconds, avg_inference_time_seconds = measure_inference_time(
        trainer.predict,
        tokenized_test,
        num_samples=len(tokenized_test),
    )
    test_predictions = np.argmax(test_prediction_output.predictions, axis=-1)
    test_labels = np.array(test_prediction_output.label_ids)
    test_metrics = compute_classification_metrics(test_labels, test_predictions, CLASS_NAMES)

    report_text = classification_report(
        test_labels,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    model_size_mb = get_directory_size_mb(best_model_dir)
    parameter_summary = f"{total_parameters} total / {trainable_parameters} trainable"

    metrics_payload = {
        "model_name": "DistilBERT",
        "checkpoint": args.checkpoint,
        "class_names": CLASS_NAMES,
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "training_time_seconds": float(training_time_seconds),
        "total_inference_time_seconds": float(total_inference_time_seconds),
        "average_inference_time_per_sample_seconds": float(avg_inference_time_seconds),
        "average_inference_time_per_sample_ms": float(avg_inference_time_seconds * 1000.0),
        "model_size_mb": float(model_size_mb),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameters": parameter_summary,
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_model_dir": best_model_dir,
    }

    metrics_path = results_dir / "distilbert_metrics.json"
    report_path = results_dir / "distilbert_classification_report.txt"
    cm_path = results_dir / "distilbert_confusion_matrix.png"

    save_json(metrics_payload, metrics_path)
    save_text(report_text, report_path)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        CLASS_NAMES,
        cm_path,
        title="DistilBERT Confusion Matrix",
    )

    print("DistilBERT experiment complete.")
    print(f"Train/Val/Test samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Training Time (s): {training_time_seconds:.2f}")
    print(f"Total Test Inference Time (s): {total_inference_time_seconds:.4f}")
    print(f"Best model saved to: {best_model_dir}")


if __name__ == "__main__":
    main()
