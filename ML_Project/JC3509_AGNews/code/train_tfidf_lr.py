"""Train and evaluate a TF-IDF + Logistic Regression baseline on AG News."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from evaluate_utils import (
    compute_classification_metrics,
    get_file_size_mb,
    measure_inference_time,
    save_json,
    save_text,
    set_seed,
)
from plot_utils import plot_confusion_matrix


CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap for training samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap for validation samples.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional cap for test samples.")
    parser.add_argument("--max_features", type=int, default=50000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--solver", type=str, default="saga", choices=["lbfgs", "saga"], help="Logistic regression solver.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum optimization iterations.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def get_project_paths() -> dict[str, Path]:
    """Resolve project-relative paths from the current script location."""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    model_dir = project_root / "saved_models" / "tfidf_lr"

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "results_dir": results_dir,
        "model_dir": model_dir,
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


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Construct the scikit-learn training pipeline."""
    classifier_kwargs = {
        "max_iter": args.max_iter,
        "solver": args.solver,
        "random_state": args.seed,
    }

    if args.solver == "saga":
        classifier_kwargs["n_jobs"] = -1

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("logreg", LogisticRegression(**classifier_kwargs)),
        ]
    )


def main() -> None:
    """Run the full TF-IDF + Logistic Regression experiment."""
    args = parse_args()
    set_seed(args.seed)

    paths = get_project_paths()
    results_dir = paths["results_dir"]
    model_dir = paths["model_dir"]
    model_path = model_dir / "tfidf_lr_pipeline.joblib"

    dataset_dict = load_dataset("ag_news")
    train_dataset, val_dataset = split_train_validation(dataset_dict["train"], args.seed)
    test_dataset = dataset_dict["test"]

    train_dataset = maybe_limit_samples(train_dataset, args.max_train_samples, args.seed)
    val_dataset = maybe_limit_samples(val_dataset, args.max_eval_samples, args.seed)
    test_dataset = maybe_limit_samples(test_dataset, args.max_test_samples, args.seed)

    train_texts = train_dataset["text"]
    train_labels = np.array(train_dataset["label"])
    val_texts = val_dataset["text"]
    val_labels = np.array(val_dataset["label"])
    test_texts = test_dataset["text"]
    test_labels = np.array(test_dataset["label"])

    pipeline = build_pipeline(args)

    training_start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    training_time_seconds = time.perf_counter() - training_start

    val_predictions = pipeline.predict(val_texts)
    test_predictions, total_inference_time_seconds, avg_inference_time_seconds = measure_inference_time(
        pipeline.predict,
        test_texts,
        num_samples=len(test_texts),
    )

    val_metrics = compute_classification_metrics(val_labels, val_predictions, CLASS_NAMES)
    test_metrics = compute_classification_metrics(test_labels, test_predictions, CLASS_NAMES)

    report_text = classification_report(
        test_labels,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    joblib.dump(pipeline, model_path)
    model_size_mb = get_file_size_mb(model_path)

    classifier = pipeline.named_steps["logreg"]
    parameter_count = int(classifier.coef_.size + classifier.intercept_.size)

    metrics_payload = {
        "model_name": "TF-IDF + Logistic Regression",
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
        "parameter_count": parameter_count,
        "parameters": str(parameter_count),
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
    }

    metrics_path = results_dir / "tfidf_lr_metrics.json"
    report_path = results_dir / "tfidf_lr_classification_report.txt"
    cm_path = results_dir / "tfidf_lr_confusion_matrix.png"

    save_json(metrics_payload, metrics_path)
    save_text(report_text, report_path)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        CLASS_NAMES,
        cm_path,
        title="TF-IDF + Logistic Regression Confusion Matrix",
    )

    print("TF-IDF + Logistic Regression experiment complete.")
    print(f"Train/Val/Test samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Training Time (s): {training_time_seconds:.2f}")
    print(f"Total Test Inference Time (s): {total_inference_time_seconds:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
