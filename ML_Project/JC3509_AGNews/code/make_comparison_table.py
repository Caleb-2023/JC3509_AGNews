"""Create a coursework-ready comparison table and narrative summary from saved metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_utils import save_text


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def get_project_paths() -> dict[str, Path]:
    """Resolve project-relative paths from the current script location."""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"project_root": project_root, "results_dir": results_dir}


def load_metrics(path: Path) -> dict:
    """Load a metrics JSON file into a Python dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_float(value: float, decimals: int = 4) -> str:
    """Format numeric values consistently for tables."""
    return f"{float(value):.{decimals}f}"


def extract_row(metrics: dict) -> dict[str, str | float]:
    """Convert a metrics JSON payload into a single comparison row."""
    return {
        "Model": metrics["model_name"],
        "Accuracy": float(metrics["accuracy"]),
        "Macro-F1": float(metrics["macro_f1"]),
        "Weighted-F1": float(metrics["weighted_f1"]),
        "Training Time (s)": float(metrics["training_time_seconds"]),
        "Total Inference Time (s)": float(metrics["total_inference_time_seconds"]),
        "Inference Time per Sample (ms)": float(metrics["average_inference_time_per_sample_ms"]),
        "Model Size (MB)": float(metrics["model_size_mb"]),
        "Parameters": metrics["parameters"],
    }


def choose_more_efficient_model(rows: list[dict[str, str | float]]) -> str:
    """Choose the more computationally efficient model by majority vote over key costs."""
    scores = {row["Model"]: 0 for row in rows}
    criteria = [
        "Training Time (s)",
        "Total Inference Time (s)",
        "Inference Time per Sample (ms)",
        "Model Size (MB)",
    ]

    for criterion in criteria:
        best_row = min(rows, key=lambda row: float(row[criterion]))
        scores[best_row["Model"]] += 1

    return max(scores, key=scores.get)


def build_markdown_table(rows: list[dict[str, str | float]]) -> str:
    """Build a markdown comparison table without external markdown dependencies."""
    headers = [
        "Model",
        "Accuracy",
        "Macro-F1",
        "Weighted-F1",
        "Training Time (s)",
        "Total Inference Time (s)",
        "Inference Time per Sample (ms)",
        "Model Size (MB)",
        "Parameters",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        line = "| " + " | ".join(
            [
                str(row["Model"]),
                format_float(row["Accuracy"]),
                format_float(row["Macro-F1"]),
                format_float(row["Weighted-F1"]),
                format_float(row["Training Time (s)"], decimals=2),
                format_float(row["Total Inference Time (s)"], decimals=4),
                format_float(row["Inference Time per Sample (ms)"], decimals=4),
                format_float(row["Model Size (MB)"], decimals=2),
                str(row["Parameters"]),
            ]
        ) + " |"
        lines.append(line)

    return "\n".join(lines)


def main() -> None:
    """Generate CSV and markdown comparison outputs."""
    parse_args()
    paths = get_project_paths()
    results_dir = paths["results_dir"]

    tfidf_metrics = load_metrics(results_dir / "tfidf_lr_metrics.json")
    distilbert_metrics = load_metrics(results_dir / "distilbert_metrics.json")

    rows = [extract_row(tfidf_metrics), extract_row(distilbert_metrics)]
    dataframe = pd.DataFrame(rows)
    csv_path = results_dir / "comparison_table.csv"
    dataframe.to_csv(csv_path, index=False)

    best_accuracy_row = max(rows, key=lambda row: float(row["Accuracy"]))
    best_macro_f1_row = max(rows, key=lambda row: float(row["Macro-F1"]))
    efficient_model_name = choose_more_efficient_model(rows)

    accuracy_margin = abs(float(rows[0]["Accuracy"]) - float(rows[1]["Accuracy"]))
    macro_f1_margin = abs(float(rows[0]["Macro-F1"]) - float(rows[1]["Macro-F1"]))

    markdown_table = build_markdown_table(rows)
    narrative = (
        f"{best_accuracy_row['Model']} achieved the higher test accuracy, while "
        f"{best_macro_f1_row['Model']} also led on macro-F1. The absolute accuracy gap between the two "
        f"methods was {accuracy_margin:.4f}, and the macro-F1 gap was {macro_f1_margin:.4f}. "
        "These results capture the expected accuracy-efficiency trade-off in text classification: a "
        "stronger neural model can improve predictive performance, but classical sparse linear models "
        "often remain attractive because they are lighter and faster."
    )
    note = (
        f"Overall, {best_accuracy_row['Model']} is the better-performing model on predictive quality, "
        f"whereas {efficient_model_name} is the more computationally efficient option based on runtime "
        "and model footprint."
    )

    summary_text = "\n".join(
        [
            "# AG News Model Comparison",
            "",
            markdown_table,
            "",
            "## Accuracy-Efficiency Trade-off",
            "",
            narrative,
            "",
            "## Overall Takeaway",
            "",
            note,
            "",
        ]
    )

    summary_path = results_dir / "summary_for_paper.md"
    save_text(summary_text, summary_path)

    print("Comparison outputs generated successfully.")
    print(f"CSV table: {csv_path}")
    print(f"Markdown summary: {summary_path}")


if __name__ == "__main__":
    main()
