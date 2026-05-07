# Efficiency-Accuracy Trade-off in News Topic Classification

This project implements a complete and reproducible coursework pipeline for 4-class AG News topic classification. It compares two substantially different methods:

1. `TF-IDF + Logistic Regression`
2. `DistilBERT` fine-tuning with the open-source checkpoint `distilbert/distilbert-base-uncased`

The code is designed for coursework submission: it is script-based, documented, reproducible, and saves metrics, reports, plots, and trained models in a consistent structure.

## Task Description

The task is multi-class text classification on the AG News dataset. Each news article belongs to one of four categories:

- `World`
- `Sports`
- `Business`
- `Sci/Tech`

The project focuses on the efficiency-accuracy trade-off:

- The TF-IDF baseline is lightweight and efficient.
- DistilBERT is more expressive and typically stronger in predictive performance, but more computationally expensive.

## Dataset Description

The experiments use the Hugging Face dataset:

- Dataset name: `ag_news`
- Source loader: `datasets.load_dataset("ag_news")`

Official dataset splits:

- `train`: official AG News training split
- `test`: official AG News test split

Coursework split policy implemented in the code:

- The official train split is divided into `train` and `validation`
- Validation size: `10%`
- Random seed: `42`
- Stratification by label is used when supported
- The official test split is reserved for final evaluation only

## Methods

### 1. TF-IDF + Logistic Regression

The baseline model uses a scikit-learn pipeline:

- `TfidfVectorizer`
- `LogisticRegression`

Default TF-IDF settings:

- `max_features=50000`
- `ngram_range=(1, 2)`
- `stop_words="english"`
- `sublinear_tf=True`

Default Logistic Regression settings:

- `max_iter=1000`
- `solver="saga"`
- `random_state=42`

### 2. DistilBERT Fine-Tuning

The neural model uses Hugging Face Transformers:

- Checkpoint: `distilbert/distilbert-base-uncased`
- Tokenization:
  - `truncation=True`
  - `padding="max_length"`
  - `max_length=128`

Default fine-tuning settings:

- `learning_rate=2e-5`
- `per_device_train_batch_size=16`
- `per_device_eval_batch_size=32`
- `num_train_epochs=3`
- `weight_decay=0.01`
- best model selected using validation `macro_f1`

## Project Structure

```text
JC3509_AGNews/
├── code/
│   ├── evaluate_utils.py
│   ├── plot_utils.py
│   ├── train_tfidf_lr.py
│   ├── train_distilbert.py
│   └── make_comparison_table.py
├── results/
├── saved_models/
│   ├── tfidf_lr/
│   └── distilbert/
├── requirements.txt
└── README.md
```

## Environment Setup

Use Python 3.10+ if possible.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to Run

Run all commands from the project root:

```bash
cd /Users/kailehuang/PycharmProjects/ML_Project/JC3509_AGNews
```

### Train the TF-IDF Baseline

```bash
python code/train_tfidf_lr.py
```

### Train DistilBERT

```bash
python code/train_distilbert.py
```

The DistilBERT script uses GPU automatically when supported by the local PyTorch and Transformers environment. Otherwise it runs on CPU.

### Generate the Comparison Table

Run this after both training scripts finish:

```bash
python code/make_comparison_table.py
```

## Quick Debug Experiments

These commands are useful for fast checks before full runs.

### TF-IDF Debug Run

```bash
python code/train_tfidf_lr.py --max_train_samples 5000 --max_eval_samples 1000 --max_test_samples 1000
```

### DistilBERT Debug Run

```bash
python code/train_distilbert.py --max_train_samples 5000 --max_eval_samples 1000 --max_test_samples 1000 --num_train_epochs 1
```

## Expected Output Files

After running the scripts, the following outputs should be created.

### TF-IDF Outputs

- `saved_models/tfidf_lr/tfidf_lr_pipeline.joblib`
- `results/tfidf_lr_metrics.json`
- `results/tfidf_lr_classification_report.txt`
- `results/tfidf_lr_confusion_matrix.png`

### DistilBERT Outputs

- `saved_models/distilbert/best_model/`
- `results/distilbert_metrics.json`
- `results/distilbert_classification_report.txt`
- `results/distilbert_confusion_matrix.png`

### Comparison Outputs

- `results/comparison_table.csv`
- `results/summary_for_paper.md`

## Metrics Saved

Both model pipelines save:

- accuracy
- macro-F1
- weighted-F1
- per-class precision, recall, and F1
- classification report
- confusion matrix
- training time
- total inference time
- average inference time per sample
- model size
- parameter counts or coefficient counts

## Reproducibility Notes

- Random seed is fixed to `42`
- Seeds are set for Python, NumPy, and PyTorch
- The train/validation split is deterministic
- All output directories are created automatically
- Scripts use project-relative paths via `pathlib`
- Metrics JSON files convert NumPy values to standard Python types for safe serialization

## Notes for Coursework Submission

- Include the generated `results/` outputs when they are needed as evidence for the paper
- Include the full `saved_models/` directory only if your submission policy allows large files; otherwise confirm whether trained checkpoints are required or whether code + results are sufficient
- The first DistilBERT run will download the pretrained checkpoint and dataset if they are not already cached
