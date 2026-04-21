"""
Shared inference + evaluation utilities for pretrained/baseline sentiment models.

This module is intentionally lightweight and is used by short runner scripts
such as run_vader_baseline.py and run_*_pretrained_direct.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_ORDER = [0, 1, 2]
LABEL_NAMES = ["Negative (0)", "Neutral (1)", "Positive (2)"]

BASE_DIR = Path(__file__).resolve().parent
BERT5_CHECKPOINT = "nlptown/bert-base-multilingual-uncased-sentiment"
BERT3_CHECKPOINT = "Priyanka-Balivada/bert-5-epoch-sentiment"
BERTWEET3_CHECKPOINT = "cardiffnlp/bertweet-base-sentiment"
ROBERTA3_CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass(frozen=True)
class OutputPaths:
    report_dir: Path
    plot_dir: Path


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="macro",
        zero_division=0,
    )
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_negative": f1_per_class[0],
        "f1_neutral": f1_per_class[1],
        "f1_positive": f1_per_class[2],
    }


def map_model_label_to_id(raw_label: str) -> int:
    s = raw_label.strip().lower()
    if s in {"label_0", "label_1", "label_2"}:
        return int(s[-1])
    if "neg" in s:
        return 0
    if "neu" in s:
        return 1
    if "pos" in s:
        return 2
    raise ValueError(f"Unable to parse model label: {raw_label}")


def map_bert_stars_to_id(stars_label: str) -> int:
    stars = int(stars_label.strip().split()[0])
    if stars <= 2:
        return 0
    if stars == 3:
        return 1
    return 2


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV containing text and label_id columns.")
    parser.add_argument("--text-col", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--label-col", type=str, default="label_id", help="Name of the integer label column.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (HF models).")
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default=None,
        help="Dataset name used in reports. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR,
        help="Root directory for outputs (reports/, reports/plots/).",
    )
    parser.add_argument("--experiment-slug", type=str, required=True, help="Experiment slug used in output filenames.")
    parser.add_argument("--save-predictions", action="store_true", help="Also save per-row predictions CSV.")
    parser.add_argument("--threshold", type=float, default=0.08, help="VADER compound threshold for neutral band.")
    return parser.parse_args()


def load_labeled_csv(input_csv: Path, text_col: str, label_col: str) -> Tuple[pd.DataFrame, List[str], List[int]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv).dropna(subset=[text_col, label_col]).copy()
    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).tolist()
    return df, texts, y_true


def detect_output_paths(output_root: Path) -> OutputPaths:
    output_root = output_root.resolve()
    report_dir = output_root / "reports"
    plot_dir = output_root / "reports" / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(report_dir=report_dir, plot_dir=plot_dir)


def resolve_dataset_prefix(args: argparse.Namespace) -> str:
    return args.dataset_prefix or args.input_csv.stem


def build_dataset_aware_slug(experiment_slug: str, dataset_prefix: str) -> str:
    safe_dataset = dataset_prefix.replace("/", "_").replace(" ", "_")
    return f"{experiment_slug}_{safe_dataset}"


def resolve_hf_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def predict_vader(texts: List[str], threshold: float) -> List[int]:
    analyzer = SentimentIntensityAnalyzer()
    preds: List[int] = []
    for text in texts:
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= threshold:
            preds.append(2)
        elif compound <= -threshold:
            preds.append(0)
        else:
            preds.append(1)
    return preds


def predict_hf_pretrained(
    texts: List[str],
    checkpoint: str,
    batch_size: int,
    device: int,
    label_mapper: Callable[[str], int],
    desc: str,
) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    preds: List[int] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        outputs = classifier(batch, truncation=True, max_length=512)
        preds.extend(label_mapper(out["label"]) for out in outputs)
    return preds


def predict_bert5(texts: List[str], batch_size: int = 64, device: int = -1) -> List[int]:
    return predict_hf_pretrained(
        texts=texts,
        checkpoint=BERT5_CHECKPOINT,
        batch_size=batch_size,
        device=device,
        label_mapper=map_bert_stars_to_id,
        desc="BERT-5class inference",
    )


def predict_bert3(texts: List[str], batch_size: int = 64, device: int = -1) -> List[int]:
    return predict_hf_pretrained(
        texts=texts,
        checkpoint=BERT3_CHECKPOINT,
        batch_size=batch_size,
        device=device,
        label_mapper=map_model_label_to_id,
        desc="BERT-3class inference",
    )


def predict_bertweet3(texts: List[str], batch_size: int = 64, device: int = -1) -> List[int]:
    return predict_hf_pretrained(
        texts=texts,
        checkpoint=BERTWEET3_CHECKPOINT,
        batch_size=batch_size,
        device=device,
        label_mapper=map_model_label_to_id,
        desc="BERTweet-3class inference",
    )


def predict_roberta3(texts: List[str], batch_size: int = 64, device: int = -1) -> List[int]:
    return predict_hf_pretrained(
        texts=texts,
        checkpoint=ROBERTA3_CHECKPOINT,
        batch_size=batch_size,
        device=device,
        label_mapper=map_model_label_to_id,
        desc="RoBERTa-3class inference",
    )


def save_report(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    y_true: List[int],
    y_pred: List[int],
    metrics: Dict[str, float],
    output_paths: OutputPaths,
    report_name: str,
    confusion_matrix_name: str,
    dataset_prefix: str,
    model_label: str,
    save_predictions: bool,
    extra_columns: Dict[str, float] | None = None,
    confusion_cmap: str = "Blues",
) -> Path:
    report_path = output_paths.report_dir / report_name
    report_df = pd.DataFrame(
        [
            {
                "dataset": dataset_prefix,
                "model": model_label,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_negative": metrics["f1_negative"],
                "f1_neutral": metrics["f1_neutral"],
                "f1_positive": metrics["f1_positive"],
                **(extra_columns or {}),
            }
        ]
    )
    report_df.to_csv(report_path, index=False)
    print(f"\n=== {model_label} Results ===")
    print(report_df.to_string(index=False))
    print(f"Report saved to: {report_path}")

    if save_predictions:
        pred_path = output_paths.report_dir / report_name.replace("_results.csv", "_predictions.csv")
        out_preds = df[[text_col, label_col]].copy()
        out_preds["pred_id"] = y_pred
        out_preds.to_csv(pred_path, index=False, encoding="utf-8")
        print(f"Predictions saved to: {pred_path}")

    plot_path = output_paths.plot_dir / confusion_matrix_name
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=confusion_cmap,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(f"{model_label} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {plot_path}")

    return report_path


def run_vader(
    args: argparse.Namespace,
    model_display_name: str = "VADER (baseline)",
) -> None:
    output_paths = detect_output_paths(args.output_root)
    dataset_prefix = resolve_dataset_prefix(args)
    effective_slug = build_dataset_aware_slug(args.experiment_slug, dataset_prefix)
    df, texts, y_true = load_labeled_csv(args.input_csv, args.text_col, args.label_col)
    y_pred = predict_vader(texts, threshold=args.threshold)
    metrics = compute_metrics(y_true, y_pred)
    save_report(
        df=df,
        text_col=args.text_col,
        label_col=args.label_col,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        output_paths=output_paths,
        report_name=f"{effective_slug}_results.csv",
        confusion_matrix_name=f"{effective_slug}_confusion_matrix.png",
        dataset_prefix=dataset_prefix,
        model_label=model_display_name,
        save_predictions=args.save_predictions,
        extra_columns={"threshold": args.threshold},
        confusion_cmap="Purples",
    )


def run_hf_direct(
    args: argparse.Namespace,
    checkpoint: str,
    model_display_name: str,
    label_mapper: Callable[[str], int],
    progress_desc: str,
) -> None:
    output_paths = detect_output_paths(args.output_root)
    dataset_prefix = resolve_dataset_prefix(args)
    effective_slug = build_dataset_aware_slug(args.experiment_slug, dataset_prefix)
    df, texts, y_true = load_labeled_csv(args.input_csv, args.text_col, args.label_col)
    device = resolve_hf_device()
    y_pred = predict_hf_pretrained(
        texts=texts,
        checkpoint=checkpoint,
        batch_size=args.batch_size,
        device=device,
        label_mapper=label_mapper,
        desc=progress_desc,
    )
    metrics = compute_metrics(y_true, y_pred)
    save_report(
        df=df,
        text_col=args.text_col,
        label_col=args.label_col,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        output_paths=output_paths,
        report_name=f"{effective_slug}_results.csv",
        confusion_matrix_name=f"{effective_slug}_confusion_matrix.png",
        dataset_prefix=dataset_prefix,
        model_label=model_display_name,
        save_predictions=args.save_predictions,
    )

