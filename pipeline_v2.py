"""
Three-stage experiment pipeline:
1. SST-3 supervised sentiment training
2. MLM adaptation on TweetEval train + dev text
3. TweetEval supervised fine-tuning
4. Final evaluation on the self-built dataset
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "datasets"
DEFAULT_SEED = 42
MAX_LENGTH = 128
NUM_LABELS = 3
LABEL_NAMES = ["Negative (0)", "Neutral (1)", "Positive (2)"]
SELFBUILT_EVAL_PATH = DEFAULT_DATA_DIR / "selfbuilt_database_corrected.xlsx"


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: Path
    model_dir: Path
    report_dir: Path
    plot_dir: Path
    checkpoint_dir: Path
    train_batch_size: int
    eval_batch_size: int
    fp16: bool
    early_stopping_patience: int
    seed: int


@dataclass(frozen=True)
class ClassificationStageConfig:
    name: str
    base_model: str
    dataset_prefix: str
    output_dir: Path
    export_dir: Path
    report_name: str
    confusion_matrix_name: str
    model_label: str
    learning_rate: float
    num_epochs: int
    weight_decay: float
    use_weighted_loss: bool
    confusion_matrix_title: str
    confusion_matrix_cmap: str


@dataclass(frozen=True)
class MLMStageConfig:
    name: str
    base_model: str
    output_dir: Path
    export_dir: Path
    report_name: str
    learning_rate: float
    num_epochs: int
    weight_decay: float
    mlm_probability: float


class WeightedLossTrainer(Trainer):
    """Trainer with optional class-weighted cross entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is None or self.class_weights is None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the three-stage pipeline."""
    parser = argparse.ArgumentParser(description="Run the three-stage SST -> MLM -> TweetEval pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help="Choose which stage to run.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=BASE_DIR)
    parser.add_argument("--stage1-model", type=Path, default=None)
    parser.add_argument("--stage2-model", type=Path, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument(
        "--stage1-weighted-loss",
        action="store_true",
        help="Enable weighted cross entropy for Stage 1 supervised SST training.",
    )
    return parser.parse_args()


def detect_runtime(args: argparse.Namespace) -> RuntimeConfig:
    """Build runtime settings from CLI arguments and the current hardware."""
    use_gpu = torch.cuda.is_available()
    train_batch_size = args.train_batch_size or (32 if use_gpu else 8)
    eval_batch_size = args.eval_batch_size or train_batch_size
    output_root = args.output_root.resolve()

    runtime = RuntimeConfig(
        data_dir=args.data_dir.resolve(),
        model_dir=output_root / "models",
        report_dir=output_root / "reports",
        plot_dir=output_root / "reports" / "plots",
        checkpoint_dir=output_root / "checkpoints",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        fp16=use_gpu,
        early_stopping_patience=args.early_stopping_patience,
        seed=DEFAULT_SEED,
    )

    print("Runtime configuration:")
    print(f"  GPU available: {use_gpu}")
    print(f"  fp16 enabled: {runtime.fp16}")
    print(f"  Data dir: {runtime.data_dir}")
    print(f"  Output root: {output_root}")
    print(f"  Train batch size: {runtime.train_batch_size}")
    print(f"  Eval batch size: {runtime.eval_batch_size}")
    print(f"  Stage 1 epochs: {args.stage1_epochs or args.epochs}")
    print(f"  Stage 2 MLM epochs: {args.stage2_epochs or args.epochs}")
    print(f"  Stage 3 epochs: {args.stage3_epochs or args.epochs}")
    print(f"  Early stopping patience: {runtime.early_stopping_patience}")
    print(f"  Stage 1 weighted loss: {args.stage1_weighted_loss}")
    if use_gpu:
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    return runtime


def ensure_directories(runtime: RuntimeConfig) -> None:
    """Create all top-level output directories."""
    for path in (
        runtime.model_dir,
        runtime.report_dir,
        runtime.plot_dir,
        runtime.checkpoint_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def cleanup_v2_outputs(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    experiment_slug: str,
) -> None:
    """Remove stale outputs for the currently requested stages."""
    paths_to_remove = []

    if args.stage in {"all", "stage1"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage1_sst_results.csv",
                runtime.report_dir / f"{experiment_slug}_stage1_selfbuilt_results.csv",
                runtime.plot_dir / f"{experiment_slug}_stage1_sst_confusion_matrix.png",
                runtime.plot_dir / f"{experiment_slug}_stage1_selfbuilt_confusion_matrix.png",
                runtime.checkpoint_dir / f"{experiment_slug}_stage1_sst",
                runtime.model_dir / f"{experiment_slug}_stage1_sst_model",
            ]
        )

    if args.stage in {"all", "stage2"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage2_mlm_report.json",
                runtime.checkpoint_dir / f"{experiment_slug}_stage2_mlm",
                runtime.model_dir / f"{experiment_slug}_stage2_mlm_model",
            ]
        )

    if args.stage in {"all", "stage3"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage3_tweet_results.csv",
                runtime.report_dir / f"{experiment_slug}_stage3_selfbuilt_results.csv",
                runtime.plot_dir / f"{experiment_slug}_stage3_tweet_confusion_matrix.png",
                runtime.plot_dir / f"{experiment_slug}_stage3_selfbuilt_confusion_matrix.png",
                runtime.checkpoint_dir / f"{experiment_slug}_stage3_tweet",
                runtime.model_dir / f"{experiment_slug}_stage3_tweet_model",
            ]
        )

    for path in paths_to_remove:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def compute_metrics(eval_pred):
    """Compute classification metrics for supervised sentiment stages."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
    }


def compute_class_weights(train_dataset: Dataset) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights."""
    label_counts = pd.Series(train_dataset["label"]).value_counts().sort_index()
    total = float(label_counts.sum())
    raw_weights = total / (len(label_counts) * label_counts)
    normalized_weights = raw_weights / raw_weights.mean()
    return torch.tensor(normalized_weights.values, dtype=torch.float)


def load_split_dataset(data_dir: Path, dataset_prefix: str) -> dict[str, Dataset]:
    """Load a supervised dataset with train/val/test CSV files."""
    def read_split(split: str) -> Dataset:
        file_path = data_dir / f"{dataset_prefix}_{split}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        df = pd.read_csv(file_path)
        required_columns = {"text", "label_id"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"{file_path} is missing columns: {sorted(missing_columns)}")
        df = df[["text", "label_id"]].rename(columns={"label_id": "label"})
        return Dataset.from_pandas(df, preserve_index=False)

    return {
        "train": read_split("train"),
        "validation": read_split("val"),
        "test": read_split("test"),
    }


def load_excel_eval_dataset(file_path: Path) -> Dataset:
    """Load the self-built evaluation dataset from Excel."""
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")
    df = pd.read_excel(file_path)
    required_columns = {"text", "label_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {sorted(missing_columns)}")
    df = df[["text", "label_id"]].rename(columns={"label_id": "label"})
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_classification_dataset(dataset_splits: dict[str, Dataset], tokenizer) -> dict[str, Dataset]:
    """Tokenize supervised text classification splits."""
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return {
        split_name: split_dataset.map(tokenize_fn, batched=True)
        for split_name, split_dataset in dataset_splits.items()
    }


def create_classification_args(config: ClassificationStageConfig, runtime: RuntimeConfig) -> TrainingArguments:
    """Create Trainer arguments for supervised classification stages."""
    return TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=runtime.train_batch_size,
        per_device_eval_batch_size=runtime.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=runtime.fp16,
        report_to="none",
        save_total_limit=2,
        seed=runtime.seed,
        logging_strategy="epoch",
    )


def create_mlm_args(config: MLMStageConfig, runtime: RuntimeConfig) -> TrainingArguments:
    """Create Trainer arguments for MLM adaptation."""
    return TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=runtime.train_batch_size,
        per_device_eval_batch_size=runtime.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=runtime.fp16,
        report_to="none",
        save_total_limit=2,
        seed=runtime.seed,
        logging_strategy="epoch",
    )


def build_classification_trainer(
    config: ClassificationStageConfig,
    runtime: RuntimeConfig,
    tokenized_dataset: dict[str, Dataset],
    tokenizer,
    class_weights: torch.Tensor | None = None,
) -> Trainer:
    """Build a classification Trainer for Stage 1 or Stage 3."""
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=NUM_LABELS,
    )
    callbacks = []
    if runtime.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=runtime.early_stopping_patience
            )
        )

    return WeightedLossTrainer(
        model=model,
        args=create_classification_args(config, runtime),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks,
        class_weights=class_weights,
    )


def build_mlm_corpus(data_dir: Path) -> Dataset:
    """Build a text-only corpus from TweetEval train + validation splits."""
    frames = []
    for split in ("train", "val"):
        path = data_dir / f"tweeteval_sentiment_3class_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError(f"{path} is missing the `text` column.")
        frames.append(df[["text"]].copy())
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    return Dataset.from_pandas(combined, preserve_index=False)


def tokenize_mlm_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenize a plain-text MLM corpus."""
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def evaluate_predictions(trainer: Trainer, test_dataset: Dataset):
    """Evaluate a classification model on a labeled dataset."""
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_negative": f1_per_class[0],
        "f1_neutral": f1_per_class[1],
        "f1_positive": f1_per_class[2],
    }
    return metrics, y_true, y_pred


def save_classification_report(report_path: Path, dataset_name: str, model_label: str, metrics: dict) -> None:
    """Save a classification report CSV."""
    report_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model": model_label,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_negative": metrics["f1_negative"],
                "f1_neutral": metrics["f1_neutral"],
                "f1_positive": metrics["f1_positive"],
            }
        ]
    )
    report_df.to_csv(report_path, index=False)
    print(report_df.to_string(index=False))
    print(f"Report saved to: {report_path}")


def save_confusion_matrix(plot_path: Path, title: str, y_true: np.ndarray, y_pred: np.ndarray, cmap: str) -> None:
    """Save a confusion matrix image for a classification run."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {plot_path}")


def run_stage1(
    runtime: RuntimeConfig,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
    num_epochs: int,
    use_weighted_loss: bool,
) -> Path:
    """Run Stage 1 SST-3 supervised sentiment training."""
    config = ClassificationStageConfig(
        name=f"{model_display_name} V2 Stage 1",
        base_model=base_model_name,
        dataset_prefix="sst_3class",
        output_dir=runtime.checkpoint_dir / f"{experiment_slug}_stage1_sst",
        export_dir=runtime.model_dir / f"{experiment_slug}_stage1_sst_model",
        report_name=f"{experiment_slug}_stage1_sst_results.csv",
        confusion_matrix_name=f"{experiment_slug}_stage1_sst_confusion_matrix.png",
        model_label=f"{model_display_name} V2 (Stage 1 SST-3)",
        learning_rate=1e-5,
        num_epochs=num_epochs,
        weight_decay=0.1,
        use_weighted_loss=use_weighted_loss,
        confusion_matrix_title=f"{model_display_name} V2 - Stage 1 SST-3",
        confusion_matrix_cmap="Blues",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = load_split_dataset(runtime.data_dir, config.dataset_prefix)
    tokenized = tokenize_classification_dataset(dataset, tokenizer)

    class_weights = None
    if config.use_weighted_loss:
        class_weights = compute_class_weights(dataset["train"])
        print(f"Using weighted cross entropy with class weights: {class_weights.tolist()}")

    trainer = build_classification_trainer(
        config=config,
        runtime=runtime,
        tokenized_dataset=tokenized,
        tokenizer=tokenizer,
        class_weights=class_weights,
    )
    trainer.train()
    trainer.save_model(str(config.export_dir))
    tokenizer.save_pretrained(str(config.export_dir))

    metrics, y_true, y_pred = evaluate_predictions(trainer, tokenized["test"])
    print(f"\n=== {config.name} Results ===")
    save_classification_report(
        runtime.report_dir / config.report_name,
        config.dataset_prefix,
        config.model_label,
        metrics,
    )
    save_confusion_matrix(
        runtime.plot_dir / config.confusion_matrix_name,
        config.confusion_matrix_title,
        y_true,
        y_pred,
        config.confusion_matrix_cmap,
    )
    return config.export_dir


def run_stage2_mlm(
    runtime: RuntimeConfig,
    experiment_slug: str,
    base_model_path: Path | str,
    num_epochs: int,
    mlm_probability: float,
) -> Path:
    """Run Stage 2 masked language modeling on TweetEval train + dev text."""
    config = MLMStageConfig(
        name="RoBERTa Base V2 Stage 2 MLM",
        base_model=str(base_model_path),
        output_dir=runtime.checkpoint_dir / f"{experiment_slug}_stage2_mlm",
        export_dir=runtime.model_dir / f"{experiment_slug}_stage2_mlm_model",
        report_name=f"{experiment_slug}_stage2_mlm_report.json",
        learning_rate=1e-5,
        num_epochs=num_epochs,
        weight_decay=0.01,
        mlm_probability=mlm_probability,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    corpus = build_mlm_corpus(runtime.data_dir)
    split_corpus = corpus.train_test_split(test_size=0.1, seed=runtime.seed)
    tokenized_train = tokenize_mlm_dataset(split_corpus["train"], tokenizer)
    tokenized_eval = tokenize_mlm_dataset(split_corpus["test"], tokenizer)

    model = AutoModelForMaskedLM.from_pretrained(config.base_model)
    callbacks = []
    if runtime.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=runtime.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=create_mlm_args(config, runtime),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config.mlm_probability,
        ),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(config.export_dir))
    tokenizer.save_pretrained(str(config.export_dir))

    eval_metrics = trainer.evaluate(tokenized_eval)
    eval_loss = float(eval_metrics["eval_loss"])
    mlm_report = {
        "stage": "stage2_mlm",
        "base_model": config.base_model,
        "eval_loss": eval_loss,
        "perplexity": math.exp(eval_loss) if eval_loss < 20 else None,
        "num_epochs": config.num_epochs,
        "mlm_probability": config.mlm_probability,
    }
    report_path = runtime.report_dir / config.report_name
    report_path.write_text(json.dumps(mlm_report, indent=2))
    print("\n=== Stage 2 MLM Report ===")
    print(json.dumps(mlm_report, indent=2))
    print(f"Report saved to: {report_path}")
    return config.export_dir


def run_stage3(
    runtime: RuntimeConfig,
    experiment_slug: str,
    base_model_path: Path | str,
    model_display_name: str,
    num_epochs: int,
) -> Path:
    """Run Stage 3 TweetEval supervised fine-tuning and final evaluation."""
    config = ClassificationStageConfig(
        name=f"{model_display_name} V2 Stage 3",
        base_model=str(base_model_path),
        dataset_prefix="tweeteval_sentiment_3class",
        output_dir=runtime.checkpoint_dir / f"{experiment_slug}_stage3_tweet",
        export_dir=runtime.model_dir / f"{experiment_slug}_stage3_tweet_model",
        report_name=f"{experiment_slug}_stage3_tweet_results.csv",
        confusion_matrix_name=f"{experiment_slug}_stage3_tweet_confusion_matrix.png",
        model_label=f"{model_display_name} V2 (Stage 3 TweetEval)",
        learning_rate=1e-5,
        num_epochs=num_epochs,
        weight_decay=0.1,
        use_weighted_loss=False,
        confusion_matrix_title=f"{model_display_name} V2 - Stage 3 TweetEval",
        confusion_matrix_cmap="Greens",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = load_split_dataset(runtime.data_dir, config.dataset_prefix)
    tokenized = tokenize_classification_dataset(dataset, tokenizer)

    trainer = build_classification_trainer(
        config=config,
        runtime=runtime,
        tokenized_dataset=tokenized,
        tokenizer=tokenizer,
        class_weights=None,
    )
    trainer.train()
    trainer.save_model(str(config.export_dir))
    tokenizer.save_pretrained(str(config.export_dir))

    tweet_metrics, y_true, y_pred = evaluate_predictions(trainer, tokenized["test"])
    print(f"\n=== {config.name} TweetEval Results ===")
    save_classification_report(
        runtime.report_dir / config.report_name,
        config.dataset_prefix,
        config.model_label,
        tweet_metrics,
    )
    save_confusion_matrix(
        runtime.plot_dir / config.confusion_matrix_name,
        config.confusion_matrix_title,
        y_true,
        y_pred,
        config.confusion_matrix_cmap,
    )

    selfbuilt_dataset = load_excel_eval_dataset(SELFBUILT_EVAL_PATH)
    tokenized_selfbuilt = tokenize_classification_dataset({"test": selfbuilt_dataset}, tokenizer)["test"]
    selfbuilt_metrics, y_true, y_pred = evaluate_predictions(trainer, tokenized_selfbuilt)

    print(f"\n=== {config.name} Self-Built Results ===")
    save_classification_report(
        runtime.report_dir / f"{experiment_slug}_stage3_selfbuilt_results.csv",
        "selfbuilt_database_corrected",
        f"{config.model_label} [Self-Built Test]",
        selfbuilt_metrics,
    )
    save_confusion_matrix(
        runtime.plot_dir / f"{experiment_slug}_stage3_selfbuilt_confusion_matrix.png",
        f"{config.model_label} - Self-Built",
        y_true,
        y_pred,
        "Purples",
    )
    return config.export_dir


def resolve_stage1_model_path(args: argparse.Namespace, runtime: RuntimeConfig, experiment_slug: str) -> Path:
    """Resolve the Stage 1 model path for V2."""
    if args.stage1_model is not None:
        model_path = args.stage1_model.resolve()
    else:
        model_path = runtime.model_dir / f"{experiment_slug}_stage1_sst_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {model_path}")
    return model_path


def resolve_stage2_model_path(args: argparse.Namespace, runtime: RuntimeConfig, experiment_slug: str) -> Path:
    """Resolve the Stage 2 MLM model path for V2."""
    if args.stage2_model is not None:
        model_path = args.stage2_model.resolve()
    else:
        model_path = runtime.model_dir / f"{experiment_slug}_stage2_mlm_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 2 MLM model not found: {model_path}")
    return model_path


def run_three_stage_experiment(
    args: argparse.Namespace,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
) -> None:
    """Run the full Stage1 -> Stage2 MLM -> Stage3 pipeline."""
    runtime = detect_runtime(args)
    ensure_directories(runtime)
    cleanup_v2_outputs(args, runtime, experiment_slug)
    set_seed(runtime.seed)

    stage1_epochs = args.stage1_epochs or args.epochs
    stage2_epochs = args.stage2_epochs or args.epochs
    stage3_epochs = args.stage3_epochs or args.epochs

    if args.stage in {"all", "stage1"}:
        stage1_model_path = run_stage1(
            runtime=runtime,
            experiment_slug=experiment_slug,
            base_model_name=base_model_name,
            model_display_name=model_display_name,
            num_epochs=stage1_epochs,
            use_weighted_loss=args.stage1_weighted_loss,
        )
    else:
        stage1_model_path = resolve_stage1_model_path(args, runtime, experiment_slug)

    if args.stage in {"all", "stage2"}:
        stage2_model_path = run_stage2_mlm(
            runtime=runtime,
            experiment_slug=experiment_slug,
            base_model_path=stage1_model_path,
            num_epochs=stage2_epochs,
            mlm_probability=args.mlm_probability,
        )
    else:
        stage2_model_path = resolve_stage2_model_path(args, runtime, experiment_slug)

    if args.stage in {"all", "stage3"}:
        run_stage3(
            runtime=runtime,
            experiment_slug=experiment_slug,
            base_model_path=stage2_model_path,
            model_display_name=model_display_name,
            num_epochs=stage3_epochs,
        )


def run_mlm_stage3_experiment(
    args: argparse.Namespace,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
) -> None:
    """Run the Tweet MLM -> TweetEval supervised pipeline without Stage 1.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        experiment_slug (str): Stable identifier used in output filenames.
        base_model_name (str): Initial Hugging Face model checkpoint name.
        model_display_name (str): Human-readable model name used in reports.

    Returns:
        None
    """
    runtime = detect_runtime(args)
    ensure_directories(runtime)

    paths_to_remove = []
    if args.stage in {"all", "stage2"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage2_mlm_report.json",
                runtime.checkpoint_dir / f"{experiment_slug}_stage2_mlm",
                runtime.model_dir / f"{experiment_slug}_stage2_mlm_model",
            ]
        )
    if args.stage in {"all", "stage3"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage3_tweet_results.csv",
                runtime.report_dir / f"{experiment_slug}_stage3_selfbuilt_results.csv",
                runtime.plot_dir / f"{experiment_slug}_stage3_tweet_confusion_matrix.png",
                runtime.plot_dir / f"{experiment_slug}_stage3_selfbuilt_confusion_matrix.png",
                runtime.checkpoint_dir / f"{experiment_slug}_stage3_tweet",
                runtime.model_dir / f"{experiment_slug}_stage3_tweet_model",
            ]
        )
    for path in paths_to_remove:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    set_seed(runtime.seed)
    stage2_epochs = args.stage2_epochs or args.epochs
    stage3_epochs = args.stage3_epochs or args.epochs

    if args.stage in {"all", "stage2"}:
        stage2_model_path = run_stage2_mlm(
            runtime=runtime,
            experiment_slug=experiment_slug,
            base_model_path=base_model_name,
            num_epochs=stage2_epochs,
            mlm_probability=args.mlm_probability,
        )
    else:
        stage2_model_path = resolve_stage2_model_path(args, runtime, experiment_slug)

    if args.stage in {"all", "stage3"}:
        run_stage3(
            runtime=runtime,
            experiment_slug=experiment_slug,
            base_model_path=stage2_model_path,
            model_display_name=model_display_name,
            num_epochs=stage3_epochs,
        )
