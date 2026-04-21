"""
Shared transfer-learning pipeline for sentiment experiments.

This module contains the reusable training and evaluation logic used by
different runner scripts so model families can be compared consistently.
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "datasets"
DEFAULT_SEED = 42
LABEL_NAMES = ["Negative (0)", "Neutral (1)", "Positive (2)"]
NUM_LABELS = 3
MAX_LENGTH = 128
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
    seed: int
    fp16: bool
    early_stopping_patience: int


@dataclass(frozen=True)
class StageConfig:
    name: str
    base_model: str
    dataset_prefix: str
    output_dir: Path
    export_dir: Path
    report_name: str
    confusion_matrix_name: str
    selfbuilt_report_name: str
    selfbuilt_confusion_matrix_name: str
    model_label: str
    learning_rate: float
    num_epochs: int
    weight_decay: float
    use_weighted_loss: bool
    confusion_matrix_title: str
    confusion_matrix_cmap: str


class WeightedLossTrainer(Trainer):
    """Trainer that optionally applies class-weighted cross entropy."""

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
    """Parse command-line arguments for a transfer-learning experiment.

    Returns:
        argparse.Namespace: Parsed CLI arguments including stage selection,
        paths, epoch settings, batch sizes, and early stopping settings.
    """
    parser = argparse.ArgumentParser(description="Run a transfer-learning sentiment experiment.")
    parser.add_argument(
        "--stage",
        choices=["all", "stage1", "stage2"],
        default="all",
        help="Choose which training stage to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing dataset CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR,
        help="Root directory for models, reports, and checkpoints.",
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=None,
        help="Existing Stage 1 model directory. Required if running only Stage 2.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override training batch size. Defaults to 32 on Colab T4 / GPU and 8 on CPU.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size. Defaults to train batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Shared epoch count used when stage-specific epoch arguments are not set.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="Epoch count for Stage 1. Overrides --epochs for Stage 1 only.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=None,
        help="Epoch count for Stage 2. Overrides --epochs for Stage 2 only.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop training after this many unimproved evaluations.",
    )
    return parser.parse_args()


def detect_runtime(args: argparse.Namespace) -> RuntimeConfig:
    """Build runtime settings from CLI arguments and hardware availability.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        RuntimeConfig: A normalized runtime configuration containing resolved
        paths, batch sizes, mixed-precision settings, a fixed random seed,
        and early stopping setup.
    """
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
        seed=DEFAULT_SEED,
        fp16=use_gpu,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("Runtime configuration:")
    print(f"  GPU available: {use_gpu}")
    print(f"  fp16 enabled: {runtime.fp16}")
    print(f"  Data dir: {runtime.data_dir}")
    print(f"  Output root: {output_root}")
    print(f"  Train batch size: {runtime.train_batch_size}")
    print(f"  Eval batch size: {runtime.eval_batch_size}")
    print(f"  Shared default epochs: {args.epochs}")
    print(f"  Stage 1 epochs: {args.stage1_epochs or args.epochs}")
    print(f"  Stage 2 epochs: {args.stage2_epochs or args.epochs}")
    print(f"  Early stopping patience: {runtime.early_stopping_patience}")
    if use_gpu:
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    return runtime


def ensure_directories(runtime: RuntimeConfig) -> None:
    """Create all output directories required by the experiment.

    Args:
        runtime (RuntimeConfig): Runtime configuration with output paths.

    Returns:
        None
    """
    for path in (
        runtime.model_dir,
        runtime.report_dir,
        runtime.plot_dir,
        runtime.checkpoint_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def cleanup_experiment_outputs(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    experiment_slug: str,
) -> None:
    """Remove stale outputs for the current experiment before a new run starts.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        runtime (RuntimeConfig): Runtime configuration with output locations.
        experiment_slug (str): Stable experiment identifier used in filenames.

    Returns:
        None
    """
    paths_to_remove = []

    # Reports and plots should always be regenerated for the stage being run.
    if args.stage in {"all", "stage1"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage1_sst_results.csv",
                runtime.report_dir / f"{experiment_slug}_stage1_selfbuilt_results.csv",
                runtime.plot_dir / f"{experiment_slug}_stage1_sst_confusion_matrix.png",
                runtime.plot_dir / f"{experiment_slug}_stage1_selfbuilt_confusion_matrix.png",
                runtime.checkpoint_dir / f"{experiment_slug}_sst_stage1",
                runtime.model_dir / f"{experiment_slug}_sst_pretrained",
            ]
        )

    if args.stage in {"all", "stage2"}:
        paths_to_remove.extend(
            [
                runtime.report_dir / f"{experiment_slug}_stage2_tweet_results.csv",
                runtime.report_dir / f"{experiment_slug}_stage2_selfbuilt_results.csv",
                runtime.plot_dir / f"{experiment_slug}_stage2_tweet_confusion_matrix.png",
                runtime.plot_dir / f"{experiment_slug}_stage2_selfbuilt_confusion_matrix.png",
                runtime.checkpoint_dir / f"{experiment_slug}_tweet_stage2",
                runtime.model_dir / f"{experiment_slug}_best_model",
            ]
        )

    for path in paths_to_remove:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def compute_metrics(eval_pred):
    """Compute validation metrics for Hugging Face Trainer callbacks.

    Args:
        eval_pred: A tuple-like object containing model logits and ground-truth
        labels provided by the Trainer evaluation loop.

    Returns:
        dict: Validation metrics including accuracy, macro precision,
        macro recall, and macro F1. `f1_macro` is used to select the best
        checkpoint during training.
    """
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
    """Compute normalized inverse-frequency class weights from the training split.

    Args:
        train_dataset (Dataset): Training split containing a `label` column.

    Returns:
        torch.Tensor: Normalized class weights ordered by label id.
    """
    label_counts = pd.Series(train_dataset["label"]).value_counts().sort_index()
    total = float(label_counts.sum())
    raw_weights = total / (len(label_counts) * label_counts)
    normalized_weights = raw_weights / raw_weights.mean()
    return torch.tensor(normalized_weights.values, dtype=torch.float)


def load_split_dataset(data_dir: Path, dataset_prefix: str) -> DatasetDict:
    """Load train, validation, and test CSV files into a DatasetDict.

    Args:
        data_dir (Path): Directory that stores the dataset CSV files.
        dataset_prefix (str): Common file prefix, such as `sst_3class` or
        `tweeteval_sentiment_3class`.

    Returns:
        DatasetDict: A Hugging Face dataset dictionary with `train`,
        `validation`, and `test` splits.
    """
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

    return DatasetDict(
        {
            "train": read_split("train"),
            "validation": read_split("val"),
            "test": read_split("test"),
        }
    )


def load_excel_eval_dataset(file_path: Path) -> Dataset:
    """Load an external evaluation dataset from an Excel file.

    Args:
        file_path (Path): Excel file containing at least `text` and `label_id`.

    Returns:
        Dataset: Hugging Face dataset with `text` and `label` columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    df = pd.read_excel(file_path)
    required_columns = {"text", "label_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {sorted(missing_columns)}")

    df = df[["text", "label_id"]].rename(columns={"label_id": "label"})
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """Tokenize every split in a DatasetDict with a shared tokenizer.

    Args:
        dataset (DatasetDict): Dataset splits containing a `text` column.
        tokenizer: A Hugging Face tokenizer compatible with the target model.

    Returns:
        DatasetDict: Tokenized dataset splits ready for Trainer input.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenize_fn, batched=True)


def create_training_args(config: StageConfig, runtime: RuntimeConfig) -> TrainingArguments:
    """Create Hugging Face TrainingArguments for one training stage.

    Args:
        config (StageConfig): Per-stage settings such as learning rate,
        epoch count, and checkpoint directory.
        runtime (RuntimeConfig): Runtime-wide settings such as batch size,
        fixed seed, and fp16 availability.

    Returns:
        TrainingArguments: Fully configured Trainer arguments for this stage.
    """
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


def build_trainer(
    config: StageConfig,
    runtime: RuntimeConfig,
    tokenized_dataset: DatasetDict,
    tokenizer,
    class_weights: torch.Tensor | None = None,
) -> Trainer:
    """Construct a Trainer for one stage of the experiment.

    Args:
        config (StageConfig): Stage-specific model and optimization settings.
        runtime (RuntimeConfig): Runtime-wide execution settings.
        tokenized_dataset (DatasetDict): Tokenized train/validation/test splits.
        tokenizer: Tokenizer paired with the model checkpoint.

    Returns:
        Trainer: A Hugging Face Trainer configured with the model, data,
        metrics, collator, and optional early stopping callback.
    """
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
        args=create_training_args(config, runtime),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks,
        class_weights=class_weights,
    )


def evaluate_predictions(trainer: Trainer, test_dataset: Dataset):
    """Run final evaluation on a test split and summarize the predictions.

    Args:
        trainer (Trainer): Trained Hugging Face Trainer instance.
        test_dataset (Dataset): Tokenized test split used for final evaluation.

    Returns:
        tuple: A tuple of `(metrics, y_true, y_pred)` where `metrics` is a dict
        of aggregate scores and `y_true` / `y_pred` are numpy arrays used for
        further reporting and visualization.
    """
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


def save_report(runtime: RuntimeConfig, config: StageConfig, metrics: dict) -> Path:
    """Write final metrics for one stage to a CSV report file.

    Args:
        runtime (RuntimeConfig): Runtime configuration containing report paths.
        config (StageConfig): Stage metadata used for report naming.
        metrics (dict): Final evaluation metrics for this stage.

    Returns:
        Path: The path to the saved CSV report.
    """
    report_path = runtime.report_dir / config.report_name
    report_df = pd.DataFrame(
        [
            {
                "dataset": config.dataset_prefix,
                "model": config.model_label,
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
    print(f"\n=== {config.name} Results ===")
    print(report_df.to_string(index=False))
    print(f"Report saved to: {report_path}")
    return report_path


def save_confusion_matrix(
    runtime: RuntimeConfig,
    config: StageConfig,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    """Generate and save a confusion-matrix plot for one stage.

    Args:
        runtime (RuntimeConfig): Runtime configuration containing plot paths.
        config (StageConfig): Stage metadata used for plot naming and styling.
        y_true (np.ndarray): Ground-truth labels from the test split.
        y_pred (np.ndarray): Predicted labels from the trained model.

    Returns:
        Path: The path to the saved confusion-matrix image.
    """
    plot_path = runtime.plot_dir / config.confusion_matrix_name
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=config.confusion_matrix_cmap,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(config.confusion_matrix_title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Confusion matrix saved to: {plot_path}")
    return plot_path


def evaluate_on_selfbuilt_dataset(
    runtime: RuntimeConfig,
    config: StageConfig,
    trainer: Trainer,
    tokenizer,
) -> None:
    """Evaluate the trained model on the corrected self-built dataset.

    Args:
        runtime (RuntimeConfig): Runtime configuration containing output paths.
        config (StageConfig): Stage metadata used for output naming.
        trainer (Trainer): Trained model wrapper.
        tokenizer: Tokenizer used by the current model.

    Returns:
        None
    """
    if not SELFBUILT_EVAL_PATH.exists():
        print(f"Skipping self-built evaluation because the file does not exist: {SELFBUILT_EVAL_PATH}")
        return

    selfbuilt_dataset = load_excel_eval_dataset(SELFBUILT_EVAL_PATH)
    tokenized_selfbuilt = selfbuilt_dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )

    metrics, y_true, y_pred = evaluate_predictions(trainer, tokenized_selfbuilt)

    selfbuilt_config = StageConfig(
        name=f"{config.name} Self-Built Evaluation",
        base_model=config.base_model,
        dataset_prefix="selfbuilt_database_corrected",
        output_dir=config.output_dir,
        export_dir=config.export_dir,
        report_name=config.selfbuilt_report_name,
        confusion_matrix_name=config.selfbuilt_confusion_matrix_name,
        selfbuilt_report_name=config.selfbuilt_report_name,
        selfbuilt_confusion_matrix_name=config.selfbuilt_confusion_matrix_name,
        model_label=f"{config.model_label} [Self-Built Test]",
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        use_weighted_loss=config.use_weighted_loss,
        confusion_matrix_title=f"{config.model_label} - Self-Built Confusion Matrix",
        confusion_matrix_cmap="Purples",
    )

    save_report(runtime, selfbuilt_config, metrics)
    save_confusion_matrix(runtime, selfbuilt_config, y_true, y_pred)


def build_stage1_config(
    runtime: RuntimeConfig,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
    num_epochs: int,
) -> StageConfig:
    """Create the Stage 1 configuration for SST-3 fine-tuning.

    Args:
        runtime (RuntimeConfig): Runtime configuration containing output roots.
        experiment_slug (str): Stable experiment identifier used in filenames.
        base_model_name (str): Hugging Face model name used to start Stage 1.
        model_display_name (str): Human-readable model name for reports.
        num_epochs (int): Number of training epochs for Stage 1.

    Returns:
        StageConfig: Stage 1 configuration object.
    """
    return StageConfig(
        name=f"{model_display_name} Stage 1",
        base_model=base_model_name,
        dataset_prefix="sst_3class",
        output_dir=runtime.checkpoint_dir / f"{experiment_slug}_sst_stage1",
        export_dir=runtime.model_dir / f"{experiment_slug}_sst_pretrained",
        report_name=f"{experiment_slug}_stage1_sst_results.csv",
        confusion_matrix_name=f"{experiment_slug}_stage1_sst_confusion_matrix.png",
        selfbuilt_report_name=f"{experiment_slug}_stage1_selfbuilt_results.csv",
        selfbuilt_confusion_matrix_name=f"{experiment_slug}_stage1_selfbuilt_confusion_matrix.png",
        model_label=f"{model_display_name} (Fine-tuned on SST-3)",
        learning_rate=1e-5,
        num_epochs=num_epochs,
        weight_decay=0.1,
        use_weighted_loss=True,
        confusion_matrix_title=f"{model_display_name} - Stage 1 SST-3 Confusion Matrix",
        confusion_matrix_cmap="Blues",
    )


def build_stage2_config(
    runtime: RuntimeConfig,
    experiment_slug: str,
    stage1_model_path: Path,
    model_display_name: str,
    num_epochs: int,
) -> StageConfig:
    """Create the Stage 2 configuration for TweetEval transfer learning.

    Args:
        runtime (RuntimeConfig): Runtime configuration containing output roots.
        experiment_slug (str): Stable experiment identifier used in filenames.
        stage1_model_path (Path): Saved Stage 1 checkpoint used to initialize
        Stage 2.
        model_display_name (str): Human-readable model name for reports.
        num_epochs (int): Number of training epochs for Stage 2.

    Returns:
        StageConfig: Stage 2 configuration object.
    """
    return StageConfig(
        name=f"{model_display_name} Stage 2",
        base_model=str(stage1_model_path),
        dataset_prefix="tweeteval_sentiment_3class",
        output_dir=runtime.checkpoint_dir / f"{experiment_slug}_tweet_stage2",
        export_dir=runtime.model_dir / f"{experiment_slug}_best_model",
        report_name=f"{experiment_slug}_stage2_tweet_results.csv",
        confusion_matrix_name=f"{experiment_slug}_stage2_tweet_confusion_matrix.png",
        selfbuilt_report_name=f"{experiment_slug}_stage2_selfbuilt_results.csv",
        selfbuilt_confusion_matrix_name=f"{experiment_slug}_stage2_selfbuilt_confusion_matrix.png",
        model_label=f"{model_display_name} (SST -> Tweet Transfer)",
        learning_rate=5e-6,
        num_epochs=num_epochs,
        weight_decay=0.1,
        use_weighted_loss=False,
        confusion_matrix_title=f"{model_display_name} - Stage 2 TweetEval Confusion Matrix",
        confusion_matrix_cmap="Greens",
    )


def build_direct_tweet_config(
    runtime: RuntimeConfig,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
    num_epochs: int,
) -> StageConfig:
    """Create a direct TweetEval baseline configuration without Stage 1.

    Args:
        runtime (RuntimeConfig): Runtime configuration containing output roots.
        experiment_slug (str): Stable experiment identifier used in filenames.
        base_model_name (str): Hugging Face model name used to initialize training.
        model_display_name (str): Human-readable model name for reports.
        num_epochs (int): Number of training epochs for the baseline run.

    Returns:
        StageConfig: Configuration object for direct TweetEval training.
    """
    return StageConfig(
        name=f"{model_display_name} Direct TweetEval Baseline",
        base_model=base_model_name,
        dataset_prefix="tweeteval_sentiment_3class",
        output_dir=runtime.checkpoint_dir / f"{experiment_slug}_direct_tweet",
        export_dir=runtime.model_dir / f"{experiment_slug}_direct_tweet_model",
        report_name=f"{experiment_slug}_direct_tweet_results.csv",
        confusion_matrix_name=f"{experiment_slug}_direct_tweet_confusion_matrix.png",
        selfbuilt_report_name=f"{experiment_slug}_direct_selfbuilt_results.csv",
        selfbuilt_confusion_matrix_name=f"{experiment_slug}_direct_selfbuilt_confusion_matrix.png",
        model_label=f"{model_display_name} (Direct TweetEval)",
        learning_rate=1e-5,
        num_epochs=num_epochs,
        weight_decay=0.1,
        use_weighted_loss=False,
        confusion_matrix_title=f"{model_display_name} - Direct TweetEval Confusion Matrix",
        confusion_matrix_cmap="Oranges",
    )


def run_stage(runtime: RuntimeConfig, config: StageConfig) -> Path:
    """Execute one full training stage from tokenization to final artifacts.

    Args:
        runtime (RuntimeConfig): Runtime configuration shared across stages.
        config (StageConfig): Stage-specific configuration.

    Returns:
        Path: The directory where the trained model for this stage was saved.
    """
    print(f"\nStarting {config.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = load_split_dataset(runtime.data_dir, config.dataset_prefix)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    class_weights = None
    if config.use_weighted_loss:
        class_weights = compute_class_weights(dataset["train"])
        print(f"Using weighted cross entropy with class weights: {class_weights.tolist()}")

    trainer = build_trainer(
        config,
        runtime,
        tokenized_dataset,
        tokenizer,
        class_weights=class_weights,
    )
    trainer.train()
    trainer.save_model(str(config.export_dir))
    tokenizer.save_pretrained(str(config.export_dir))

    metrics, y_true, y_pred = evaluate_predictions(trainer, tokenized_dataset["test"])
    save_report(runtime, config, metrics)
    save_confusion_matrix(runtime, config, y_true, y_pred)
    evaluate_on_selfbuilt_dataset(runtime, config, trainer, tokenizer)
    return config.export_dir


def resolve_stage1_model_path(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    experiment_slug: str,
) -> Path:
    """Resolve the Stage 1 model path when running Stage 2 only.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        runtime (RuntimeConfig): Runtime configuration with default output paths.
        experiment_slug (str): Experiment identifier used to derive default
        model paths.

    Returns:
        Path: Path to an existing Stage 1 model directory.
    """
    if args.stage1_model is not None:
        model_path = args.stage1_model.resolve()
    else:
        model_path = runtime.model_dir / f"{experiment_slug}_sst_pretrained"

    if not model_path.exists():
        raise FileNotFoundError(
            "Stage 2 requires a Stage 1 model. "
            f"Expected model directory at: {model_path}"
        )
    return model_path


def run_transfer_experiment(
    args: argparse.Namespace,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
) -> None:
    """Run the full two-stage transfer-learning experiment for one model family.

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
    cleanup_experiment_outputs(args, runtime, experiment_slug)
    set_seed(runtime.seed)

    stage1_epochs = args.stage1_epochs or args.epochs
    stage2_epochs = args.stage2_epochs or args.epochs

    if args.stage in {"all", "stage1"}:
        stage1_config = build_stage1_config(
            runtime=runtime,
            experiment_slug=experiment_slug,
            base_model_name=base_model_name,
            model_display_name=model_display_name,
            num_epochs=stage1_epochs,
        )
        stage1_model_path = run_stage(runtime, stage1_config)
    else:
        stage1_model_path = resolve_stage1_model_path(args, runtime, experiment_slug)

    if args.stage in {"all", "stage2"}:
        stage2_config = build_stage2_config(
            runtime=runtime,
            experiment_slug=experiment_slug,
            stage1_model_path=stage1_model_path,
            model_display_name=model_display_name,
            num_epochs=stage2_epochs,
        )
        run_stage(runtime, stage2_config)


def run_direct_tweet_experiment(
    args: argparse.Namespace,
    experiment_slug: str,
    base_model_name: str,
    model_display_name: str,
) -> None:
    """Run a direct TweetEval baseline without Stage 1 transfer learning.

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

    paths_to_remove = [
        runtime.report_dir / f"{experiment_slug}_direct_tweet_results.csv",
        runtime.report_dir / f"{experiment_slug}_direct_selfbuilt_results.csv",
        runtime.plot_dir / f"{experiment_slug}_direct_tweet_confusion_matrix.png",
        runtime.plot_dir / f"{experiment_slug}_direct_selfbuilt_confusion_matrix.png",
        runtime.checkpoint_dir / f"{experiment_slug}_direct_tweet",
        runtime.model_dir / f"{experiment_slug}_direct_tweet_model",
    ]
    for path in paths_to_remove:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    set_seed(runtime.seed)
    direct_epochs = args.stage2_epochs or args.epochs
    direct_config = build_direct_tweet_config(
        runtime=runtime,
        experiment_slug=experiment_slug,
        base_model_name=base_model_name,
        model_display_name=model_display_name,
        num_epochs=direct_epochs,
    )
    run_stage(runtime, direct_config)
