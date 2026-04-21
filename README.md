# NLP Project

A two-stage sentiment classification project.

This repository currently supports two experiment pipelines:
- `roberta-base`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

Both pipelines use the same transfer-learning workflow:
- Stage 1: fine-tune on SST-3
- Stage 2: load the Stage 1 checkpoint and continue training on TweetEval 3-class

## Project Structure

```text
NLP_project/
├── datasets/
│   ├── sst_3class_train.csv
│   ├── sst_3class_val.csv
│   ├── sst_3class_test.csv
│   ├── tweeteval_sentiment_3class_train.csv
│   ├── tweeteval_sentiment_3class_val.csv
│   └── tweeteval_sentiment_3class_test.csv
├── training_pipeline.py
├── run.py
├── run_roberta_base.py
├── run_twitter_roberta.py
├── requirements.txt
└── README.md
```

File roles:
- `training_pipeline.py`
  - Shared training logic: data loading, tokenization, Trainer setup, evaluation, and export
- `run_roberta_base.py`
  - Runs the `roberta-base` experiment
- `run_twitter_roberta.py`
  - Runs the `twitter-roberta-base-sentiment-latest` experiment

Running an experiment will generate:
- `models/`
- `reports/`
- `reports/plots/`
- `checkpoints/`

## Environment

Recommended environment:
- Colab
- GPU: `T4`
- Python 3.10+

Default behavior:
- Automatically enables `fp16` when a GPU is available
- Uses `train_batch_size=32` on GPU
- Uses `train_batch_size=8` on CPU
- Selects the best checkpoint using `f1_macro`
- Uses `EarlyStoppingCallback` to stop training when validation performance stops improving

## Installation

### Local

```bash
pip install -r requirements.txt
```

### Colab

```python
!pip install -r requirements.txt
```

## Dataset Format

Each CSV file in `datasets/` must contain at least:
- `text`
- `label_id`

The code automatically renames `label_id` to `label` for Hugging Face `Trainer`.

Label mapping:
- `0`: Negative
- `1`: Neutral
- `2`: Positive

## Colab Usage

### 1. Clone the GitLab repository

```python
!git clone <your-gitlab-repo-url>
%cd NLP_project
```

### 2. Install dependencies

```python
!pip install -r requirements.txt
```

### 3. Check GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

You should normally see `Tesla T4`.

### 4. Run the `roberta-base` experiment

```python
!python run_roberta_base.py
```

### 5. Run the `twitter-roberta-base-sentiment-latest` experiment

```python
!python run_twitter_roberta.py
```

### 6. Save outputs to Google Drive

Colab local storage is temporary, so it is better to save outputs to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Example:

```python
!python run_roberta_base.py --output-root /content/drive/MyDrive/nlp_project_runs
!python run_twitter_roberta.py --output-root /content/drive/MyDrive/nlp_project_runs
```

## Common Commands

### Run the full experiment

```bash
python run_roberta_base.py
python run_twitter_roberta.py
```

### Run only Stage 1

```bash
python run_roberta_base.py --stage stage1
```

### Run only Stage 2

Make sure the matching Stage 1 model already exists:

```bash
python run_roberta_base.py --stage stage2 --stage1-model ./models/roberta_base_sst_pretrained
python run_twitter_roberta.py --stage stage2 --stage1-model ./models/twitter_roberta_sst_pretrained
```

### Use different epoch counts for the two stages

```bash
python run_roberta_base.py --stage1-epochs 4 --stage2-epochs 2
python run_twitter_roberta.py --stage1-epochs 4 --stage2-epochs 2
```

### Adjust early stopping

```bash
python run_roberta_base.py --early-stopping-patience 2
```

Notes:
- Training stops early when validation `f1_macro` does not improve
- `--early-stopping-patience 0` can be treated as disabling early stopping

### Adjust batch size

```bash
python run_roberta_base.py --train-batch-size 16 --eval-batch-size 16
```

## Function Flow

High-level function flow in `training_pipeline.py`:

```text
runner main()
  -> parse_args()
  -> run_transfer_experiment(...)
       -> detect_runtime(args)
       -> ensure_directories(runtime)
       -> set_seed(42)
       -> build_stage1_config(...)                        if stage1 is requested
       -> run_stage(runtime, stage1_config)
            -> AutoTokenizer.from_pretrained(...)
            -> load_split_dataset(...)
                 -> read_split("train" / "val" / "test")
            -> tokenize_dataset(...)
            -> build_trainer(...)
                 -> create_training_args(...)
                 -> AutoModelForSequenceClassification.from_pretrained(...)
                 -> EarlyStoppingCallback(...)
            -> trainer.train()
            -> trainer.save_model(...)
            -> tokenizer.save_pretrained(...)
            -> evaluate_predictions(...)
            -> save_report(...)
            -> save_confusion_matrix(...)
       -> resolve_stage1_model_path(...)                 if only stage2 is requested
       -> build_stage2_config(...)                       if stage2 is requested
       -> run_stage(runtime, stage2_config)
```

In short:
- `parse_args` / `detect_runtime`
  - Decide which stage to run, epoch counts, batch size, output paths, and GPU settings
- `run_transfer_experiment`
  - Orchestrates the full two-stage experiment
- `build_stage*_config`
  - Defines stage-specific settings and output names
- `run_stage`
  - Executes one full stage: load data, tokenize, train, evaluate, export
- `compute_metrics`
  - Computes `accuracy`, `precision_macro`, `recall_macro`, and `f1_macro`
- `save_report` / `save_confusion_matrix`
  - Export final artifacts

## Outputs

By default, outputs are saved under the project root. Different experiments use different names so their results do not overwrite each other.

`roberta-base` outputs:
- `models/roberta_base_sst_pretrained/`
- `models/roberta_base_best_model/`
- `reports/roberta_base_stage1_sst_results.csv`
- `reports/roberta_base_stage2_tweet_results.csv`
- `reports/plots/roberta_base_stage1_sst_confusion_matrix.png`
- `reports/plots/roberta_base_stage2_tweet_confusion_matrix.png`

`twitter-roberta` outputs:
- `models/twitter_roberta_sst_pretrained/`
- `models/twitter_roberta_best_model/`
- `reports/twitter_roberta_stage1_sst_results.csv`
- `reports/twitter_roberta_stage2_tweet_results.csv`
- `reports/plots/twitter_roberta_stage1_sst_confusion_matrix.png`
- `reports/plots/twitter_roberta_stage2_tweet_confusion_matrix.png`
