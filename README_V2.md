# NLP Project V2

This document describes the second experiment line only.

The original two-stage pipeline is documented separately and remains unchanged.

## Pipeline Overview

The V2 experiment uses a three-stage workflow:

1. Stage 1: SST-3 supervised sentiment training
2. Stage 2: MLM adaptation on TweetEval train + validation text
3. Stage 3: TweetEval supervised sentiment fine-tuning
4. Final evaluation on the self-built dataset

This pipeline is intended to separate:
- general supervised sentiment learning from SST-3
- domain adaptation to TweetEval text through masked language modeling
- final TweetEval sentiment fine-tuning

## Files

```text
NLP_project/
├── pipeline_v2.py
├── run_roberta_base_v2.py
├── EXPERIMENT_LOG_V2.md
└── README_V2.md
```

File roles:
- `pipeline_v2.py`
  - Shared logic for the three-stage experiment
- `run_roberta_base_v2.py`
  - Entry point for the RoBERTa-base V2 pipeline
- `EXPERIMENT_LOG_V2.md`
  - Log file for V2 experiments only

## Entry Point

Run the full V2 pipeline:

```bash
python run_roberta_base_v2.py
```

Run a single stage:

```bash
python run_roberta_base_v2.py --stage stage1
python run_roberta_base_v2.py --stage stage2
python run_roberta_base_v2.py --stage stage3
```

## CLI Parameters

These parameters are currently supported by `run_roberta_base_v2.py` through `pipeline_v2.py`.

### Core execution

```bash
--stage {all,stage1,stage2,stage3}
--data-dir PATH
--output-root PATH
```

### Resume from existing checkpoints

```bash
--stage1-model PATH
--stage2-model PATH
```

These are mainly used when running only Stage 2 or Stage 3.

### Batch size

```bash
--train-batch-size INT
--eval-batch-size INT
```

Defaults:
- GPU: `train_batch_size=32`
- CPU: `train_batch_size=8`

### Epoch control

```bash
--epochs INT
--stage1-epochs INT
--stage2-epochs INT
--stage3-epochs INT
```

Behavior:
- `--epochs` provides the shared default
- stage-specific arguments override it for their own stage only

### Early stopping

```bash
--early-stopping-patience INT
```

### Stage 1 weighted loss

```bash
--stage1-weighted-loss
```

If provided, Stage 1 uses weighted cross entropy.

### MLM masking ratio

```bash
--mlm-probability FLOAT
```

Default:
- `0.15`

## Example Commands

Run the full pipeline with stage-specific epochs:

```bash
python run_roberta_base_v2.py --stage1-epochs 4 --stage2-epochs 3 --stage3-epochs 3
```

Run Stage 1 with weighted loss:

```bash
python run_roberta_base_v2.py --stage stage1 --stage1-epochs 4 --stage1-weighted-loss
```

Run Stage 2 MLM starting from an existing Stage 1 checkpoint:

```bash
python run_roberta_base_v2.py --stage stage2 --stage1-model ./models/roberta_base_v2_stage1_sst_model --stage2-epochs 3
```

Run Stage 3 from an existing Stage 2 MLM checkpoint:

```bash
python run_roberta_base_v2.py --stage stage3 --stage2-model ./models/roberta_base_v2_stage2_mlm_model --stage3-epochs 3
```

## Outputs

The V2 pipeline writes separate artifacts so it does not overwrite the original experiment line.

Stage 1 outputs:
- `models/roberta_base_v2_stage1_sst_model/`
- `reports/roberta_base_v2_stage1_sst_results.csv`
- `reports/plots/roberta_base_v2_stage1_sst_confusion_matrix.png`

Stage 2 outputs:
- `models/roberta_base_v2_stage2_mlm_model/`
- `reports/roberta_base_v2_stage2_mlm_report.json`

Stage 3 outputs:
- `models/roberta_base_v2_stage3_tweet_model/`
- `reports/roberta_base_v2_stage3_tweet_results.csv`
- `reports/plots/roberta_base_v2_stage3_tweet_confusion_matrix.png`
- `reports/roberta_base_v2_stage3_selfbuilt_results.csv`
- `reports/plots/roberta_base_v2_stage3_selfbuilt_confusion_matrix.png`

## Suggested Logging Practice

For each run, record the exact command in `EXPERIMENT_LOG_V2.md`, then copy:
- Stage 1 metrics
- Stage 2 MLM report values
- Stage 3 TweetEval metrics
- Final self-built metrics

This makes it much easier to compare whether the MLM stage is helping or not.
