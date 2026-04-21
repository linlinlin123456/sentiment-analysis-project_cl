# Experiment Log V2

This log is for the new experiment pipeline only.

The original experiment records remain in:
- `EXPERIMENT_LOG.md`

## New Pipeline

The new workflow is defined as:

1. Stage 1: SST-3 supervised sentiment training
2. Stage 2: MLM on TweetEval train + dev text
3. Stage 3: TweetEval supervised sentiment fine-tuning
4. Final test on the self-built dataset

## Purpose

This experiment line is designed to separate three different learning objectives:

- Stage 1 teaches general 3-class sentiment structure from SST-3
- Stage 2 adapts the language model to TweetEval text style through masked language modeling
- Stage 3 performs final supervised sentiment fine-tuning on TweetEval
- The final self-built evaluation checks whether the full pipeline generalizes beyond the benchmark dataset

## Planned Comparison Questions

This new log should help answer the following questions:

1. Does MLM adaptation on TweetEval text improve downstream sentiment classification?
2. Does the three-stage pipeline outperform the previous direct TweetEval baseline?
3. Does the three-stage pipeline improve performance on the self-built dataset?
4. Does Stage 1 still help after adding Stage 2 MLM adaptation?

## Default Record Template

Use the following template for each run.

---

## Run N

### Pipeline

- Stage 1: SST-3 supervised sentiment training
- Stage 2: MLM on TweetEval train + dev text
- Stage 3: TweetEval supervised sentiment fine-tuning
- Final test: self-built dataset

### Parameters

| Item | Value |
|---|---|
| Base model | |
| Stage selection | `all` / `stage1` / `stage2` / `stage3` |
| Data dir | |
| Output root | |
| Stage 1 model path | |
| Stage 2 model path | |
| Stage 1 epochs | |
| Stage 1 learning rate | |
| Stage 1 weighted loss flag | |
| Stage 2 MLM epochs | |
| Stage 2 MLM learning rate | |
| Stage 2 MLM corpus | `TweetEval train + dev` |
| MLM probability | |
| Stage 3 epochs | |
| Stage 3 learning rate | |
| Train batch size | |
| Eval batch size | |
| Early stopping patience | |
| Seed | `42` |
| Command | |

### Stage 1 Results

| Metric | Value |
|---|---:|
| Dataset | `sst_3class` |
| Accuracy | `0.7477` |
| Precision Macro | |
| Recall Macro | |
| F1 Macro | `0.6090` |
| F1 Negative | `0.8166` |
| F1 Neutral | `0.1730` |
| F1 Positive | `0.8373` |

### Stage 2 MLM Notes

| Item | Value |
|---|---|
| Training completed | `Yes` |
| Model path | |
| Eval loss | |
| Perplexity | |
| Notes | Results not pasted yet |

### Stage 3 Results

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.7382` |
| Precision Macro | |
| Recall Macro | |
| F1 Macro | `0.7353` |
| F1 Negative | `0.7120` |
| F1 Neutral | `0.7292` |
| F1 Positive | `0.7648` |

### Final Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | `0.7550` |
| Precision Macro | `0.761242` |
| Recall Macro | `0.750915` |
| F1 Macro | `0.748932` |
| F1 Negative | `0.706897` |
| F1 Neutral | `0.686567` |
| F1 Positive | `0.853333` |

### Interpretation

- 
- 
- 

---

## Current Status

## Run 1

### Pipeline

- Stage 1: SST-3 supervised sentiment training
- Stage 2: MLM on TweetEval train + dev text
- Stage 3: TweetEval supervised sentiment fine-tuning
- Final test: self-built dataset

### Parameters

| Item | Value |
|---|---|
| Base model | `roberta-base` |
| Stage selection | `all` |
| Data dir | default |
| Output root | default |
| Stage 1 model path | auto-generated |
| Stage 2 model path | auto-generated |
| Stage 1 epochs | `1` |
| Stage 1 learning rate | `1e-5` |
| Stage 1 weighted loss flag | not specified |
| Stage 2 MLM epochs | `2` |
| Stage 2 MLM learning rate | `1e-5` |
| Stage 2 MLM corpus | `TweetEval train + dev` |
| MLM probability | default (`0.15`) |
| Stage 3 epochs | `3` |
| Stage 3 learning rate | `1e-5` |
| Train batch size | default |
| Eval batch size | default |
| Early stopping patience | `2` |
| Seed | `42` |
| Metric | `f1_macro` |
| Command | pending |

### Stage 1 Results

| Metric | Value |
|---|---:|
| Dataset | `sst_3class` |
| Accuracy | |
| Precision Macro | |
| Recall Macro | |
| F1 Macro | |
| F1 Negative | |
| F1 Neutral | |
| F1 Positive | |

### Stage 2 MLM Notes

| Item | Value |
|---|---|
| Training completed | |
| Model path | |
| Eval loss | |
| Perplexity | |
| Notes | |

### Stage 3 Results

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | |
| Precision Macro | |
| Recall Macro | |
| F1 Macro | |
| F1 Negative | |
| F1 Neutral | |
| F1 Positive | |

### Final Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | |
| Precision Macro | |
| Recall Macro | |
| F1 Macro | |
| F1 Negative | |
| F1 Neutral | |
| F1 Positive | |

### Interpretation

- Stage 1 SST performance is weak on the neutral class, even in the V2 pipeline.
- Stage 3 TweetEval performance is competitive and appears stronger than the earlier transfer runs from the original pipeline.
- Self-built performance is solid and now fully recorded.

## Run 2

### Pipeline

- Stage 1: SST-3 supervised sentiment training
- Stage 2: MLM on TweetEval train + dev text
- Stage 3: TweetEval supervised sentiment fine-tuning
- Final test: self-built dataset

### Parameters

| Item | Value |
|---|---|
| Base model | `roberta-base` |
| Stage selection | `all` |
| Data dir | default |
| Output root | default |
| Stage 1 model path | auto-generated |
| Stage 2 model path | auto-generated |
| Stage 1 epochs | `1` |
| Stage 1 learning rate | `1e-5` |
| Stage 1 weighted loss flag | `enabled` |
| Stage 2 MLM epochs | `2` |
| Stage 2 MLM learning rate | `1e-5` |
| Stage 2 MLM corpus | `TweetEval train + dev` |
| MLM probability | default (`0.15`) |
| Stage 3 epochs | `3` |
| Stage 3 learning rate | `1e-5` |
| Train batch size | default |
| Eval batch size | default |
| Early stopping patience | `2` |
| Seed | `42` |
| Metric | `f1_macro` |
| Command | `python run_roberta_base_v2.py --stage1-epochs 1 --stage2-epochs 2 --stage3-epochs 3 --early-stopping-patience 2 --stage1-weighted-loss` |

### Stage 1 Results

| Metric | Value |
|---|---:|
| Dataset | `sst_3class` |
| Accuracy | `0.724895` |
| Precision Macro | `0.673273` |
| Recall Macro | `0.670949` |
| F1 Macro | `0.670361` |
| F1 Negative | `0.773756` |
| F1 Neutral | `0.390244` |
| F1 Positive | `0.847082` |

### Stage 2 MLM Notes

| Item | Value |
|---|---|
| Training completed | `Yes` |
| Model path | |
| Eval loss | |
| Perplexity | |
| Notes | Results not pasted yet |

### Stage 3 Results

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.736060` |
| Precision Macro | `0.734099` |
| Recall Macro | `0.728774` |
| F1 Macro | `0.731228` |
| F1 Negative | `0.700930` |
| F1 Neutral | `0.732143` |
| F1 Positive | `0.760611` |

### Final Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | `0.740000` |
| Precision Macro | `0.755119` |
| Recall Macro | `0.735763` |
| F1 Macro | `0.733011` |
| F1 Negative | `0.684685` |
| F1 Neutral | `0.666667` |
| F1 Positive | `0.847682` |

### Interpretation

- Weighted loss substantially improved Stage 1 neutral-class performance compared with Run 1.
- The Stage 3 TweetEval result became slightly worse than Run 1.
- The self-built result also became slightly worse than Run 1.
- In this V2 configuration, Stage 1 weighted loss helped the source task but did not improve the final target-task outcomes.

## Run 3

### Pipeline

- Stage 2: MLM on TweetEval train + dev text
- Stage 3: TweetEval supervised sentiment fine-tuning
- Final test: self-built dataset
- Stage 1: skipped

### Parameters

| Item | Value |
|---|---|
| Base model | `roberta-base` |
| Stage selection | `stage2 + stage3` |
| Data dir | default |
| Output root | default |
| Stage 1 model path | not used |
| Stage 2 model path | auto-generated |
| Stage 1 epochs | skipped |
| Stage 1 learning rate | skipped |
| Stage 1 weighted loss flag | not used |
| Stage 2 MLM epochs | `2` |
| Stage 2 MLM learning rate | `1e-5` |
| Stage 2 MLM corpus | `TweetEval train + dev` |
| MLM probability | default (`0.15`) |
| Stage 3 epochs | `3` |
| Stage 3 learning rate | `1e-5` |
| Train batch size | default |
| Eval batch size | default |
| Early stopping patience | `2` |
| Seed | `42` |
| Metric | `f1_macro` |
| Command | `python run_roberta_base_mlm_stage3.py --stage2-epochs 2 --stage3-epochs 3 --early-stopping-patience 2` |

### Stage 2 MLM Notes

| Item | Value |
|---|---|
| Training completed | `Yes` |
| Model path | `models/roberta_base_mlm_stage3_stage2_mlm_model/` |
| Eval loss | |
| Perplexity | |
| Notes | MLM report values were not pasted |

### Stage 3 Results

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.742905` |
| Precision Macro | `0.738884` |
| Recall Macro | `0.740093` |
| F1 Macro | `0.739322` |
| F1 Negative | `0.712246` |
| F1 Neutral | `0.734885` |
| F1 Positive | `0.770833` |

### Final Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | `0.790000` |
| Precision Macro | `0.803819` |
| Recall Macro | `0.788206` |
| F1 Macro | `0.783012` |
| F1 Negative | `0.714286` |
| F1 Neutral | `0.742857` |
| F1 Positive | `0.891892` |

### Interpretation

- This run isolates the effect of Tweet MLM without SST Stage 1.
- On TweetEval, it outperformed the previous direct baseline.
- On the self-built dataset, it also outperformed the previous direct baseline.
- This is the strongest evidence so far that Tweet-domain MLM is useful, while SST Stage 1 is not necessary for improvement in the current setup.
