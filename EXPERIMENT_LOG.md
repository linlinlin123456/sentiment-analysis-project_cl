# Experiment Log

This document records the RoBERTa-base experiments discussed so far.

## Notes

- Model family: `roberta-base`
- Pipeline: `SST-3 -> TweetEval 3-class`
- Seed: `42`
- Best checkpoint metric: `f1_macro`
- Early stopping: enabled in the current codebase
- Stage 1 learning rate: `1e-5`
- Stage 2 learning rate: `5e-6`
- Weight decay: `0.1`
- Batch size: assumed to be the default GPU setting unless manually overridden
- Some run-specific parameters such as exact epoch counts were not explicitly recorded in the chat, so they are marked as unknown when not confirmed

## Experiment 1

### Parameters

| Item | Value |
|---|---|
| Runner | `run_roberta_base.py` |
| Command | `python run_roberta_base.py --stage1-epochs 4 --stage2-epochs 2 --early-stopping-patience 2` |
| Stage 1 epochs | `4` |
| Stage 2 epochs | `2` |
| Train batch size | Likely default GPU batch size |
| Early stopping patience | `2` |
| Stage 1 learning rate | `1e-5` |
| Stage 2 learning rate | `5e-6` |

### Results

#### Stage 1

| Metric | Value |
|---|---:|
| Dataset | `sst_3class` |
| Accuracy | `0.770464` |
| Precision Macro | `0.690093` |
| Recall Macro | `0.680460` |
| F1 Macro | `0.676551` |
| F1 Negative | `0.831601` |
| F1 Neutral | `0.335196` |
| F1 Positive | `0.862857` |

#### Stage 2

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.734558` |
| Precision Macro | `0.730574` |
| Recall Macro | `0.728607` |
| F1 Macro | `0.729575` |
| F1 Negative | `0.695883` |
| F1 Neutral | `0.724964` |
| F1 Positive | `0.767878` |

### Observations

- Stage 1 learned negative and positive classes well.
- Stage 1 neutral performance was much weaker than the other two classes.
- Stage 2 recovered a much more balanced class-level performance on TweetEval.

## Experiment 2

### Parameters

| Item | Value |
|---|---|
| Runner | `run_roberta_base.py` |
| Command | `python run_roberta_base.py --stage1-epochs 2 --stage2-epochs 3` |
| Stage 1 epochs | `2` |
| Stage 2 epochs | `3` |
| Train batch size | Likely default GPU batch size |
| Early stopping patience | Default value in codebase |
| Stage 1 learning rate | `1e-5` |
| Stage 2 learning rate | `5e-6` |

### Results

#### Stage 1

| Metric | Value |
|---|---:|
| Dataset | `sst_3class` |
| Accuracy | `0.766245` |
| Precision Macro | `0.679713` |
| Recall Macro | `0.663898` |
| F1 Macro | `0.651979` |
| F1 Negative | `0.830831` |
| F1 Neutral | `0.269592` |
| F1 Positive | `0.855513` |

#### Stage 2

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.735392` |
| Precision Macro | `0.729787` |
| Recall Macro | `0.734914` |
| F1 Macro | `0.732249` |
| F1 Negative | `0.706339` |
| F1 Neutral | `0.722967` |
| F1 Positive | `0.767442` |

### Observations

- Stage 1 neutral performance dropped further compared with Experiment 1.
- Stage 2 still remained stable and slightly improved in macro F1.
- The Stage 1 issue appears to be class-specific rather than a full-model failure.

## Comparison Summary

| Experiment | Stage 1 F1 Macro | Stage 1 F1 Neutral | Stage 2 F1 Macro | Stage 2 F1 Neutral |
|---|---:|---:|---:|---:|
| Experiment 1 | `0.676551` | `0.335196` | `0.729575` | `0.724964` |
| Experiment 2 | `0.651979` | `0.269592` | `0.732249` | `0.722967` |

## Experiment 3

### Parameters

| Item | Value |
|---|---|
| Runner | `run_roberta_base.py` |
| Command | `python run_roberta_base.py --stage1-epochs 4 --stage2-epochs 2 --early-stopping-patience 2` |
| Stage 1 epochs | `4` |
| Stage 2 epochs | `2` |
| Train batch size | Likely default GPU batch size |
| Early stopping patience | `2` |
| Stage 1 learning rate | `1e-5` |
| Stage 2 learning rate | `5e-6` |
| Stage 1 weighted cross entropy | `Enabled` |

### Results

#### Stage 1

| Metric | Value |
|---|---:|
| Dataset | `sst_3class` |
| Accuracy | `0.754430` |
| Precision Macro | `0.697916` |
| Recall Macro | `0.696819` |
| F1 Macro | `0.696410` |
| F1 Negative | `0.806306` |
| F1 Neutral | `0.422414` |
| F1 Positive | `0.860511` |

#### Stage 2

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.730551` |
| Precision Macro | `0.727141` |
| Recall Macro | `0.726007` |
| F1 Macro | `0.726489` |
| F1 Negative | `0.697468` |
| F1 Neutral | `0.718287` |
| F1 Positive | `0.763713` |

#### Stage 1 Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | `0.715000` |
| Precision Macro | `0.713748` |
| Recall Macro | `0.710292` |
| F1 Macro | `0.710976` |
| F1 Negative | `0.716418` |
| F1 Neutral | `0.608000` |
| F1 Positive | `0.808511` |

#### Stage 2 Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | `0.775000` |
| Precision Macro | `0.786917` |
| Recall Macro | `0.772086` |
| F1 Macro | `0.769274` |
| F1 Negative | `0.719298` |
| F1 Neutral | `0.710145` |
| F1 Positive | `0.878378` |

### Observations

- Weighted cross entropy clearly improved Stage 1 macro performance.
- The largest gain was on the neutral class, which rose from `0.335196` in Experiment 1 to `0.422414`.
- Stage 1 accuracy decreased, but macro F1 improved, which is expected when the loss becomes more balanced across classes.
- Stage 2 did not improve with this change and became slightly worse than Experiment 1 and Experiment 2.
- This suggests the weighted loss helped the source task itself, but did not improve transfer quality to TweetEval in this configuration.
- On the self-built dataset, the weighted-transfer Stage 2 result (`f1_macro = 0.769274`) is strong, but still slightly below the direct TweetEval baseline (`0.773745`).

## Updated Comparison Summary

| Experiment | Stage 1 Setup | Stage 1 F1 Macro | Stage 1 F1 Neutral | Stage 2 F1 Macro | Stage 2 F1 Neutral |
|---|---|---:|---:|---:|---:|
| Experiment 1 | Baseline | `0.676551` | `0.335196` | `0.729575` | `0.724964` |
| Experiment 2 | Baseline | `0.651979` | `0.269592` | `0.732249` | `0.722967` |
| Experiment 3 | Weighted cross entropy | `0.696410` | `0.422414` | `0.726489` | `0.718287` |

## Experiment 4

### Parameters

| Item | Value |
|---|---|
| Runner | `run_roberta_base_direct_tweet.py` |
| Command | Direct TweetEval baseline |
| Training setup | `roberta-base -> TweetEval` |
| Stage 1 | Not used |
| Stage 2 / Direct training | Used |

### Results

#### TweetEval

| Metric | Value |
|---|---:|
| Dataset | `tweeteval_sentiment_3class` |
| Accuracy | `0.742237` |
| Precision Macro | `0.741936` |
| Recall Macro | `0.733834` |
| F1 Macro | `0.737594` |
| F1 Negative | `0.707252` |
| F1 Neutral | `0.736336` |
| F1 Positive | `0.769194` |

#### Self-Built Evaluation

| Metric | Value |
|---|---:|
| Dataset | `selfbuilt_database_corrected` |
| Accuracy | `0.780000` |
| Precision Macro | `0.795148` |
| Recall Macro | `0.778146` |
| F1 Macro | `0.773745` |
| F1 Negative | `0.714286` |
| F1 Neutral | `0.728571` |
| F1 Positive | `0.878378` |

### Observations

- The direct TweetEval baseline is currently the strongest TweetEval result among the RoBERTa-base experiments.
- Its `f1_macro = 0.737594` is higher than all transfer-learning runs recorded so far.
- This suggests that SST Stage 1 pretraining has not yet produced a transfer gain for `roberta-base`.
- The self-built evaluation result is also strong, but it is not yet directly comparable with the earlier transfer runs because those runs were completed before self-built evaluation was added to the pipeline.

## Direct Comparison on TweetEval

| Experiment | Training Setup | TweetEval F1 Macro |
|---|---|---:|
| Experiment 1 | `roberta-base -> SST -> TweetEval` | `0.729575` |
| Experiment 2 | `roberta-base -> SST -> TweetEval` | `0.732249` |
| Experiment 3 | `roberta-base -> SST(weighted CE) -> TweetEval` | `0.726489` |
| Experiment 4 | `roberta-base -> TweetEval` | `0.737594` |

## Direct Comparison on Self-Built Evaluation

| Experiment | Training Setup | Self-Built F1 Macro |
|---|---|---:|
| Experiment 3 Stage 1 | `roberta-base -> SST(weighted CE)` | `0.710976` |
| Experiment 3 Stage 2 | `roberta-base -> SST(weighted CE) -> TweetEval` | `0.769274` |
| Experiment 4 | `roberta-base -> TweetEval` | `0.773745` |

## Current Interpretation

- The main weakness is Stage 1 neutral-class learning on SST-3.
- Stage 2 performance is comparatively stable, so Stage 1 is the better place to optimize next.
- The next diagnostic step is to inspect SST-3 class distribution and decide whether class imbalance is large enough to justify weighted cross entropy.
