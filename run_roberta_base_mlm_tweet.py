"""
Runner for the MLM -> TweetEval supervised experiment without SST Stage 1.
"""

from pipeline_v2 import parse_args, run_mlm_stage3_experiment


def main() -> None:
    """Run the RoBERTa-base Tweet MLM -> TweetEval supervised experiment."""
    args = parse_args()
    run_mlm_stage3_experiment(
        args=args,
        experiment_slug="roberta_base_mlm_stage3",
        base_model_name="roberta-base",
        model_display_name="RoBERTa Base MLM",
    )


if __name__ == "__main__":
    main()
