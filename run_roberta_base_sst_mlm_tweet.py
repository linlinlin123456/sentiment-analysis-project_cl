"""
Runner for the three-stage RoBERTa-base experiment:
SST-3 supervised -> TweetEval MLM -> TweetEval supervised -> self-built test
"""

from pipeline_v2 import parse_args, run_three_stage_experiment


def main() -> None:
    """Run the RoBERTa-base V2 three-stage experiment."""
    args = parse_args()
    run_three_stage_experiment(
        args=args,
        experiment_slug="roberta_base_v2",
        base_model_name="roberta-base",
        model_display_name="RoBERTa Base",
    )


if __name__ == "__main__":
    main()
