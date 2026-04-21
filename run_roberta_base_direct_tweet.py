"""
Runner for the direct TweetEval baseline using RoBERTa-base.
"""

from training_pipeline import parse_args, run_direct_tweet_experiment


def main() -> None:
    """Run the direct TweetEval baseline with RoBERTa-base.

    Returns:
        None
    """
    args = parse_args()
    run_direct_tweet_experiment(
        args=args,
        experiment_slug="roberta_base",
        base_model_name="roberta-base",
        model_display_name="RoBERTa Base",
    )


if __name__ == "__main__":
    main()
