"""
Runner for the Twitter-RoBERTa transfer-learning experiment.
"""

from training_pipeline import parse_args, run_transfer_experiment


def main() -> None:
    """Run the Twitter-RoBERTa transfer-learning experiment.

    Returns:
        None
    """
    args = parse_args()
    run_transfer_experiment(
        args=args,
        experiment_slug="twitter_roberta",
        base_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_display_name="Twitter RoBERTa",
    )


if __name__ == "__main__":
    main()
