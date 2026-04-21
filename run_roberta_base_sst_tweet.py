"""
Runner for the RoBERTa-base transfer-learning experiment.
"""

from training_pipeline import parse_args, run_transfer_experiment


def main() -> None:
    """Run the RoBERTa-base transfer-learning experiment.

    Returns:
        None
    """
    args = parse_args()
    run_transfer_experiment(
        args=args,
        experiment_slug="roberta_base",
        base_model_name="roberta-base",
        model_display_name="RoBERTa Base",
    )


if __name__ == "__main__":
    main()
