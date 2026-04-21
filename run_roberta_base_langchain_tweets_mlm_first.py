"""
Runner for the isolated RoBERTa-base MLM -> SST -> TweetEval pipeline
with SST balancing and optional LangChain refinement.
"""

from pipeline_twitter_roberta_base_langchain_tweets_mlm_first import (
    parse_args,
    run_three_stage_experiment,
)


def main() -> None:
    """Run the isolated RoBERTa-base experimental pipeline."""
    args = parse_args()
    run_three_stage_experiment(
        args=args,
        experiment_slug="roberta_base_langchain_tweets_mlm_first",
        base_model_name="roberta-base",
        model_display_name="RoBERTa Base LangChain Tweets MLM First",
    )


if __name__ == "__main__":
    main()
