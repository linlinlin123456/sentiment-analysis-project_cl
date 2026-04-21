"""
Runner for the isolated Twitter-RoBERTa SST -> MLM -> TweetEval pipeline
with SST downsampling and optional LangChain refinement.
"""

from pipeline_twitter_roberta_base_langchain_tweets import (
    parse_args,
    run_three_stage_experiment,
)


def main() -> None:
    """Run the isolated Twitter-RoBERTa experimental pipeline."""
    args = parse_args()
    run_three_stage_experiment(
        args=args,
        experiment_slug="twitter_roberta_base_langchain_tweets",
        base_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_display_name="Twitter RoBERTa Base LangChain Tweets",
    )


if __name__ == "__main__":
    main()
