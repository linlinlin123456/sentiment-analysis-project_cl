"""
Runner for the Bertweet-3class pretrained model (direct call).
"""

from pretrained_pipeline import map_model_label_to_id, parse_args, run_hf_direct


BERTWEET3_CHECKPOINT = "cardiffnlp/bertweet-base-sentiment"


if __name__ == "__main__":
    args = parse_args(description="Run Bertweet-3class pretrained direct-call.")
    run_hf_direct(
        args=args,
        checkpoint=BERTWEET3_CHECKPOINT,
        model_display_name="BERTweet-3class (pretrained, direct call)",
        label_mapper=map_model_label_to_id,
        progress_desc="Bertweet-3class inference",
    )