"""
Runner for the RoBERTa-3class pretrained model (direct call).
"""

from pretrained_pipeline import map_model_label_to_id, parse_args, run_hf_direct


ROBERTA3_CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment-latest"


if __name__ == "__main__":
    args = parse_args(description="Run RoBERTa-3class pretrained direct-call.")
    run_hf_direct(
        args=args,
        checkpoint=ROBERTA3_CHECKPOINT,
        model_display_name="RoBERTa-3class (pretrained, direct call)",
        label_mapper=map_model_label_to_id,
        progress_desc="RoBERTa-3class inference",
    )

