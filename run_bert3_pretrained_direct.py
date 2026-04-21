"""
Runner for the BERT-3class pretrained model (direct call).
"""

from pretrained_pipeline import map_model_label_to_id, parse_args, run_hf_direct


BERT3_CHECKPOINT = "Priyanka-Balivada/bert-5-epoch-sentiment"


if __name__ == "__main__":
    args = parse_args(description="Run BERT-3class pretrained direct-call.")
    run_hf_direct(
        args=args,
        checkpoint=BERT3_CHECKPOINT,
        model_display_name="BERT-3class (pretrained, direct call)",
        label_mapper=map_model_label_to_id,
        progress_desc="BERT-3class inference",
    )

