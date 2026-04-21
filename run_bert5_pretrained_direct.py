"""
Runner for the BERT-5class pretrained model (mapped to 3 classes).
"""

from pretrained_pipeline import map_bert_stars_to_id, parse_args, run_hf_direct


BERT5_CHECKPOINT = "nlptown/bert-base-multilingual-uncased-sentiment"


if __name__ == "__main__":
    args = parse_args(description="Run BERT-5class pretrained direct-call (mapped to 3-class).")
    run_hf_direct(
        args=args,
        checkpoint=BERT5_CHECKPOINT,
        model_display_name="BERT-5class (pretrained, direct call)",
        label_mapper=map_bert_stars_to_id,
        progress_desc="BERT-5class inference",
    )

