"""
Analyze the class distribution of the SST Stage 1 training split.

This script reports raw label counts, class ratios, imbalance severity, and a
suggested set of inverse-frequency class weights for weighted cross entropy.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datasets" / "sst_3class_train.csv"
LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def main() -> None:
    """Load the SST Stage 1 training split and print imbalance diagnostics."""
    df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "label_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    counts = df["label_id"].value_counts().sort_index()
    total = int(counts.sum())
    ratios = counts / total

    min_count = int(counts.min())
    max_count = int(counts.max())
    imbalance_ratio = max_count / min_count

    class_weights = total / (len(counts) * counts)
    normalized_weights = class_weights / class_weights.mean()

    summary = pd.DataFrame(
        {
            "label_id": counts.index,
            "label_name": [LABEL_NAMES.get(label, str(label)) for label in counts.index],
            "count": counts.values,
            "ratio": ratios.values,
            "suggested_weight": normalized_weights.values,
        }
    )

    print("SST Stage 1 training split:", DATA_PATH)
    print(f"Total samples: {total}")
    print(f"Imbalance ratio (largest/smallest class): {imbalance_ratio:.3f}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    if imbalance_ratio >= 1.5:
        print("Recommendation: weighted cross entropy is worth trying.")
        print("Reason: class imbalance is non-trivial and may be contributing to the weak neutral F1.")
    else:
        print("Recommendation: class imbalance alone is probably not severe enough to justify weighted cross entropy.")
        print("Reason: the label distribution is relatively balanced, so the neutral-class issue is more likely caused by class difficulty or representation mismatch.")

    print()
    print("Suggested normalized class weights:")
    print(
        "{"
        + ", ".join(
            f"{int(label)}: {weight:.6f}"
            for label, weight in zip(summary["label_id"], summary["suggested_weight"])
        )
        + "}"
    )


if __name__ == "__main__":
    main()
