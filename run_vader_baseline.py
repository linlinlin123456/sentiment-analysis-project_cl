"""
Runner for the VADER baseline sentiment experiment (3-class).
"""

from pretrained_pipeline import parse_args, run_vader


def main() -> None:
    args = parse_args(description="Run VADER baseline (3-class) on a labeled CSV.")
    run_vader(args=args, model_display_name="VADER (baseline)")


if __name__ == "__main__":
    main()

