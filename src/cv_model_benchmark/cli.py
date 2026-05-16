from __future__ import annotations

import argparse
from pathlib import Path

from .hardware import hardware_json
from .matrix import expand_matrix, format_markdown_table, write_results_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv-benchmark",
        description="Helpers for the detector compression thesis benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("hardware", help="Print local hardware and PyTorch CUDA info as JSON.")

    plan_parser = subparsers.add_parser("plan", help="Print the configured experiment matrix.")
    plan_parser.add_argument("--models", default="configs/models.yaml")
    plan_parser.add_argument("--experiments", default="configs/experiments.yaml")

    template_parser = subparsers.add_parser(
        "results-template",
        help="Create an empty CSV with the expected result columns.",
    )
    template_parser.add_argument("--output", default="results/experiment_results_template.csv")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "hardware":
        print(hardware_json())
        return

    if args.command == "plan":
        rows = expand_matrix(args.models, args.experiments)
        print(format_markdown_table(rows))
        return

    if args.command == "results-template":
        output_path = write_results_template(Path(args.output))
        print(f"Created {output_path}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
