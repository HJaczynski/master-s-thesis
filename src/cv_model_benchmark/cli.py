from __future__ import annotations

import argparse
from pathlib import Path

from .hardware import hardware_json
from .matrix import expand_matrix, format_markdown_table, write_results_template
from .nightly import doctor, run_nightly


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

    doctor_parser = subparsers.add_parser("doctor", help="Check overnight-run dependencies and data paths.")
    doctor_parser.add_argument("--config", default="configs/nightly.yaml")
    doctor_parser.add_argument("--profile", default="overnight")

    nightly_parser = subparsers.add_parser("run-nightly", help="Run a profile-driven overnight experiment queue.")
    nightly_parser.add_argument("--config", default="configs/nightly.yaml")
    nightly_parser.add_argument("--models-config", default="configs/models.yaml")
    nightly_parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    nightly_parser.add_argument("--profile", default="overnight")
    nightly_parser.add_argument("--dry-run", action="store_true")
    nightly_parser.add_argument("--limit", type=int)
    nightly_parser.add_argument("--model", action="append", dest="models")
    nightly_parser.add_argument("--variant", action="append", dest="variants")

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

    if args.command == "doctor":
        import json

        print(json.dumps(doctor(args.config, args.profile), indent=2, sort_keys=True))
        return

    if args.command == "run-nightly":
        run_dir = run_nightly(
            config_path=args.config,
            models_path=args.models_config,
            experiments_path=args.experiments_config,
            profile=args.profile,
            dry_run=args.dry_run,
            limit=args.limit,
            only_models=args.models,
            only_variants=args.variants,
        )
        print(run_dir)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
