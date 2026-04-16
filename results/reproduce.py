from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_LTST_RUN = REPO_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"
DEFAULT_SOLAR_ALIGNMENT = REPO_ROOT / "artifact" / "runs" / "solar_20260408T051653Z" / "solar_alignment.csv"
DEFAULT_SOLAR_FIGURE = REPO_ROOT / "artifact" / "figures" / "figure2_solar_alignment_plot.png"


def run_command(args: list[str]) -> None:
    print("+", " ".join(f'"{part}"' if " " in part else part for part in args), flush=True)
    subprocess.run(args, check=True)


def find_latest_ltst_run() -> Path:
    runs_root = REPO_ROOT / "artifact" / "runs"
    candidates = sorted(path for path in runs_root.glob("ltst_full_86_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No ltst_full_86_* run directory found under {runs_root}")
    return candidates[-1]


def run_ltst_full(records: str) -> None:
    script = REPO_ROOT / "artifact" / "ltst_full_86_study.py"
    run_command(
        [
            sys.executable,
            str(script),
            "--records",
            records,
            "--db",
            "ltstdb",
            "--st-ext",
            "stc",
            "--beat-ext",
            "atr",
            "--top-n",
            "1102",
            "--checkpoint-every",
            "5",
            "--save-beat-level",
        ]
    )


def run_ltst_hmm(run_dir: Path | None) -> None:
    base_run_dir = run_dir.resolve() if run_dir else find_latest_ltst_run()
    hmm_script = REPO_ROOT / "artifact" / "ltst_transition_hmm.py"
    derive_script = RESULTS_DIR / "derive_ltst_transition_hmm_metrics.py"
    output_dir = base_run_dir / "transition_hmm_onset_cooldown32"

    run_command(
        [
            sys.executable,
            str(hmm_script),
            "--run-dir",
            str(base_run_dir),
            "--out-dir-name",
            "transition_hmm_onset_cooldown32",
            "--alert-active-tail-beats",
            "16",
            "--alert-max-beats",
            "96",
            "--alert-cooldown-beats",
            "32",
        ]
    )
    run_command(
        [
            sys.executable,
            str(derive_script),
            "--run-dir",
            str(output_dir),
            "--pre-event-beats",
            "500",
        ]
    )


def run_solar_hmm() -> None:
    script = REPO_ROOT / "solar.py"
    run_command(
        [
            sys.executable,
            str(script),
            "--write-feature-csv",
            "--write-alignment-csv",
            "--beta-short",
            "0.10",
            "--beta-long",
            "0.01",
            "--projective-min-state-energy",
            "1e-8",
            "--hmm-rolling-window-minutes",
            "15",
            "--hmm-merge-gap-minutes",
            "15",
            "--hmm-alert-active-tail-minutes",
            "30",
            "--hmm-alert-max-minutes",
            "180",
            "--hmm-alert-cooldown-minutes",
            "60",
        ]
    )


def run_solar_hmm_transition_only_dev() -> None:
    script = REPO_ROOT / "solar.py"
    run_command(
        [
            sys.executable,
            str(script),
            "--start-time",
            "2007-01-01",
            "--end-time",
            "2008-12-31",
            "--write-feature-csv",
            "--beta-short",
            "0.20",
            "--beta-long",
            "0.003",
            "--projective-min-state-energy",
            "1e-8",
            "--hmm-rolling-window-minutes",
            "15",
            "--hmm-merge-gap-minutes",
            "5",
            "--hmm-alert-active-tail-minutes",
            "0",
            "--hmm-alert-max-minutes",
            "180",
            "--hmm-alert-cooldown-minutes",
            "180",
        ]
    )


def run_solar_figure(alignment_csv: Path, output: Path) -> None:
    script = RESULTS_DIR / "reproduce_solar_alignment_figure.py"
    run_command(
        [
            sys.executable,
            str(script),
            "--alignment-csv",
            str(alignment_csv.resolve()),
            "--output",
            str(output.resolve()),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform reproduction runner for the finished cardiac and solar results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ltst_full = subparsers.add_parser("ltst-full", help="Rebuild the 86-record LTST cohort from PhysioNet.")
    ltst_full.add_argument("--records", default="all", help="Comma-separated LTST records or 'all'.")

    ltst_hmm = subparsers.add_parser("ltst-hmm", help="Run the finalized LTST transition-HMM and derive beat-level metrics.")
    ltst_hmm.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Base ltst_full_86_* run directory. Defaults to the latest matching run under artifact/runs.",
    )

    subparsers.add_parser("solar-hmm", help="Run the finalized layered solar HMM on the OMNI cache.")
    subparsers.add_parser(
        "solar-hmm-transition-dev",
        help="Run the 2007-2008 anchored solar transition-only dev pass with the selected beta pair and longer refractory.",
    )

    solar_figure = subparsers.add_parser("solar-figure", help="Regenerate the Figure 2 solar alignment plot from an alignment CSV.")
    solar_figure.add_argument(
        "--alignment-csv",
        type=Path,
        default=DEFAULT_SOLAR_ALIGNMENT,
        help="Solar alignment CSV to plot.",
    )
    solar_figure.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SOLAR_FIGURE,
        help="Output PNG path for the regenerated figure.",
    )

    all_parser = subparsers.add_parser("all", help="Run the full LTST cohort build, LTST HMM, and solar HMM in sequence.")
    all_parser.add_argument("--records", default="all", help="Comma-separated LTST records or 'all'.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ltst-full":
        run_ltst_full(records=args.records)
        return
    if args.command == "ltst-hmm":
        run_ltst_hmm(run_dir=args.run_dir)
        return
    if args.command == "solar-hmm":
        run_solar_hmm()
        return
    if args.command == "solar-hmm-transition-dev":
        run_solar_hmm_transition_only_dev()
        return
    if args.command == "solar-figure":
        run_solar_figure(alignment_csv=args.alignment_csv, output=args.output)
        return
    if args.command == "all":
        run_ltst_full(records=args.records)
        run_ltst_hmm(run_dir=None)
        run_solar_hmm()
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
