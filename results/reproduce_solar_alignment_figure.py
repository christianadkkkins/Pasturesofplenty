from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import solar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the solar pooled alignment figure from a saved alignment CSV.")
    parser.add_argument("--alignment-csv", type=Path, required=True, help="Path to a solar_alignment.csv file.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    args = parser.parse_args()

    alignment_csv = args.alignment_csv.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alignment_df = pd.read_csv(alignment_csv)
    solar.make_pooled_plot(alignment_df, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
