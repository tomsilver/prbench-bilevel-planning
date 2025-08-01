"""Script to create a pandas DataFrame from experiment results.

Examples:
    TODO
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _main(results_dir: Path, columns: list[str], config_columns: list[str], mean_agg_columns: list[str],
          output_file: Path | None = None) -> None:
    import ipdb; ipdb.set_trace()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create aggregated DataFrame from experiment results",
    )
    
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing experiment results"
    )
    
    parser.add_argument(
        "--columns",
        nargs="*",
        default=["success", "steps", "planning_time"],
        help="Specific metric columns to include (default: success, steps, planning_time)"
    )
    
    parser.add_argument(
        "--config_columns",
        nargs="*",
        default=["env"],
        help="Specific config columns to include (default: env)."
    )
    
    parser.add_argument(
        "--mean_agg_columns",
        nargs="*",
        default=["seed", "eval_episode"],
        help="Columns to aggregate over using mean (default: seed, eval_episode)"
    )
    
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Output CSV file path (if not specified, prints to stdout)"
    )
    
    args = parser.parse_args()
    _main(args.results_dir, args.columns, args.config_columns, args.mean_agg_columns,
          args.output_file)