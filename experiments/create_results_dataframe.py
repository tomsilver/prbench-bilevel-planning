"""Script to create a pandas DataFrame from experiment results.

Examples:
  python experiments/create_results_dataframe.py --results_dir logs/2025-08-01/14-36-09
"""

import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Any


import pandas as pd


def _main(results_dir: Path, columns: list[str], config_columns: list[str], mean_agg_columns: list[str],
          output_file: Path | None = None) -> None:
    
    # Can only aggregate over columns that exist.
    assert set(mean_agg_columns).issubset(set(columns) | set(config_columns))
    assert not set(columns) & set(config_columns)  # danger!

    # Load the configs and results from the subdirectories.
    results_with_configs = []
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue
        results_file = subdir / "results.csv"
        config_file = subdir / "config.yaml"
        if not results_file.exists() or not config_file.exists():
            continue
        results = pd.read_csv(results_file)
        config: DictConfig = OmegaConf.load(config_file)
        results_with_configs.append((results, config))

    # Combine everything into one dataframe.
    combined_data: list[dict[str, Any]] = []
    for results, config in results_with_configs:
        # Get the config columns once.
        config_data: dict[str, Any] = {}
        for col in config_columns:
            config_data[col] = OmegaConf.select(config, col)
        for _, row in results.iterrows():
            combined_row: dict[str, Any] = config_data.copy()
            # Extract regular columns.
            for col in columns:
                combined_row[col] = row[col]
            combined_data.append(combined_row)

    # Combine into a larger dataframe.
    combined_df = pd.DataFrame(combined_data)
    print(combined_df)
    


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
        default=["success", "steps", "planning_time", "eval_episode"],
        help="Specific metric columns to include (default: success, steps, planning_time)"
    )
    
    parser.add_argument(
        "--config_columns",
        nargs="*",
        default=["env.env_name", "seed", "max_abstract_plans", "samples_per_step"],
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