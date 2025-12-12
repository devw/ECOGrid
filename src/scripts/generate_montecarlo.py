#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.cli_parser import base_parser, safe_run
from .presentation.report import show_header, show_config, show_summary
from src.data.montecarlo_generators.calibrated_generator import (
    GeneratorConfig, generate_all_scenarios, save_all_data
)


def run():
    args = base_parser(defaults={
        "n_agents": 10000,
        "n_bins": 20,
        "n_replications": 100,
        "noise_std": 0.05,
        "seed": 42,
        "output": Path("data/montecarlo")
    }).parse_args()

    config = GeneratorConfig(
        n_agents=args.n_agents,
        n_bins=args.n_bins,
        n_replications=args.n_replications,
        noise_std=args.noise_std,
        random_seed=args.seed,
        output_dir=args.output
    )

    show_header("ECOGrid Montecarlo Data Generator", "2.0")
    show_config(vars(config))

    data = generate_all_scenarios(config)
    files = save_all_data(data, config.output_dir, config)

    show_summary(files)


if __name__ == "__main__":
    safe_run(run)
