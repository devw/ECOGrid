from argparse import ArgumentParser
from pathlib import Path
import inspect
import os

def safe_run(main_func):
    try:
        main_func()
    except FileNotFoundError as e:
        import sys
        print(f"❌ {e}")
        sys.exit(1)

def base_parser(
    description: str | None = None,
    defaults: dict | None = None
) -> ArgumentParser:

    if description is None:
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        filename = os.path.basename(caller_file)
        description = Path(filename).stem.replace("_", " ").capitalize()

    parser = ArgumentParser(description=description)

    # Parametri comuni
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing input data files"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the output file"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )

    # Parametri Montecarlo
    parser.add_argument(
        "--n-agents",
        type=int,
        help="Number of agents per scenario (Montecarlo)"
    )

    parser.add_argument(
        "--n-replications",
        type=int,
        help="Monte Carlo replications (Montecarlo)"
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        help="Number of bins in the heatmap (Montecarlo)"
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        help="Noise standard deviation (Montecarlo)"
    )

    # Parametri ABM
    parser.add_argument(
        "--n-consumers",
        type=int,
        help="Number of consumer agents (ABM)"
    )

    parser.add_argument(
        "--n-prosumers",
        type=int,
        help="Number of prosumer agents (ABM)"
    )

    parser.add_argument(
        "--n-grid-agents",
        type=int,
        help="Number of grid agents (ABM)"
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        help="Number of simulation steps (ABM)"
    )

    # Applica eventuali valori di default specifici dello script
    if defaults:
        parser.set_defaults(**defaults)

    return parser
