# src/utils/cli.py

from argparse import ArgumentParser
from pathlib import Path
import inspect
import os

def safe_run(main_func):
    """
    Runs a function inside a FileNotFoundError-safe wrapper.
    Ensures consistent CLI error handling across scripts.
    """
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
    """
    Returns a preconfigured ArgumentParser.
    - description: optional, auto-generated from filename if None
    - defaults: optional dictionary of default argument values
    """

    if description is None:
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        filename = os.path.basename(caller_file)
        description = Path(filename).stem.replace("_", " ").capitalize()

    parser = ArgumentParser(description=description)

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

    # Montecarlo arguments
    parser.add_argument(
        "--n-agents",
        type=int,
        help="Number of agents per scenario"
    )

    parser.add_argument(
        "--n-replications",
        type=int,
        help="Number of Monte Carlo replications"
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        help="Number of bins in the heatmap"
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        help="Noise standard deviation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )


    if defaults:
        parser.set_defaults(**defaults)

    return parser