from pathlib import Path
import sys

from src.analysis.prim_tables.demographic_table_builder import build_demographic_table
from src.visualization.tables.latex_demographic_table import render_latex_table
from src.visualization.tables.markdown_demographic_table import render_markdown_table
from src.utils.cli_parser import base_parser, safe_run
from src.utils.file_utils import load_csv_or_fail  # opzionale per validazioni extra

def main():
    args = base_parser(defaults={"output_dir": Path("/tmp")}).parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "demographic_profiles.csv"
    summary_csv_path = data_dir / "prim_trajectory_summary.csv"
    raw_csv_path = data_dir / "prim_trajectory_raw.csv"

    # costruiamo DataFrame finale
    df = build_demographic_table(csv_path, summary_csv_path, raw_csv_path)

    # scriviamo output
    render_latex_table(df, output_dir / "demographic_profiles.tex")
    render_markdown_table(df, output_dir / "demographic_profiles.md")
    print(f"✅ Table generation complete. Outputs in {output_dir}")


if __name__ == "__main__":
    safe_run(main)
