"""
Generate all presentation tables: demographic profiles and aggregate metrics.

Usage:
    python -m src.scripts.presentation.generate_all_tables \
      --data-dir data/montecarlo_calibrated_fixed \
      --output-dir /tmp/
"""
from pathlib import Path
import sys

from src.analysis.prim_tables.demographic_table_builder import build_demographic_table
from src.visualization.tables.latex_demographic_table import render_latex_table
from src.visualization.tables.markdown_demographic_table import render_markdown_table
from src.utils.cli_parser import base_parser, safe_run


def generate_demographic_tables(data_dir: Path, output_dir: Path) -> None:
    """Generate demographic profile tables (existing functionality)."""
    print("📊 Generating demographic profile tables...")
    
    csv_path = data_dir / "demographic_profiles.csv"
    summary_csv_path = data_dir / "prim_trajectory_summary.csv"
    raw_csv_path = data_dir / "prim_trajectory_raw.csv"

    # Build DataFrame
    df = build_demographic_table(csv_path, summary_csv_path, raw_csv_path)

    # Render outputs
    tex_path = output_dir / "demographic_profiles.tex"
    md_path = output_dir / "demographic_profiles.md"
    
    render_latex_table(df, tex_path)
    render_markdown_table(df, md_path)
    
    print(f"  ✅ {tex_path.absolute()}")
    print(f"  ✅ {md_path.absolute()}")


def generate_aggregate_metrics_tables(data_dir: Path, output_dir: Path) -> None:
    """Generate aggregate metrics by scenario tables (NEW - placeholder)."""
    print("📈 Generating aggregate metrics tables...")
    print("  ⚠️  Not yet implemented - coming next!")
    # TODO: Implement in next step
    # from src.analysis.prim_tables.aggregate_metrics_builder import build_aggregate_metrics_table
    # from src.visualization.tables.latex_aggregate_table import render_latex_aggregate_table
    # from src.visualization.tables.markdown_aggregate_table import render_markdown_aggregate_table
    #
    # df = build_aggregate_metrics_table(data_dir)
    # render_latex_aggregate_table(df, output_dir / "aggregate_metrics_by_scenario.tex")
    # render_markdown_aggregate_table(df, output_dir / "aggregate_metrics_by_scenario.md")


def main():
    """Main entry point."""
    args = base_parser(defaults={"output": Path("/tmp")}).parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()

    # Generate all tables
    generate_demographic_tables(data_dir, output_dir)
    print()
    generate_aggregate_metrics_tables(data_dir, output_dir)
    
    print()
    print(f"✅ Table generation complete. Outputs in {output_dir}")


if __name__ == "__main__":
    safe_run(main)