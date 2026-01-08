from pathlib import Path
from ._utils.tables_config import DEMOGRAPHIC_TABLE_COLUMNS, DEMOGRAPHIC_TABLE_ORDER, DEMOGRAPHIC_TABLE_CAPTION

def render_markdown_table(df, output_path: Path):
    df_out = df.rename(columns=DEMOGRAPHIC_TABLE_COLUMNS)
    df_out = df_out[DEMOGRAPHIC_TABLE_ORDER]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(DEMOGRAPHIC_TABLE_CAPTION + "\n" + df_out.to_markdown(index=False))
    print(f"📄 Markdown saved: {output_path}")
