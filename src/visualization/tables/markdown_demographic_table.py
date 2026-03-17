from pathlib import Path
from ._utils.tables_config import DEMOGRAPHIC_TABLE_COLUMNS, DEMOGRAPHIC_TABLE_ORDER, DEMOGRAPHIC_TABLE_CAPTION

def render_markdown_table(df, output_path):

    markdown = df.to_markdown(
        index=False,
        floatfmt=".3f"
    )

    with open(output_path, "w") as f:
        f.write(markdown)
