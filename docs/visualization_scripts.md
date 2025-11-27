# 📊 Visualization Scripts Guide

This document explains how to run the visualization scripts available in this project, which generate adoption heatmaps, PRIM peeling trajectory figures, and PRIM demographic tables.

## ⚠️ Important Usage Note: Modular Execution

Following the recent code refactoring, all visualization scripts use **relative imports** (`from ._module import...`). Therefore, they must be executed as **Python modules** using the `python -m` command.

  * **Execution Location:** Always run the commands from the **project root** directory (the directory containing the `src` folder).

-----

## 🛠️ Overview of Scripts

| Script                          | Description                                                                                             | Output                         |
| :------------------------------ | :------------------------------------------------------------------------------------------------------ | :----------------------------- |
| `adoption_heatmap_generator.py` | Generates heatmaps of the adoption rate as a function of Trust and Income across policy scenarios.      | PNG figures saved in `/tmp/`   |
| `prim_trajectory.py`            | Generates the PRIM peeling trajectory figure (coverage vs density) for different policy scenarios.      | PNG figure saved in `/tmp/`    |
| `demographic_table.py`          | Generates a Markdown table with demographic profiles of high-adoption segments across policy scenarios. | Markdown file saved in `/tmp/` |

-----

## ⚙️ Installation

Ensure you have installed all dependencies listed in `requirements.txt`. The following are required for visualization scripts:

```bash
pip install -r requirements.txt
```

**Optional:** If running the Markdown table script, ensure `tabulate` is installed:

```bash
pip install tabulate
```

-----

## 🚀 Usage

Navigate to the project root and run the scripts using the **module execution** syntax (`python -m`).

### 🌡️ Adoption Heatmap

Execute the modular script:

```bash
python -m src.visualization.adoption_heatmap_generator
```

This will generate heatmaps of adoption rate for each scenario and save them to `/tmp/`.

### 📈 PRIM Peeling Trajectory

Execute the modular script:

```bash
python -m src.visualization.prim_trajectory
```

This will generate the PRIM peeling trajectory figure and save it to `/tmp/`. The console will display the saved file path.

### 📝 PRIM Demographic Table

Execute the modular script:

```bash
python -m src.visualization.demographic_table
```

This will generate a Markdown table of demographic profiles and save it to `/tmp/`. The console will display the saved file path.

-----

## 💾 Output

All visualization scripts save outputs to `/tmp/` by default. Example paths:

  * `/tmp/adoption_heatmap.png`
  * `/tmp/figura2_prim_trajectory.png`
  * `/tmp/demographic_profiles.md`

You can open the Markdown table in any editor or render it in Jupyter notebooks.

## 📝 Notes

  * Scripts can currently read data from available CSV files. In the future, they will also support outputs produced by the MESA simulations.
  * You can customize paths in the scripts if needed, but `/tmp/` is the default for temporary outputs. The default paths are now defined in `src/visualization/_config/settings.py`.
  * For high-quality figures, you may modify figure size and DPI parameters inside the `_config/settings.py` file.