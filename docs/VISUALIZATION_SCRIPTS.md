# 📊 Visualization Scripts Guide

This project provides scripts to generate **adoption heatmaps**, **PRIM trajectory figures**, and **demographic tables** from simulation data.

## ⚠️ Important Note: Modular Execution

All visualization scripts now use **relative imports**, so they must be run as **Python modules** from the **project root**:

```bash
python -m src.visualization.<script_name>
````

---

## 🛠️ Available Scripts

| Script                             | Description                                                                 | Output                    |
| :--------------------------------- | :-------------------------------------------------------------------------- | :------------------------ |
| `adoption_heatmap_generator.py`    | Heatmaps of adoption rate vs Trust and Income for each scenario.           | PNG saved in `/tmp/`      |
| `prim_trajectory.py`               | PRIM peeling trajectories (coverage vs density) for NI, EI, and SI.        | PNG saved in `/tmp/`      |
| `plot_prim_single_trajectory.py`   | Single-scenario PRIM peeling trajectory with optimal box highlighted.      | PNG saved in `/tmp/`      |
| `demographic_table.py`             | Markdown table of demographic profiles of high-adoption segments.          | Markdown saved in `/tmp/` |

---

## 🚀 Usage Examples

### 🌡️ Adoption Heatmap

```bash
python -m src.visualization.adoption_heatmap_generator --data-dir data/montecarlo --output /tmp/adoption_montecarlo.png
```

### 📈 PRIM Peeling Trajectory

```bash
python -m src.visualization.prim_trajectory --data-dir data/montecarlo --output /tmp/prim_trajectory.png
python -m src.visualization.plot_prim_single_trajectory --data-dir data/montecarlo --output /tmp/prim_trajectory.png
```

### 📝 Demographic Table

```bash
python -m src.scripts.presentation.generate_all_tables --data-dir data/montecarlo_calibrated_fixed --output /tmp/

---

## 💾 Output

Default outputs are saved to `/tmp/`. Example files:

* `/tmp/adoption_montecarlo.png`
* `/tmp/prim_trajectory.png`
* `/tmp/demographic_profiles.md`

---

## 📝 Notes

* Scripts read data from the specified `--data-dir`.
* Default paths and figure settings are defined in `src/visualization/_config/settings.py`.
* Monte Carlo and MESA simulation outputs can both be used once available.
