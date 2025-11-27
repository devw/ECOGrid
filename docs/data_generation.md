# 📦 Data Generation Guide

ECOGrid uses a robust synthetic (dummy) data pipeline to ensure reproducibility, efficient development, and reliable simulation testing. This guide explains why synthetic data is important, how to generate it, and how to validate it.

---

## 🎯 Purpose of Synthetic Data

Synthetic data is essential for:
- **Testing & Development** — Provides consistent and predictable inputs.
- **Reproducibility** — Ensures anyone can reproduce baseline results.
- **Environment Setup** — Enables experimentation before introducing real data.

All generated data is stored in `data/dummy/`.

---

## 1️⃣ Data Generation

Data generation is handled by scripts in `src/scripts/` and `src/data/generators/`.

### A. Full Generation

Generate all datasets with default parameters:

```bash
python src/scripts/generate_dummy_data.py
````

You can also pass parameters to customize the data generation process.
For example, to set the number of replications:

```bash
python src/scripts/generate_dummy_data.py --n-replications 300
```

---

## 🔧 Available CLI Parameters

| Parameter          | Type  | Default          | Description                                                                         |
| ------------------ | ----- | ---------------- | ----------------------------------------------------------------------------------- |
| `--n-agents`       | int   | `10000`          | Number of agents generated per scenario.                                            |
| `--n-bins`         | int   | `20`             | Number of bins per axis (e.g., 20×20 grid) for heatmap generation.                  |
| `--noise-std`      | float | `0.05`           | Standard deviation of random noise added to adoption rates to simulate uncertainty. |
| `--seed`           | int   | `42`             | Random seed for ensuring reproducibility.                                           |
| `--output`         | Path  | `data/dummy`     | Output directory for generated CSV/JSON files.                                      |
| `--n-replications` | int   | varies by config | Number of replications used for adoption rate sampling.                             |

### Example: Full Custom Generation

```bash
python src/scripts/generate_dummy_data.py \
    --n-agents 5000 \
    --n-replications 300 \
    --n-bins 15 \
    --noise-std 0.03 \
    --seed 1234 \
    --output data/custom_low_noise
```

---

## B. Generated Datasets (6 Total)

| Dataset                     | Description                                 | Usage                     |
| --------------------------- | ------------------------------------------- | ------------------------- |
| **👤 Agents**               | Demographic attributes (Trust, Income)      | ABM initialization        |
| **📈 Adoption Rates**       | Scenario-specific adoption rates (NI/SI/EI) | Benchmarks, PRIM analysis |
| **🗺️ Heatmap Grid**        | Sampled Trust–Income combinations           | Heatmap visualizations    |
| **📦 PRIM Boxes**           | Critical hyper-rectangles from PRIM         | Heatmap box overlays      |
| **📉 PRIM Trajectory**      | Coverage–Density trade-offs                 | PRIM trajectory plots     |
| **👥 Demographic Profiles** | Aggregated traits of high-adoption segments | Analytical tables         |

---

## 2️⃣ Data Validation

Validate generated datasets against schemas in `src/data/schemas.py`:

```bash
python src/scripts/validate_dummy_data.py
```

---

## ⚙️ Configuration

All generation parameters—agents, Trust/Income ranges, grid resolution, noise levels, replications—are defined in:

```
config/dummy_data.yaml
```

Modify this file to fully customize your synthetic data pipeline.

---

## 📚 References

* See **Dummy Data Configuration** in the Architecture Guide.
* For initial setup steps, refer to **Getting Started**.