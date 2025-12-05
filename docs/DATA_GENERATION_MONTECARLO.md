# 📦 MonteCarlo Data Generation Guide

ECOGrid uses a robust MonteCarlo-based synthetic data pipeline to ensure reproducibility, efficient development, and reliable simulation testing. This guide explains why MonteCarlo data is important, how to generate it, and how to validate it.

---

## 🎯 Purpose of MonteCarlo Data

MonteCarlo-generated data is essential for:
- **Testing & Development** — Provides consistent and statistically meaningful scenario inputs.
- **Reproducibility** — Ensures anyone can reproduce baseline results with the same random seed.
- **Scenario Exploration** — Enables experimentation across multiple stochastic replications.

All generated data is stored in `data/montecarlo/`.

---

## 1️⃣ Data Generation

Data generation is handled by scripts in `src/scripts/` and logic in `src/data/montecarlo_generators/`.

### A. Full Generation

Generate all MonteCarlo datasets with default parameters:

```bash
python -m src.scripts.generate_montecarlo
````

You can also pass parameters to customize the data generation process.
For example, to set the number of replications:

```bash
python -m src.scripts.generate_montecarlo --n-replications 300
```

---

## 🔧 Available CLI Parameters

| Parameter          | Type  | Default           | Description                                                                         |
| ------------------ | ----- | ----------------- | ----------------------------------------------------------------------------------- |
| `--n-agents`       | int   | `10000`           | Number of agents generated per scenario.                                            |
| `--n-bins`         | int   | `20`              | Number of bins per axis (e.g., 20×20 grid) for heatmap generation.                  |
| `--noise-std`      | float | `0.05`            | Standard deviation of random noise added to adoption rates to simulate uncertainty. |
| `--seed`           | int   | `42`              | Random seed for ensuring reproducibility.                                           |
| `--output`         | Path  | `data/montecarlo` | Output directory for generated CSV/JSON files.                                      |
| `--n-replications` | int   | varies by config  | Number of MonteCarlo replications used for adoption rate sampling.                  |

---

### Example: Full Custom Generation

```bash
python -m src.scripts.generate_montecarlo \
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
python -m src.scripts.validate_montecarlo_data
```

---

## ⚙️ Configuration

All generation parameters—agents, Trust/Income ranges, grid resolution, noise levels, replications—are defined in:

```
config/montecarlo_data.yaml
```

Modify this file to fully customize your MonteCarlo data pipeline.

---

## 📚 References

* See **MonteCarlo Data Configuration** in the Architecture Guide.
* For initial setup steps, refer to **Getting Started**.