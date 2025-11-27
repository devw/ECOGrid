# 📦 Data Generation Guide

ECOGrid relies on a robust data pipeline built on synthetic (dummy) data to ensure reproducibility, development efficiency, and reliable simulation testing.

This document explains why synthetic data is important, how to generate it, and how to validate it.

---

## 🎯 Purpose of Synthetic Data

Dummy data is essential for:

- **Testing & Development** — Provides consistent and predictable inputs for unit tests and integration.
- **Reproducibility** — Ensures anyone can run the simulations and obtain the same baseline results.
- **Working Environment Setup** — Allows full experimentation before introducing real datasets.

All generated data is saved in:

```
data/dummy/
```

---

## 1️⃣ Data Generation

Data generation is handled by scripts located in  
`src/scripts/` and `src/data/generators/`.

### A. Full Generation

Run the main script to generate all required datasets at once:

```bash
python src/scripts/generate_dummy_data.py
```

---

## B. Generated Datasets (All 6)

The full pipeline produces six specialized datasets:

| Dataset | Description | Usage |
|--------|-------------|--------|
| **👤 Agents** | Demographic attributes including Trust and Income | Input for ABM initialization |
| **📈 Adoption Rates** | Scenario-specific adoption rates (NI/SI/EI) | Benchmarks and PRIM analysis |
| **🗺️ Heatmap Grid** | Sampled Trust–Income combinations | Heatmap creation (Figure 1) |
| **📦 PRIM Boxes** | Critical hyper-rectangles identified by PRIM | Box contours on heatmaps |
| **📉 PRIM Trajectory** | Coverage–Density trade-off during peeling | PRIM trajectory plot (Figure 2) |
| **👥 Demographic Profiles** | Aggregated traits of high-adoption groups | Analytical tables (Table III) |

---

## 2️⃣ Data Validation

After generating the datasets, you can validate them against predefined schemas located in `src/data/schemas.py`.

```bash
python src/scripts/validate_dummy_data.py
```

---

## ⚙️ Configuration

All parameters for data generation—number of agents, Trust/Income ranges, grid resolution—are defined in:

```
config/dummy_data.yaml
```

You can modify this file to customize your synthetic data.

---

## 📚 References

- For configuration details, see **Dummy Data Configuration** in the Architecture Guide.
- For initial setup instructions, see **Getting Started**.
