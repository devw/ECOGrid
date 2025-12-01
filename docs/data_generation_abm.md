# 📦 ABM Energy Community Data Generation Guide

ECOGrid now supports generating synthetic data using **Agent-Based Modeling (ABM)** to simulate energy community dynamics. This guide explains why ABM-generated data is important, how to generate it, and how to structure the outputs for downstream analysis.

---

## 🎯 Purpose of ABM Data

ABM-generated data is essential for:

* **Testing & Development** — Provides realistic agent-level trajectories.
* **Scenario Analysis** — Enables experimentation with different incentive policies and agent compositions.
* **Reproducibility** — Ensures anyone can reproduce the same trajectories using a fixed random seed.

All generated data is stored in `data/abm/`.

---

## 1️⃣ Data Generation

Data generation is handled by:

* **Script:** `src/scripts/generate_abm_energy_community.py`
* **Logic:** `src/simulation/model.py` and `src/simulation/agents/`

### A. Run a Simulation

Generate ABM data with default parameters:

```bash
python -m src.scripts.generate_abm_energy_community
```

---

### B. Customize Parameters

You can adjust the number of agents, steps, seed, and output folder. Example:

```bash
python -m src.scripts.generate_abm_energy_community \
    --n-consumers 10 \
    --n-prosumers 5 \
    --n-grids 1 \
    --n-steps 50 \
    --seed 1234 \
    --output data/abm_custom
```

---

### 🔧 Available CLI Parameters

| Parameter       | Type | Default    | Description                        |
| --------------- | ---- | ---------- | ---------------------------------- |
| `--n-consumers` | int  | `2`        | Number of Consumer agents          |
| `--n-prosumers` | int  | `2`        | Number of Prosumer agents          |
| `--n-grids`     | int  | `1`        | Number of Grid agents              |
| `--n-steps`     | int  | `3`        | Number of simulation steps         |
| `--seed`        | int  | `42`       | Random seed for reproducibility    |
| `--output`      | Path | `data/abm` | Directory to save CSV/JSON outputs |

---

### 2️⃣ Output Structure

Each run creates a folder named after the seed:

```
data/abm/
└── run_seed_42/
    ├── agents_step_1.csv
    ├── agents_step_2.csv
    ├── ...
    └── simulation_output.json
```

* `agents_step_*.csv` → Agent states at each simulation step.
* `simulation_output.json` → Full aggregated simulation trajectory for easy analysis.

---

### 3️⃣ Notes

* Agents are currently **mocked**, meaning behavior logic is minimal.
* Future versions will implement realistic decision rules and interactions.
* CSVs are compatible with downstream notebooks and visualizations.

---

### 📚 References

* See **MonteCarlo Data Generation Guide** for comparisons and data pipeline conventions.
* For initial setup steps, refer to **Getting Started**.

