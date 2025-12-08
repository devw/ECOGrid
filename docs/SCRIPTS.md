# 📝 SCRIPTS GUIDE

This document lists all executable scripts in the ECOGrid project.
Commands include basic execution and example parameters.

---

## 1️⃣ Data Generation Scripts

### **Generate ABM Data**

```bash
# Using defaults
python -m src.scripts.generate_abm \
    --n-consumers 10 \
    --n-prosumers 5 \
    --n-grid-agents 1 \
    --n-steps 5 \
    --seed 1234 \
    --output data/abm_custom
```

Example with custom parameters:

```bash
# With custom scenario and config
python -m src.scripts.generate_abm \
    --n-consumers 20 \
    --n-prosumers 10 \
    --n-grid-agents 2 \
    --n-steps 10 \
    --seed 5678 \
    --scenario high_trust_policy \
    --config config/high_trust.yaml \
    --output data/abm_experiments
```

### **Generate Monte Carlo Data**

```bash
python -m src.scripts.generate_montecarlo
```

Example with custom parameters:

```bash
python -m src.scripts.generate_montecarlo \
    --n-agents 5000 \
    --n-replications 300 \
    --n-bins 15 \
    --noise-std 0.03 \
    --seed 1234 \
    --output data/custom_low_noise
```

### **Generate Dummy Data**

```bash
python -m src.scripts.generate_dummy
```

---

## 2️⃣ Experiment Scripts

### **Run Baseline Experiment**

```bash
python -m src.experiments.run_baseline
```

### **Run Scenarios Experiment**

```bash
python -m src.experiments.run_scenarios
```

### **Run Sensitivity Analysis**

```bash
python -m src.experiments.run_sensitivity
```

---

## 3️⃣ Visualization Scripts

> All visualization scripts must be run as **Python modules from project root**:

```bash
python -m src.visualization.<script_name>
```

### **Adoption Heatmap**

```bash
python -m src.visualization.adoption_heatmap_generator --data-dir data/montecarlo --output /tmp/adoption_montecarlo.png
```

### **PRIM Trajectory**

```bash
python -m src.visualization.prim_trajectory --data-dir data/montecarlo --output /tmp/prim_trajectory.png
```

### **Demographic Table**

```bash
python -m src.visualization.demographic_table --data-dir data/montecarlo --output /tmp/demographic_profiles.md
```

---

## 💾 Output

* Default outputs are saved to `/tmp/` or the folder specified with `--output`.
* ABM and Monte Carlo simulation outputs can be used as input for visualization scripts.
