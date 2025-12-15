# 📝 ECOGrid Scripts Guide

Executable scripts in the ECOGrid project. Run from project root as Python modules.

---

## 1️⃣ Data Generation

### **ABM Data**

```bash
# Default
python -m src.scripts.generate_abm --n-consumers 10 --n-prosumers 5 --n-grid-agents 1 --n-steps 5 --seed 1234 --output data/abm_custom

# Custom scenario
python -m src.scripts.generate_abm --n-consumers 20 --n-prosumers 10 --n-grid-agents 2 --n-steps 10 --seed 5678 --scenario high_trust_policy --config config/high_trust.yaml --output data/abm_experiments
````

### **Monte Carlo Data**

```bash
python -m src.scripts.generate_montecarlo

# Custom
python -m src.scripts.generate_montecarlo --n-agents 5000 --n-replications 100 --n-bins 15 --noise-std 0.03 --seed 1234 --output data/montecarlo_calibrated_fixed
```

### **Dummy Data**

```bash
python -m src.scripts.generate_dummy
```

---

## 2️⃣ Experiments

```bash
python -m src.experiments.run_baseline
python -m src.experiments.run_scenarios
python -m src.experiments.run_sensitivity
```

---

## 3️⃣ Analysis

### **CSV Validation & Summary**

```bash
python src/analysis/csv_validation_analysis.py -d data/montecarlo_calibrated_fixed
```

Sample output:

```
─────────────────────────────
📋 SECTION 5: SUMMARY TABLE
─────────────────────────────
Scenario             Avg Align    Target OK    LIFT       Overall
───────────────────────────────────────────────────────────────
No Incentive         ❌            ✅            ✅ 1.69x   ✅ Good
Services Incentive   ✅            ✅            ⭐ 2.68x   ✅ Good
Economic Incentive   ❌            ✅            ⚠️ 1.30x   ⚠️ Partial
```

---

## 4️⃣ Visualization

```bash
# Run as Python modules from root
python -m src.visualization.<script_name>
```

Examples:

```bash
python -m src.visualization.adoption_heatmap_generator --data-dir data/montecarlo_calibrated_fixed --output /tmp/adoption_montecarlo.png
python -m src.visualization.prim_trajectory --data-dir data/montecarlo_calibrated_fixed --output /tmp/prim_trajectory.png
python -m src.visualization.demographic_table --data-dir data/montecarlo --output /tmp/demographic_profiles.md
```

---

## 💾 Output

* Default outputs saved to `/tmp/` or folder specified with `--output`.
* ABM and Monte Carlo outputs can feed visualization scripts.

