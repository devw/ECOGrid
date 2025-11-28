# 🔋 ECOGrid - Energy Community Optimization & Grid Analysis

ECOGrid simulates Energy Communities using **two complementary approaches**:

1. **Agent-Based Modeling (ABM)** via MESA to explore individual decisions and scenario discovery.
2. **MonteCarlo Simulation** for large-scale stochastic data generation and reproducible experimentation.

---

## 🎯 What Does This Project Do?

ECOGrid helps answer critical research questions:

  - 💡 Which incentives work best to increase adoption?
  - 👥 What types of people are most likely to join?
  - 💰 How do trust and income affect decision-making?
  - 📊 What policies maximize community participation?
  - 🎲 How can stochastic simulations support scenario analysis?

---

## 🧩 Key Features

  - ⚡ **Agent-Based Modeling** with MESA.
  - 🎲 **MonteCarlo Data Generation** for robust synthetic datasets.
  - 🗺️ **Scenario Discovery** using PRIM.
  - 📈 **3 Policy Scenarios**: No Incentive (NI), Services Incentive (SI), Economic Incentive (EI).
  - 🔧 **Reproducible Data Pipelines**: MonteCarlo output and validation notebooks.
  - 🗂️ **Visual Reports**: Heatmaps, PRIM trajectory plots, and demographic tables.

---

## 🔗 Documentation Index (The ECOGrid Launchpad)

| Topic | Focus | File Link |
| :--- | :--- | :--- |
| **🚀 Getting Started** | Installation, setup, and first run commands | [🎓 `getting_started.md`](./docs/getting_started.md) |
| **🏗️ Architecture** | Design principles (SOLID/DRY) and system structure | [🏗️ `architecture.md`](./docs/architecture.md) |
| **📊 Reports & Viz** | Detailed descriptions of all generated reports (Heatmaps, PRIM Trajectory, Tables) | [🗺️ `visualization_scripts.md`](./docs/visualization_scripts.md) |
| **🧪 API Reference** | Function and class documentation | [🔍 `api_reference.md`](./docs/api_reference.md) |
| **📦 MonteCarlo Pipeline** | Guide to generating and validating MonteCarlo datasets | [🎲 `data_generation_montecarlo.md`](./docs/data_generation_montecarlo.md) |
| **⚙️ Tutorials** | Step-by-step guides for specific usage scenarios | [📖 `tutorial.md`](./docs/tutorial.md) |

---

## 📁 Project Structure (High Level)

```

ECOGrid/
├── src/                        # 🐍 Core Python code (Simulation, Analysis, Incentives)
├── tests/                      # ✅ Unit and integration tests
├── data/                       # 💾 Input/output storage (raw, processed, MonteCarlo results)
├── config/                     # ⚙️ YAML configuration files (base, scenarios, MonteCarlo)
├── docs/                       # 📚 Documentation files (see table above)
├── notebooks/                  # 📓 Jupyter analysis and validation notebooks
└── README.md                   # 📖 This file

```

---

## 🛠️ Built With

  - **MESA** - Agent-based modeling framework
  - **Python 3.9+** - Programming language
  - **NumPy & Pandas** - Data processing
  - **Matplotlib & Seaborn** - Chart generation
  - **PyYAML** - Configuration management
  - **Pytest** - Testing framework

---

## 📄 License & Contact

This research project is licensed under **CC BY-NC-ND 4.0**.

**Authors:** G. Antonio Pierro  
**Contact:** antonio.pierro@gmail.com
```