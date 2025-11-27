# 🔋 ECOGrid - Energy Community Optimization & Grid Analysis

Agent-Based simulation using MESA to discover optimal incentive policies for Energy Communities through Scenario Discovery techniques.

## 🎯 What Does This Project Do?

ECOGrid simulates how people decide to join energy communities under different incentive programs. It helps answer critical research questions:

  - 💡 Which incentives work best to increase adoption?
  - 👥 What types of people are most likely to join?
  - 💰 How do trust and income affect decision-making?
  - 📊 What policies maximize community participation?

-----

## 🧩 Key Features

  - ⚡ **Agent-Based Modeling** with the MESA framework.
  - 🎲 **Scenario Discovery** using the PRIM algorithm.
  - 📈 **3 Policy Scenarios**: No Incentive (NI), Services Incentive (SI), Economic Incentive (EI).
  - 🗺️ **Visual Reports**: Heatmaps, trajectory plots, and demographic tables for analysis.
  - 🔧 **Comprehensive Dummy Data System** for reproducible testing and development.

-----

## 🔗 Documentation Index (The ECOGrid Launchpad)

This project is large, and the detailed guides are now located in the `docs/` folder to improve navigation. Start here to find what you need:

| Topic | Focus | File Link |
| :--- | :--- | :--- |
| **🚀 Getting Started** | **Installation, setup, and first run commands.** | [🎓 `getting_started.md`](./docs/getting_started.md) |
| **🏗️ Architecture** | **Design principles (SOLID/DRY) and system structure.** | [🏗️ `architecture.md`](./docs/architecture.md) |
| **📊 Reports & Viz** | **Detailed descriptions of all generated reports (Heatmaps, PRIM Trajectory, Tables).** | [🗺️ `visualization_scripts.md`](./docs/visualization_scripts.md) |
| **🧪 API Reference** | **Detailed function and class documentation, including testing instructions.** | [🔍 `api_reference.md`](./docs/api_reference.md) |
| **📦 Data Pipeline** | **Guide to generating and managing dummy data.** | [🎲 `data_generation.md`](./docs/data_generation.md) |
| **⚙️ Tutorials** | **Step-by-step guides for specific usage scenarios.** | [📖 `tutorial.md`](./docs/tutorial.md) |

-----

## 📁 Project Structure (High Level)

```
ECOGrid/
├── src/                        # 🐍 All core Python code (Simulation, Analysis, Incentives)
├── tests/                      # ✅ Unit and integration tests
├── data/                       # 💾 Input/output data storage (raw, processed, results)
├── config/                     # ⚙️ YAML configuration files (base, scenarios, dummy data)
├── docs/                       # 📚 Detailed documentation files (see table above)
├── notebooks/                  # 📓 Jupyter analysis and validation notebooks
└── README.md                   # 📖 This file!
```

-----

## 🛠️ Built With

This project relies on the following key tools and frameworks:

  - **MESA** - Agent-based modeling framework
  - **Python 3.9+** - Programming language
  - **NumPy & Pandas** - Data processing
  - **Matplotlib & Seaborn** - Chart generation
  - **PyYAML** - Configuration management
  - **Pytest** - Testing framework

-----

## 📄 License & Contact

This research project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.

**Policy Summary:**
* You are free to **share** (copy and redistribute) the material in any medium or format.
* The material **cannot be used for commercial purposes**.
* You may **not distribute modified material**.

For any permissions regarding **commercial use**, **modification**, or general inquiries, please contact the project owner:

**Authors:**
G. Antonio Pierro

**Contact:**
antonio.pierro@gmail.com

-----

⭐ If you find this project useful, please star it on GitHub\!