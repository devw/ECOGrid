# 🏗️ ECOGrid Architecture and Design Principles

This document provides a high-level overview of the project structure, the core design principles behind ECOGrid, and the available configuration options.

## 📁 Project Structure Overview

ECOGrid is built on the **MESA Agent-Based Modeling (ABM)** framework and includes data processing, scenario analysis, and visualization in a modular architecture.

Main components (located in `src/`):

```
src/simulation/   → MESA agents and core model logic
src/scenarios/    → Scenario generation and parameter sampling
src/incentives/   → Incentive policies (SI, EI, NI)
src/analysis/     → Metrics, PRIM algorithm, sensitivity analysis
src/data/         → Data generation, processing, validation
```

---

## 🎨 Design Principles

ECOGrid follows modern software engineering principles to ensure maintainability and scalability:

- **SOLID** — each module has one clear responsibility  
- **DRY** — shared logic is centralized (`src/utils/`, `incentive_utils.py`)  
- **Modular** — new generators, scenarios, incentives, and visualizations can be added easily  
- **Testable** — all components can be tested in isolation  
- **Clear workflow pipeline**  
  *Data Generation → Processing → Simulation/Experiments → Analysis → Visualization*

---

## ⚙️ Configuration Management

Runtime parameters, model settings, and experiment controls are managed through YAML files in the `config/` directory.

### 🧪 Dummy Data Configuration (`config/dummy_data.yaml`)

```yaml
agents:
  n_agents: 10000
  trust_range: [0, 1]
  income_range: [0, 100]

scenarios:
  - NI   # No Incentive
  - SI   # Services Incentive
  - EI   # Economic Incentive

prim:
  n_runs: 100
  grid_resolution: 50
```

---

## 📝 Roadmap

Planned development for ECOGrid:

- [x] Project restructuring for complete dummy data generation  
- [x] Implementation of specialized data generators  
- [x] Shared utilities (PRIM, statistics, incentives)  
- [ ] Full implementation of MESA agent classes  
- [ ] Complete PRIM algorithm implementation  
- [ ] Visualization pipeline  
- [ ] Sensitivity analysis  
- [ ] Test coverage expanded to 90%+  
- [ ] Full API documentation  
- [ ] Optional: interactive web dashboard  

---

For instructions on setting up the project and running simulations, see **Getting Started**.
