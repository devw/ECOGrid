# 🔋 ECOGrid - Energy Community Optimization & Grid Analysis

Agent-Based simulation using MESA to discover optimal incentive policies for Energy Communities through Scenario Discovery techniques.

## 🎯 What Does This Project Do?

ECOGrid simulates how people decide to join energy communities under different incentive programs. It helps answer questions like:

- 💡 Which incentives work best to increase adoption?
- 👥 What types of people are most likely to join?
- 💰 How do trust and income affect decision-making?
- 📊 What policies maximize community participation?

## 🧩 Key Features

- ⚡ **Agent-Based Modeling** with MESA framework
- 🎲 **Scenario Discovery** using PRIM algorithm
- 📈 **3 Policy Scenarios**: No Incentive (NI), Services Incentive (SI), Economic Incentive (EI)
- 🗺️ **Visual Reports**: Heatmaps, trajectory plots, demographic tables
- 🔧 **Comprehensive Dummy Data System** for testing and development

## 📁 Project Structure

```
ECOGrid/
├── src/                        # 🐍 All Python code
│   ├── simulation/            # 🤖 MESA agents and model
│   ├── scenarios/             # 🎯 Scenario generation and sampling
│   ├── incentives/            # 💵 Incentive policy logic + shared utils
│   ├── data/                  # 📦 Data generators, processors, schemas
│   │   ├── generators/       # ✨ Specialized dummy data generators
│   │   └── processors/       # 🔄 Data loading, validation, aggregation
│   ├── analysis/              # 📊 Metrics, PRIM analysis, sensitivity
│   ├── visualization/         # 📉 Charts, heatmaps, trajectories
│   ├── utils/                 # 🛠️ Shared utilities (PRIM, stats, config)
│   ├── experiments/           # 🧪 Simulation run scripts
│   └── scripts/               # 🔨 Data generation and validation scripts
├── tests/                     # ✅ Unit and integration tests
├── data/                      # 💾 Input/output data storage
│   ├── dummy/                # 🎲 Generated dummy data (CSVs)
│   ├── processed/            # 🔄 Processed datasets
│   ├── raw/                  # 📥 Raw input data
│   └── results/              # 📊 Simulation outputs
├── config/                    # ⚙️ YAML configuration files
│   ├── base.yaml             # 🔧 Base configuration
│   ├── dummy_data.yaml       # 🎲 Dummy data generation config
│   └── scenarios/            # 📋 Scenario-specific configs
├── docs/                      # 📚 Documentation
├── notebooks/                 # 📓 Jupyter notebooks
│   └── dummy_data_validation.ipynb  # ✅ Data validation notebook
└── README.md                  # 📖 This file!
```

## 🚀 Getting Started

### 1️⃣ Installation

```bash
# Clone the repository
git clone <repo-url>
cd ECOGrid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2️⃣ Generate Dummy Data

Start by creating comprehensive synthetic test data:

```bash
# Generate all dummy data at once
python src/scripts/generate_dummy_data.py

# Or generate specific datasets individually:
python -m src.data.generators.agent_generator
python -m src.data.generators.adoption_generator
python -m src.data.generators.heatmap_generator
python -m src.data.generators.prim_box_generator
python -m src.data.generators.prim_trajectory_generator
python -m src.data.generators.demographic_profile_generator
```

This creates realistic dummy data for:
- 👤 **Agents**: Demographics with Trust and Income attributes
- 📈 **Adoption Rates**: Scenario-specific adoption patterns (NI/SI/EI)
- 🗺️ **Heatmap Grid**: Trust-Income combinations for visualization
- 📦 **PRIM Boxes**: Critical parameter subspaces identified by PRIM
- 📉 **PRIM Trajectory**: Coverage-Density trade-off data
- 👥 **Demographic Profiles**: High-adoption segment characteristics

**Validate generated data:**

```bash
python src/scripts/validate_dummy_data.py
```

### 3️⃣ Run Your First Simulation

```bash
python src/experiments/run_baseline.py
```

### 4️⃣ Run Scenario Discovery

```bash
python src/experiments/run_scenarios.py
```

This will test all three policy scenarios and generate reports! 📊

## 📊 What Reports You Get

ECOGrid generates three main outputs based on the paper:

### 1. 🗺️ Figure 1: Adoption Rate Heatmaps
Visual maps showing which combinations of trust and income lead to high adoption:
- Three separate heatmaps for NI, SI, and EI scenarios
- Yellow PRIM box boundaries highlighting optimal parameter subspaces
- Color gradient from low (dark purple) to high (bright yellow) adoption
- Shows how incentives shift adoption toward high-trust agents

**Generated from:** `data/dummy/heatmap_grid.csv` + `data/dummy/prim_boxes.csv`

### 2. 📈 Figure 2: PRIM Peeling Trajectory
Graph showing the Coverage-Density trade-off during iterative peeling:
- **Coverage**: Proportion of population in each subgroup (%)
- **Density**: Adoption rate within subgroup (%)
- Stars indicate final selected boxes
- Diagonal dashed line represents random targeting baseline
- SI shows dramatic peeling (6% coverage, 81% density)
- EI shows moderate peeling (31% coverage, 65% density)
- NI remains flat (uniform baseline, no high-density segments)

**Generated from:** `data/dummy/prim_trajectory.csv`

### 3. 📋 Table III: Demographic Profile Analysis
Table breaking down high-adoption segments per scenario:
- Parameter ranges (Trust, Income, etc.)
- Coverage: % of population in subgroup
- Density: Adoption rate within subgroup
- Lift: Ratio of subgroup density to scenario baseline
- Based on 10,000 agents per scenario from 100 simulation runs

**Generated from:** `data/dummy/demographic_profiles.csv`

All outputs saved in: `data/results/`

## ⚙️ Configuration

### Base Configuration

Edit `config/base.yaml` to customize simulation parameters:

```yaml
simulation:
  n_agents: 10000        # Number of people to simulate
  n_steps: 365           # Simulation days
  
incentives:
  economic_rate: 0.20    # Economic incentive amount (20%)
  service_value: 100     # Service incentive value
  
prim:
  alpha: 0.05           # Peeling rate (5%)
  threshold: 0.75       # Minimum density threshold
  min_support: 0.05     # Minimum coverage for boxes
```

### Dummy Data Configuration

Edit `config/dummy_data.yaml` to customize data generation:

```yaml
agents:
  n_agents: 10000
  trust_range: [0, 1]
  income_range: [0, 100]
  
scenarios:
  - NI  # No Incentive
  - SI  # Services Incentive
  - EI  # Economic Incentive
  
prim:
  n_runs: 100           # Number of simulation runs
  grid_resolution: 50   # Heatmap grid size
```

## 🧪 Testing

Run tests to make sure everything works:

```bash
# All tests
pytest tests/

# With coverage report
pytest --cov=src tests/

# Specific test modules
pytest tests/test_agents.py -v
pytest tests/test_data_generators.py -v
pytest tests/test_prim_analysis.py -v
```

## 📚 Learn More

Check out the `docs/` folder:
- 🏗️ `architecture.md` - How the system is built
- 🎓 `getting_started.md` - Beginner tutorial
- 📖 `tutorial.md` - Step-by-step examples
- 🔍 `api_reference.md` - Function documentation
- 🎲 `data_generation.md` - Dummy data generation guide

## 🛠️ Built With

- **MESA** - Agent-based modeling framework
- **Python 3.9+** - Programming language
- **NumPy & Pandas** - Data processing
- **Matplotlib & Seaborn** - Beautiful charts
- **PyYAML** - Configuration management
- **Pytest** - Testing framework

## 🎨 Design Principles

- ✅ **SOLID**: Each module has a single, well-defined responsibility
- ✅ **DRY**: Shared utilities centralized in `utils/` and `incentives/incentive_utils.py`
- ✅ **Modular**: Easy to add new generators, scenarios, and visualizations
- ✅ **Testable**: Every component can be tested in isolation
- ✅ **Separation of Concerns**: Generators → Processors → Analysis → Visualization

## 📝 Roadmap

- [x] ♻️ Restructure project for comprehensive dummy data generation
- [x] ✨ Implement specialized data generators (6 types)
- [x] 🔧 Create shared utilities (PRIM, stats, incentives)
- [ ] 🤖 Implement complete MESA agent classes
- [ ] 🎲 Implement full PRIM algorithm
- [ ] 📊 Complete visualization pipeline
- [ ] 🔬 Add sensitivity analysis
- [ ] ✅ Expand test coverage to 90%+
- [ ] 📚 Complete API documentation
- [ ] 🌐 Optional: Interactive web dashboard

## 🤝 Contributing

Want to help improve ECOGrid?

1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/cool-feature`
3. 💾 Commit your changes: `git commit -m '✨ add cool feature'`
4. 📤 Push to branch: `git push origin feature/cool-feature`
5. 🎉 Open a Pull Request

**Commit Message Convention:** We use emoji prefixes! See examples:
- ✨ `:sparkles:` - New features
- 🐛 `:bug:` - Bug fixes
- ♻️ `:recycle:` - Refactoring
- 📝 `:memo:` - Documentation
- ✅ `:white_check_mark:` - Tests

## 📄 License

[Specify your license here]

## 👥 Authors

[Your name(s) here]

## 📧 Contact

[Email or repository link]

---

⭐ If you find this project useful, please star it on GitHub!

🐛 Found a bug? Open an issue!

💡 Have an idea? We'd love to hear it!