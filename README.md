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
- 🔧 **Dummy Data Generators** for quick testing without real data

## 📁 Project Structure

```
ECOGrid/
├── src/                        # 🐍 All Python code
│   ├── simulation/            # 🤖 MESA agents and model
│   ├── scenarios/             # 🎯 Scenario generation and PRIM
│   ├── incentives/            # 💵 Incentive policy logic
│   ├── data/                  # 📦 Data generators and loaders
│   ├── analysis/              # 📊 Metrics and statistics
│   ├── visualization/         # 📉 Charts and plots
│   ├── utils/                 # 🛠️ Helper functions
│   └── experiments/           # 🧪 Simulation run scripts
├── tests/                     # ✅ Unit and integration tests
├── data/                      # 💾 Input/output data storage
├── config/                    # ⚙️ YAML configuration files
├── docs/                      # 📚 Documentation
├── notebooks/                 # 📓 Jupyter notebooks
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

Start by creating synthetic test data:

```bash
python src/data/generators/demographic_generator.py
python src/data/generators/consumption_generator.py
python src/data/generators/production_generator.py
```

This creates fake but realistic:
- 👤 Agent demographics (age, income, trust levels)
- ⚡ Energy consumption profiles
- ☀️ Solar production patterns

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

ECOGrid generates three main outputs:

### 1. 🗺️ Adoption Rate Heatmaps
Visual maps showing which combinations of trust and income lead to high adoption. Includes:
- Three separate plots for NI, SI, and EI scenarios
- Yellow boxes highlighting "sweet spots" found by PRIM
- Color gradient from low (purple) to high (yellow) adoption

### 2. 📈 PRIM Peeling Trajectory
Graph showing the trade-off between:
- **Coverage**: What % of people are in the group?
- **Density**: What % of the group actually adopted?

Helps identify the most efficient targeting strategy.

### 3. 📋 Demographic Profile Table
Table breaking down who joins in each scenario:
- Age, income, trust level ranges
- How many people (coverage %)
- Adoption rate (density %)
- Improvement over baseline (lift)

All outputs saved in: `data/results/`

## ⚙️ Configuration

Edit `config/base.yaml` to customize:

```yaml
simulation:
  n_agents: 10000        # Number of people to simulate
  n_steps: 365           # Simulation days
  
incentives:
  economic_rate: 0.20    # Economic incentive amount
  service_value: 100     # Service incentive value
  
prim:
  alpha: 0.05           # Peeling rate
  threshold: 0.75       # Minimum density
```

## 🧪 Testing

Run tests to make sure everything works:

```bash
# All tests
pytest tests/

# With coverage report
pytest --cov=src tests/

# Specific test file
pytest tests/test_agents.py -v
```

## 📚 Learn More

Check out the `docs/` folder:
- 🏗️ `architecture.md` - How the system is built
- 🎓 `getting_started.md` - Beginner tutorial
- 📖 `tutorial.md` - Step-by-step examples
- 🔍 `api_reference.md` - Function documentation

## 🛠️ Built With

- **MESA** - Agent-based modeling framework
- **Python 3.9+** - Programming language
- **NumPy & Pandas** - Data processing
- **Matplotlib & Seaborn** - Beautiful charts
- **PyYAML** - Configuration management
- **Pytest** - Testing framework

## 🎨 Design Principles

- ✅ **SOLID**: Each module does one thing well
- ✅ **DRY**: Don't repeat yourself - reuse code
- ✅ **Modular**: Easy to add new features
- ✅ **Testable**: Every part can be tested separately

## 📝 Roadmap

- [ ] 🤖 Implement base MESA agents
- [ ] 🎲 Create working dummy data generators
- [ ] 📊 Build heatmap and PRIM visualizations
- [ ] 🔬 Implement PRIM algorithm
- [ ] ✅ Complete test suite
- [ ] 📚 Detailed API documentation
- [ ] 🌐 Optional: Interactive web dashboard

## 🤝 Contributing

Want to help improve ECOGrid?

1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/cool-feature`
3. 💾 Commit your changes: `git commit -m 'Add cool feature'`
4. 📤 Push to branch: `git push origin feature/cool-feature`
5. 🎉 Open a Pull Request

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