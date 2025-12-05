# 🚀 Getting Started with ECOGrid

This guide explains how to clone the repository, set up your development environment, generate synthetic test data, and run your first simulation.

## 🔧 Prerequisites

Before starting, ensure you have:
- Git
- Python 3.13+

## 1️⃣ Installation and Environment Setup

### A. Clone the Repository
```bash
git clone <repo-url>
cd ECOGrid
```

### B. Create and Activate Virtual Environment
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
# venv\Scripts\activate
```

### C. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## 2️⃣ Generate Dummy Data

Data is generated based on `config/dummy_data.yaml` and saved in `data/dummy/`.

### A. Full Data Generation
```bash
python src/scripts/generate_dummy_data.py
```

### B. Validate Data (Optional)
```bash
python src/scripts/validate_dummy_data.py
```

## 3️⃣ Run Your First Simulation

### A. Baseline Simulation
```bash
python src/experiments/run_baseline.py
```

### B. Scenario Discovery (All 3 Policies + PRIM)
```bash
python src/experiments/run_scenarios.py
```

## 📚 Next Steps
- Architecture Guide  
- Reports & Visualization Guide  
- Configuration Guide
