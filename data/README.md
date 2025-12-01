# Data Directory

All simulation data outputs. **Data files are not tracked by Git.**

## Structure
```
data/
├── abm/              # Agent-Based Model (Mesa) simulations
├── montecarlo/       # Monte Carlo simulations  
└── dummy/            # Test/development data
```

Each simulation type contains only `raw/` directory with direct outputs from generation scripts.

## Generating Data
```bash
# Generate ABM data
python src/scripts/generate_abm.py

# Generate Monte Carlo data  
python src/scripts/generate_montecarlo.py

# Generate dummy data
python src/scripts/generate_dummy.py
```

## Visualization

Data in `raw/` folders are ready for matplotlib visualization tools in `src/visualization/`.

## .gitignore

All data files (`.csv`, `.json`, `.png`) are ignored. Only directory structure is tracked.
