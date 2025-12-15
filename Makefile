# Makefile per ECOGrid Project
# Alias brevissimi: abm, mc, csv, hm

# Percorsi e variabili
ABM_SCRIPT=python -m src.scripts.generate_abm
MC_SCRIPT=python -m src.scripts.generate_montecarlo
CSV_ANALYSIS=python -m src.analysis.csv_validation_analysis
HEATMAP=python -m src.visualization.adoption_heatmap_generator

ABM_OUTPUT=data/abm_experiments
MC_OUTPUT=data/montecarlo_calibrated_fixed
HEATMAP_OUT=/tmp/adoption_montecarlo.png
CONFIG=config/high_trust.yaml
SEED=1234

# -----------------------------
#  Data Generation
# -----------------------------
abm:
	$(ABM_SCRIPT) \
		--n-consumers 20 \
		--n-prosumers 10 \
		--n-grid-agents 2 \
		--n-steps 10 \
		--seed 5678 \
		--scenario high_trust_policy \
		--config $(CONFIG) \
		--output $(ABM_OUTPUT)

mc:
	$(MC_SCRIPT) \
		--n-agents 5000 \
		--n-replications 100 \
		--n-bins 15 \
		--noise-std 0.03 \
		--seed $(SEED) \
		--output $(MC_OUTPUT)

# -----------------------------
#  Analysis
# -----------------------------
csv:
	$(CSV_ANALYSIS) -d $(MC_OUTPUT)

# -----------------------------
#  Visualization
# -----------------------------
hm:
	$(HEATMAP) --data-dir $(MC_OUTPUT) --output $(HEATMAP_OUT)

# -----------------------------
#  All-in-one
# -----------------------------
all: mc csv

.PHONY: abm mc csv hm all
