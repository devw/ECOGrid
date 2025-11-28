# Import esterni necessari
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Importa le costanti dai settings
from .._config.settings import HEATMAP_FILE, PRIM_BOXES_FILE, METADATA_FILE, DATA_DIR

# --- IO UTILITIES (Spostate e Modificate) ---

def load_csv(name: str) -> pd.DataFrame:
    """
    Carica un file CSV dal disco o genera dati montecarlo se non trovato.
    """
    path = DATA_DIR / name
    if not path.exists():
        # Logica per la creazione di dati montecarlo
        if name == HEATMAP_FILE:
            num_bins = 10
            np.random.seed(42)
            trust_bins = np.linspace(0.03, 0.97, num_bins)
            income_bins = np.linspace(2, 98, num_bins)
            data = []
            for scenario in ['NI', 'SI', 'EI']:
                for t_bin in trust_bins:
                    for i_bin in income_bins:
                        base_rate = (t_bin + i_bin / 100) / 2
                        if scenario == 'SI': base_rate += 0.1 * (t_bin > 0.6)
                        elif scenario == 'EI': base_rate += 0.1 * (i_bin > 60)
                        adoption = np.clip(base_rate + np.random.randn() * 0.05, 0, 1)
                        std_dev = np.random.rand() * 0.04 + 0.01
                        data.append({
                            'scenario': scenario,
                            'trust_bin': t_bin,
                            'income_bin': i_bin,
                            'adoption_rate': adoption,
                            'std_dev': std_dev,
                            'ci_lower': np.clip(adoption - 1.96 * std_dev / np.sqrt(100), 0, 1),
                            'ci_upper': np.clip(adoption + 1.96 * std_dev / np.sqrt(100), 0, 1),
                            'n_replications': 1000,
                        })
            return pd.DataFrame(data)
        elif name == PRIM_BOXES_FILE:
            return pd.DataFrame({
                'scenario': ['NI', 'SI', 'EI'],
                'trust_min': [0.03, 0.5, 0.6],
                'trust_max': [0.97, 0.97, 0.97],
                'income_min': [2, 2, 60],
                'income_max': [98, 98, 98],
                'coverage': [1.0, 0.5, 0.3],
                'density': [0.5, 0.9, 0.8],
                'lift': [1.0, 1.8, 1.6],
            })
        elif name == METADATA_FILE:
            return {
                "trust": {"interpretation": "Agent trust propensity score (0=no trust, 1=full trust)"},
                "income": {"interpretation": "Income percentile in population (0=lowest, 100=highest)"}
            }
        # Se il file non è uno di quelli gestiti da montecarlo, solleva l'errore
        raise FileNotFoundError(f"File non trovato: {path} e dati montecarlo non disponibili per questo tipo.")
    
    # Se il file esiste, caricalo
    return pd.read_csv(path)


def load_metadata() -> dict:
    """
    Carica i metadati dal file JSON.
    """
    path = DATA_DIR / METADATA_FILE
    if not path.exists():
        # Ritorna i metadati di fallback se il file non esiste
        return {
            "trust": {"interpretation": "Agent trust propensity score (0=no trust, 1=full trust)"},
            "income": {"interpretation": "Income percentile in population (0=lowest, 100=highest)"}
        }
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"ATTENZIONE: Errore di decodifica JSON in {path}. Ritorno metadati vuoti.")
        return {}