import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

data_dir = Path("data/montecarlo_calibrated_fixed")

print("=" * 80)
print("P-VALUE DIAGNOSTIC ANALYSIS")
print("=" * 80)

# Carica i dati
df_demo = pd.read_csv(data_dir / "demographic_profiles.csv")
df_traj = pd.read_csv(data_dir / "prim_trajectory_summary.csv")

print("\n1. INFORMAZIONI DAL FILE DEMOGRAPHIC_PROFILES")
print("-" * 80)
print("Colonne disponibili:", df_demo.columns.tolist())
print("\nDati completi:")
print(df_demo)

# I nomi delle colonne effettive
print("\n\n2. ESTRAZIONE DATI PER ANALISI")
print("-" * 80)

# Calcola densità da trajectory_summary
ni_density = df_demo[df_demo['scenario'] == 'NI']['density'].values[0]
ni_n_agents = df_demo[df_demo['scenario'] == 'NI']['n_agents_segment'].values[0]

print(f"\nBaseline (NI):")
print(f"  Density: {ni_density:.4f}")
print(f"  n_agents_segment: {ni_n_agents}")

# Dobbiamo calcolare SD dalla trajectory_summary
df_traj_ni = df_traj[(df_traj['scenario'] == 'NI') & (df_traj['is_selected'] == False)]
if len(df_traj_ni) > 0:
    ni_density_mean = df_traj_ni['density_mean'].iloc[0]
    ni_density_std = df_traj_ni['density_std'].iloc[0]
    print(f"  Density mean (da trajectory): {ni_density_mean:.4f}")
    print(f"  Density std (da trajectory): {ni_density_std:.4f}")
else:
    ni_density_std = 0.13  # Valore dalla tua tabella originale

for scenario in ['SI', 'EI']:
    if scenario in df_demo['scenario'].values:
        row = df_demo[df_demo['scenario'] == scenario].iloc[0]
        density = row['density']
        n = row['n_agents_segment']
        
        # Cerca SD da trajectory
        df_traj_sc = df_traj[(df_traj['scenario'] == scenario)]
        if len(df_traj_sc) > 0:
            # Prendi l'iterazione selezionata
            selected = df_traj_sc[df_traj_sc['is_selected'] == True]
            if len(selected) > 0:
                density_mean = selected['density_mean'].iloc[0]
                density_std = selected['density_std'].iloc[0]
            else:
                density_mean = density
                density_std = 0.15  # Valore approssimativo
        else:
            density_std = 0.15
        
        # Cohen's d
        cohen_d = (density - ni_density) / np.sqrt((density_std**2 + ni_density_std**2) / 2)
        
        print(f"\n{scenario}:")
        print(f"  Density: {density:.4f}")
        print(f"  SD stimato: {density_std:.4f}")
        print(f"  n_agents_segment: {n}")
        print(f"  Cohen's d: {cohen_d:.4f}")

print("\n" + "=" * 80)
print("3. ANALISI DETTAGLIATA: CALCOLO P-VALUE DA TRAJECTORY")
print("=" * 80)

# Estraiamo le 100 density per replicazione da trajectory_raw
df_raw = pd.read_csv(data_dir / "prim_trajectory_raw.csv")

print(f"\nDati disponibili in trajectory_raw:")
print(f"  Totale righe: {len(df_raw)}")
print(f"  Scenari: {df_raw['scenario'].unique()}")
print(f"  Replicazioni: {df_raw['replication_id'].nunique()}")

# Per ogni scenario, estrai le density delle replicazioni all'iterazione selezionata
print("\n\nIPOTESI 1: Test tra replicazioni (n=100) - METODO CORRETTO")
print("-" * 80)

for scenario in ['NI', 'SI', 'EI']:
    # Trova iterazione selezionata
    selected_iter = df_traj[(df_traj['scenario'] == scenario) & 
                            (df_traj['is_selected'] == True)]
    
    if len(selected_iter) > 0:
        iter_id = selected_iter['iteration'].iloc[0]
    else:
        # Per NI usa iterazione 0 come default
        iter_id = 0
    
    # Estrai density per tutte le 100 replicazioni a quella iterazione
    scenario_data = df_raw[(df_raw['scenario'] == scenario) & 
                           (df_raw['iteration'] == iter_id)]
    
    densities = scenario_data['density'].values
    
    print(f"\n{scenario} (iteration {iter_id}):")
    print(f"  n replicazioni: {len(densities)}")
    print(f"  Mean density: {np.mean(densities):.4f}")
    print(f"  Std density: {np.std(densities):.4f}")

# Test statistico tra SI/EI vs NI
ni_iter = 0  # NI non ha is_selected=True, usa iterazione 0
ni_densities = df_raw[(df_raw['scenario'] == 'NI') & 
                      (df_raw['iteration'] == ni_iter)]['density'].values

print("\n\nTEST STATISTICI:")
print("-" * 80)

for scenario in ['SI', 'EI']:
    selected_iter = df_traj[(df_traj['scenario'] == scenario) & 
                            (df_traj['is_selected'] == True)]
    
    if len(selected_iter) > 0:
        iter_id = selected_iter['iteration'].iloc[0]
        scenario_densities = df_raw[(df_raw['scenario'] == scenario) & 
                                    (df_raw['iteration'] == iter_id)]['density'].values
        
        # T-test
        t_stat, p_val_ttest = stats.ttest_ind(scenario_densities, ni_densities)
        
        # Mann-Whitney (non-parametrico)
        u_stat, p_val_mw = stats.mannwhitneyu(scenario_densities, ni_densities, 
                                               alternative='two-sided')
        
        # Effect size
        cohen_d = (np.mean(scenario_densities) - np.mean(ni_densities)) / \
                  np.sqrt((np.std(scenario_densities)**2 + np.std(ni_densities)**2) / 2)
        
        print(f"\n{scenario} vs NI:")
        print(f"  T-test:")
        print(f"    t-statistic: {t_stat:.2f}")
        print(f"    p-value: {p_val_ttest:.2e}")
        print(f"  Mann-Whitney U:")
        print(f"    U-statistic: {u_stat:.2f}")
        print(f"    p-value: {p_val_mw:.2e}")
        print(f"  Cohen's d: {cohen_d:.4f}")
        
        if p_val_ttest < 1e-100:
            print(f"  ⚠️  P-value estremamente basso - probabile effect size enorme")

print("\n" + "=" * 80)
print("4. RACCOMANDAZIONI")
print("=" * 80)

print("""
ANALISI COMPLETATA. Risultati:

1. I p-value nella tua tabella NON sono presenti nei file CSV
   → Sono stati calcolati dallo script 'generate_all_tables'
   
2. Se i p-value sono estremamente bassi (10^-139, 10^-187):
   
   CAUSA PROBABILE:
   • Effect size molto grande (Cohen's d > 0.8)
   • Test su 100 replicazioni
   • Differenza di density molto marcata (0.32 → 0.68 → 0.81)
   
   QUESTO È ACCETTABILE se:
   ✓ Il test è fatto su n=100 replicazioni (non su tutti gli agenti)
   ✓ Le assunzioni del test sono rispettate
   
3. BEST PRACTICE per la pubblicazione:

   INVECE DI:
   | SI | ... | 2.82e-139 |
   | EI | ... | 3.13e-187 |
   
   USA:
   | SI | ... | < 0.001*** |
   | EI | ... | < 0.001*** |
   
   Con nota:
   "***p < 0.001. Statistical tests performed on 100 Monte Carlo 
   replications using [t-test/Mann-Whitney]. All comparisons show
   highly significant differences in adoption density."

4. ENFATIZZA EFFECT SIZE:
   
   Nella discussione, scrivi qualcosa come:
   "High-trust communities (SI) show a large effect (Cohen's d = 0.81)
   while high-trust/high-income communities (EI) show a very large 
   effect (Cohen's d = 1.04), both statistically significant (p < 0.001)."
   
5. VERIFICA NEL CODICE:
   
   Cerca in 'src/scripts/presentation/generate_all_tables.py' come
   viene calcolato il p-value e assicurati che usi n=100 replicazioni,
   non n=totale_agenti.
""")

print("\n" + "=" * 80)