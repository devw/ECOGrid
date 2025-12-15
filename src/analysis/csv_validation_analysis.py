"""
CSV Data Analysis: Comparison with Theoretical Targets
"""

import argparse
import pandas as pd
from pathlib import Path
from .utils.analysis import analyze_weighted, analyze_brackets, analyze_prim
from .utils.analysis_constants import SCENARIOS

load = lambda d: map(pd.read_csv, (Path(d)/f for f in ('heatmap_grid.csv', 'prim_boxes.csv')))

by_scenario = lambda df, fn: {s: fn(df[df.scenario == s], s) for s in SCENARIOS}

def print_summary(weighted, prim_res, brackets_res):
    """
    Print a nicely aligned summary table with Avg, L/M/H adoption, Target, and Lift.
    """
    short = {
        'Low (0-20K)': 'L',
        'Middle (20-50K)': 'M',
        'High (50-100K)': 'H'
    }
    bracket_labels = ['Low (0-20K)', 'Middle (20-50K)', 'High (50-100K)']

    # Column widths
    widths = {
        'Scenario': 8,
        'Avg': 15,
        'Bracket': 12,
        'Target': 8,
        'Lift': 20
    }

    # Header
    header = (f"{'Scenario':<{widths['Scenario']}} | "
              f"{'Avg':<{widths['Avg']}} | "
              f"{'L':<{widths['Bracket']}} | "
              f"{'M':<{widths['Bracket']}} | "
              f"{'H':<{widths['Bracket']}} | "
              f"{'Target':<{widths['Target']}} | "
              f"{'Lift':<{widths['Lift']}}")
    print("\n📊 SUMMARY")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for s in SCENARIOS:
        w, p = weighted[s], prim_res[s]

        # Avg
        lo, hi = w['range']
        exp_avg = (lo + hi)/2
        obs_avg = w['value']
        avg_icon = '✅' if w['alignment']['aligned'] else '❌'
        avg_str = f"{exp_avg:.1f}/{obs_avg:.1f} {avg_icon}"

        # Fasce
        l_vals = []
        for br in bracket_labels:
            br_data = brackets_res.get(s, {}).get(br)
            if br_data:
                obs = br_data['avg']
                lo_hi = br_data.get('range', (obs, obs))
                in_range = lo_hi[0] <= obs <= lo_hi[1]
                icon = '✅' if in_range else ''
                l_vals.append(f"{obs:.1f} {icon}")
            else:
                l_vals.append("-")

        # Target
        exp_tgt = short.get(p['expected'], p['expected'])
        tgt_icon = '✅' if p['correct'] else '❌'
        tgt_str = f"{exp_tgt} {tgt_icon}"

        # Lift
        q, e = p['lift_q']
        lift_str = f"{p['lift']:.2f}x {e} {q}"

        # Print row with alignment
        print(f"{s:<{widths['Scenario']}} | "
              f"{avg_str:<{widths['Avg']}} | "
              f"{l_vals[0]:<{widths['Bracket']}} | "
              f"{l_vals[1]:<{widths['Bracket']}} | "
              f"{l_vals[2]:<{widths['Bracket']}} | "
              f"{tgt_str:<{widths['Target']}} | "
              f"{lift_str:<{widths['Lift']}}")

def main(base_dir):
    heatmap, prim = load(base_dir)
    weighted = by_scenario(heatmap, analyze_weighted)
    prim_res = analyze_prim(prim, heatmap)
    brackets = by_scenario(heatmap, analyze_brackets) 
    print_summary(weighted, prim_res, brackets)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', default='data/montecarlo_calibrated_fixed')
    main(ap.parse_args().d)
