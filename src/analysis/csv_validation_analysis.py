"""
CSV Data Analysis: Comparison with Theoretical Targets
"""

import argparse
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from .utils.analysis import analyze_weighted, analyze_brackets, analyze_prim
from .utils.analysis_constants import SCENARIOS

# --- load CSVs ---
def load_data(base_dir):
    path = Path(base_dir)
    return pd.read_csv(path/'heatmap_grid.csv'), pd.read_csv(path/'prim_boxes.csv')

# --- scenario processing ---
def by_scenario(df, fn):
    return {s: fn(df[df.scenario == s], s) for s in SCENARIOS}

# --- build scenario row for tabulate ---
def make_row(s, weighted, prim_res, brackets_res):
    w, p = weighted[s], prim_res[s]
    lo, hi, val = w['range'][0], w['range'][1], w['value']
    avg_icon = '✅' if w['alignment']['aligned'] else '❌'
    avg_str = f"{(lo+hi)/2:.1f}(exp)/{val:.1f}(obs) {avg_icon}"

    brs = ['Low (0-20K)', 'Middle (20-50K)', 'High (50-100K)']
    fmt_bracket = lambda d: (
        (lambda a, r: f"{a:.1f} {'✅' if r[0] <= a <= r[1] else ''}")
        (d['avg'], d['range']) if d else '-'
    )
    br_vals = [fmt_bracket(brackets_res.get(s, {}).get(b)) for b in brs]

    short = {'Low (0-20K)':'L','Middle (20-50K)':'M','High (50-100K)':'H'}
    tgt_icon = '✅' if p['correct'] else '❌'
    lift_str = f"{p['lift']:.2f}x {p['lift_q'][1]} {p['lift_q'][0]}"
    target_str = short.get(p['expected'], p['expected']) + ' ' + tgt_icon
    return [s, avg_str] + br_vals + [target_str, lift_str]

# --- print summary using tabulate ---
def print_summary(weighted, prim_res, brackets_res):
    headers = ['Scenario','Avg','L','M','H','Target','Lift']
    table = [make_row(s, weighted, prim_res, brackets_res) for s in SCENARIOS]
    print("\n📊 SUMMARY\n")
    print(tabulate(table, headers=headers, tablefmt='github'))

# --- main ---
def main(base_dir):
    heatmap, prim = load_data(base_dir)
    weighted = by_scenario(heatmap, analyze_weighted)
    brackets = by_scenario(heatmap, analyze_brackets)
    prim_res = analyze_prim(prim, heatmap)
    print_summary(weighted, prim_res, brackets)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', default='data/montecarlo_calibrated_fixed')
    main(ap.parse_args().d)
