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
    lo, hi = w['range']
    exp_avg, obs_avg = (lo+hi)/2, w['value']
    avg_icon = '✅' if w['alignment']['aligned'] else '❌'
    avg_str = f"{exp_avg:.1f}/{obs_avg:.1f} {avg_icon}"

    brs = ['Low (0-20K)','Middle (20-50K)','High (50-100K)']
    short = {'Low (0-20K)':'L','Middle (20-50K)':'M','High (50-100K)':'H'}
    br_vals = []
    for b in brs:
        d = brackets_res.get(s, {}).get(b)
        if d:
            icon = '✅' if d['range'][0] <= d['avg'] <= d['range'][1] else ''
            br_vals.append(f"{d['avg']:.1f} {icon}")
        else:
            br_vals.append('-')

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
