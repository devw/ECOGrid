"""
CSV Data Analysis: Comparison with Theoretical Targets
"""

import argparse
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from .utils.analysis import analyze_weighted, analyze_brackets, analyze_prim
from .utils.analysis_constants import SCENARIOS

# --- Data classes ---
@dataclass
class BracketResult:
    avg: float; lo: float; hi: float; icon: str=''

@dataclass
class ScenarioResult:
    name: str; exp_avg: float; obs_avg: float; avg_icon: str
    brackets: dict; target: str; target_icon: str; lift: str

# --- helpers ---
load = lambda d: map(pd.read_csv, (Path(d)/f for f in ('heatmap_grid.csv','prim_boxes.csv')))
by_scenario = lambda df, fn: {s: fn(df[df.scenario==s], s) for s in SCENARIOS}

def make_result(s, weighted, prim_res, brackets_res):
    w, p = weighted[s], prim_res[s]
    lo, hi = w['range']
    exp_avg, obs_avg = (lo+hi)/2, w['value']
    avg_icon = '✅' if w['alignment']['aligned'] else '❌'

    brs = ['Low (0-20K)', 'Middle (20-50K)', 'High (50-100K)']
    short = {'Low (0-20K)':'L','Middle (20-50K)':'M','High (50-100K)':'H'}
    brackets = {}
    for br in brs:
        d = brackets_res.get(s, {}).get(br)
        if d:
            icon = '✅' if d['range'][0] <= d['avg'] <= d['range'][1] else ''
            brackets[br] = BracketResult(d['avg'], *d['range'], icon)
        else:
            brackets[br] = BracketResult(-1,-1,-1,'-')

    tgt_icon = '✅' if p['correct'] else '❌'
    target = short.get(p['expected'], p['expected'])
    lift_str = f"{p['lift']:.2f}x {p['lift_q'][1]} {p['lift_q'][0]}"
    return ScenarioResult(s, exp_avg, obs_avg, avg_icon, brackets, target, tgt_icon, lift_str)

# --- print summary Markdown-friendly ---
def print_summary(weighted, prim_res, brackets_res):
    col_names = ['Scenario', 'Avg', 'L', 'M', 'H', 'Target', 'Lift']
    col_widths = [8, 15, 12, 12, 12, 8, 20]

    def fmt_cell(val, w): return f"{val:^{w}}"

    # Header
    header = "| " + " | ".join(fmt_cell(n,w) for n,w in zip(col_names, col_widths)) + " |"
    sep = "|-" + "-|-".join("-"*w for w in col_widths) + "-|"
    print("\n📊 SUMMARY\n" + sep + "\n" + header + "\n" + sep)

    # Rows
    for s in SCENARIOS:
        res = make_result(s, weighted, prim_res, brackets_res)
        avg_str = f"{res.exp_avg:.1f}/{res.obs_avg:.1f} {res.avg_icon}"
        br_vals = [f"{b.avg:.1f} {b.icon}" if b.avg>=0 else '-' for b in res.brackets.values()]
        row_vals = [res.name, avg_str] + br_vals + [f"{res.target} {res.target_icon}", res.lift]
        row = "| " + " | ".join(fmt_cell(v,w) for v,w in zip(row_vals, col_widths)) + " |"
        print(row)
    print(sep)
# --- main ---
def main(base_dir):
    heatmap, prim = load(base_dir)
    weighted = by_scenario(heatmap, analyze_weighted)
    brackets = by_scenario(heatmap, analyze_brackets)
    prim_res = analyze_prim(prim, heatmap)
    print_summary(weighted, prim_res, brackets)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', default='data/montecarlo_calibrated_fixed')
    main(ap.parse_args().d)
