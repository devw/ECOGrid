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

# --- build ScenarioResult ---
def make_result(s, weighted, prim_res, brackets_res):
    w, p = weighted[s], prim_res[s]
    lo, hi = w['range']; exp_avg, obs_avg = (lo+hi)/2, w['value']
    avg_icon = '✅' if w['alignment']['aligned'] else '❌'
    short = {'Low (0-20K)':'L','Middle (20-50K)':'M','High (50-100K)':'H'}
    brs = ['Low (0-20K','Middle (20-50K)','High (50-100K)']
    brackets = {b: (lambda d: BracketResult(d['avg'], *d['range'], '✅' if d['range'][0]<=d['avg']<=d['range'][1] else ''))(brackets_res[s][b]) 
                if brackets_res.get(s,{}).get(b) else BracketResult(-1,-1,-1,'-') for b in brs}
    tgt_icon = '✅' if p['correct'] else '❌'
    lift_str = f"{p['lift']:.2f}x {p['lift_q'][1]} {p['lift_q'][0]}"
    return ScenarioResult(s, exp_avg, obs_avg, avg_icon, brackets, short.get(p['expected'],p['expected']), tgt_icon, lift_str)

# --- print summary ---
def print_summary(weighted, prim_res, brackets_res):
    widths = dict(Scenario=8, Avg=15, Bracket=12, Target=8, Lift=20)
    header = f"{'Scenario':<{widths['Scenario']}} | {'Avg':<{widths['Avg']}} | " \
             f"{'L':<{widths['Bracket']}} | {'M':<{widths['Bracket']}} | {'H':<{widths['Bracket']}} | " \
             f"{'Target':<{widths['Target']}} | {'Lift':<{widths['Lift']}}"
    print("\n📊 SUMMARY\n" + "-"*len(header))
    print(header); print("-"*len(header))
    for s in SCENARIOS:
        res = make_result(s, weighted, prim_res, brackets_res)
        avg_str = f"{res.exp_avg:.1f}/{res.obs_avg:.1f} {res.avg_icon}"
        br_vals = [f"{b.avg:.1f} {b.icon}" if b.avg>=0 else '-' for b in res.brackets.values()]
        print(f"{res.name:<{widths['Scenario']}} | {avg_str:<{widths['Avg']}} | "
              f"{br_vals[0]:<{widths['Bracket']}} | {br_vals[1]:<{widths['Bracket']}} | {br_vals[2]:<{widths['Bracket']}} | "
              f"{res.target:<{widths['Target']}} {res.target_icon} | {res.lift:<{widths['Lift']}}")

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
