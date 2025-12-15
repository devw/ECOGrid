"""
CSV Data Analysis: Comparison with Theoretical Targets
"""

import pandas as pd
from pathlib import Path

SCENARIOS = ['NI', 'EI', 'SI']

THEORETICAL_DATA = {
    'NI': {'weighted_avg': (14, 20), 'expected_target': 'High (50-100K)',
           'income_brackets': {
               'Low (0-20K)': (8, 15),
               'Middle (20-50K)': (15, 22),
               'High (50-100K)': (20, 28)}},
    'EI': {'weighted_avg': (29, 40), 'expected_target': 'Low (0-20K)',
           'income_brackets': {
               'Low (0-20K)': (30, 40),
               'Middle (20-50K)': (28, 38),
               'High (50-100K)': (30, 42)}},
    'SI': {'weighted_avg': (20, 30), 'expected_target': 'High (50-100K)',
           'income_brackets': {
               'Low (0-20K)': (12, 20),
               'Middle (20-50K)': (18, 28),
               'High (50-100K)': (35, 48)}}
}

LIFT_QUALITY = [(1.10, 'Poor', '❌'), (1.50, 'Moderate', '⚠️'),
                (2.00, 'Good', '✅'), (float('inf'), 'Excellent', '⭐')]

income_bracket = lambda x: (
    'Low (0-20K)' if x < 20 else
    'Middle (20-50K)' if x < 50 else
    'High (50-100K)'
)

def check_alignment(v, r):
    lo, hi = r
    if lo <= v <= hi:
        pos = ['Lower third', 'Center', 'Upper third'][int(3*(v-lo)/(hi-lo))]
        return dict(aligned=True, gap=0, position=pos)
    return dict(aligned=False, gap=v-(lo if v < lo else hi),
                position='Below' if v < lo else 'Above')

def lift_quality(v):
    return next((q, e) for t, q, e in LIFT_QUALITY if v < t)

def load_data(base='data/montecarlo_calibrated_fixed'):
    p = Path(base)
    return (pd.read_csv(p/'heatmap_grid.csv'),
            pd.read_csv(p/'prim_boxes.csv'))

def analyze_by_scenario(df, fn):
    return {s: fn(df[df.scenario == s], s) for s in SCENARIOS}

def analyze_weighted(df, s):
    avg = df.adoption_rate.mean() * 100
    r = THEORETICAL_DATA[s]['weighted_avg']
    return dict(value=avg, range=r, alignment=check_alignment(avg, r))

def analyze_brackets(df, s):
    df = df.assign(bracket=df.income_bin.map(income_bracket))
    return {
        b: (lambda d: None if d.empty else {
            'avg': d.adoption_rate.mean()*100,
            'n': len(d),
            'alignment': check_alignment(d.adoption_rate.mean()*100, r)
        })(df[df.bracket == b])
        for b, r in THEORETICAL_DATA[s]['income_brackets'].items()
    }

def analyze_prim(prim, heatmap):
    def _one(df, s):
        box = prim[prim.scenario == s].iloc[0]
        avg = df.adoption_rate.mean() * 100
        adopt = avg * box.lift
        bracket = (
            'All brackets (no segmentation)' if (box.income_min, box.income_max) == (0, 100)
            else income_bracket(box.income_min)
        )
        tq = THEORETICAL_DATA[s]['income_brackets'].get(bracket)
        return {
            'target': bracket,
            'expected': THEORETICAL_DATA[s]['expected_target'],
            'correct': bracket == THEORETICAL_DATA[s]['expected_target'],
            'lift': box.lift,
            'lift_q': lift_quality(box.lift),
            'adoption': adopt,
            'alignment': check_alignment(adopt, tq) if tq else None,
            'coverage': box.coverage*100,
            'density': box.density*100
        }
    return analyze_by_scenario(heatmap, _one)

def main():
    heatmap, prim = load_data()
    weighted = analyze_by_scenario(heatmap, analyze_weighted)
    brackets = analyze_by_scenario(heatmap, analyze_brackets)
    prim_res = analyze_prim(prim, heatmap)
    print_summary_table(weighted, prim)


if __name__ == "__main__":
    main()
