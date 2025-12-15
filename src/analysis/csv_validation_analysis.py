"""
CSV Data Analysis: Comparison with Theoretical Targets
"""

import argparse
import pandas as pd
from pathlib import Path

SCENARIOS = ['NI', 'EI', 'SI']

THEORY = {
    'NI': dict(
        weighted=(14, 20), target='High (50-100K)',
        brackets={'Low (0-20K)': (8, 15), 'Middle (20-50K)': (15, 22), 'High (50-100K)': (20, 28)}
    ),
    'EI': dict(
        weighted=(29, 40), target='Low (0-20K)',
        brackets={'Low (0-20K)': (30, 40), 'Middle (20-50K)': (28, 38), 'High (50-100K)': (30, 42)}
    ),
    'SI': dict(
        weighted=(20, 30), target='High (50-100K)',
        brackets={'Low (0-20K)': (12, 20), 'Middle (20-50K)': (18, 28), 'High (50-100K)': (35, 48)}
    )
}

LIFT_Q = [(1.10, 'Poor', '❌'), (1.50, 'Moderate', '⚠️'),
          (2.00, 'Good', '✅'), (float('inf'), 'Excellent', '⭐')]

income_bracket = lambda x: (
    'Low (0-20K)' if x < 20 else
    'Middle (20-50K)' if x < 50 else
    'High (50-100K)'
)

align = lambda v, r: (
    {'aligned': True, 'gap': 0,
     'position': ['Lower third', 'Center', 'Upper third'][int(3*(v-r[0])/(r[1]-r[0]))]}
    if r[0] <= v <= r[1]
    else {'aligned': False, 'gap': v-(r[0] if v < r[0] else r[1]),
          'position': 'Below' if v < r[0] else 'Above'}
)

lift_quality = lambda v: next((q, e) for t, q, e in LIFT_Q if v < t)

load = lambda d: map(pd.read_csv, (Path(d)/f for f in ('heatmap_grid.csv', 'prim_boxes.csv')))

by_scenario = lambda df, fn: {s: fn(df[df.scenario == s], s) for s in SCENARIOS}

def analyze_weighted(df, s):
    v = df.adoption_rate.mean() * 100
    r = THEORY[s]['weighted']
    return dict(value=v, range=r, alignment=align(v, r))

def analyze_brackets(df, s):
    df = df.assign(b=df.income_bin.map(income_bracket))
    return {
        k: None if (d := df[df.b == k]).empty else {
            'avg': (v := d.adoption_rate.mean() * 100),
            'n': len(d),
            'alignment': align(v, r)
        }
        for k, r in THEORY[s]['brackets'].items()
    }

def analyze_prim(prim, heatmap):
    def one(df, s):
        box = prim[prim.scenario == s].iloc[0]
        avg = df.adoption_rate.mean() * 100
        adopt = avg * box.lift
        bracket = (
            'All brackets (no segmentation)'
            if (box.income_min, box.income_max) == (0, 100)
            else income_bracket(box.income_min)
        )
        r = THEORY[s]['brackets'].get(bracket)
        return dict(
            target=bracket,
            expected=THEORY[s]['target'],
            correct=bracket == THEORY[s]['target'],
            lift=box.lift,
            lift_q=lift_quality(box.lift),
            adoption=adopt,
            alignment=align(adopt, r) if r else None,
            coverage=box.coverage * 100,
            density=box.density * 100
        )
    return by_scenario(heatmap, one)

def print_summary(weighted, prim_res):
    short = {
        'Low (0-20K)': 'L',
        'Middle (20-50K)': 'M',
        'High (50-100K)': 'H'
    }

    print("\n📊 SUMMARY\n" + "-" * 60)
    for s in SCENARIOS:
        w, p = weighted[s], prim_res[s]

        lo, hi = w['range']
        exp_avg = (lo + hi) / 2
        obs_avg = w['value']
        avg_icon = '✅' if w['alignment']['aligned'] else '❌'

        exp_tgt, obs_tgt = p['expected'], p['target']
        tgt_icon = '✅' if p['correct'] else '❌'

        tgt_str = (
            f"Target exp={short.get(exp_tgt, exp_tgt)}"
            if p['correct']
            else f"Target exp={exp_tgt} obs={obs_tgt}"
        )

        q, e = p['lift_q']

        print(
            f"{s}: "
            f"Avg exp={exp_avg:.1f} obs={obs_avg:.1f} {avg_icon} | "
            f"{tgt_str} {tgt_icon} | "
            f"LIFT {p['lift']:.2f}x {e} {q}"
        )

def main(base_dir):
    heatmap, prim = load(base_dir)
    weighted = by_scenario(heatmap, analyze_weighted)
    prim_res = analyze_prim(prim, heatmap)
    print_summary(weighted, prim_res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', default='data/montecarlo_calibrated_fixed')
    main(ap.parse_args().d)
