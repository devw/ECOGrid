# src/analysis/utils/analysis.py

from .analysis_constants import SCENARIOS, THEORETICAL_DATA, LIFT_QUALITY, income_bracket

def check_alignment(v, r):
    lo, hi = r
    if lo <= v <= hi:
        pos = ['Lower third', 'Center', 'Upper third'][int(3*(v-lo)/(hi-lo))]
        return dict(aligned=True, gap=0, position=pos)
    return dict(aligned=False, gap=v-(lo if v < lo else hi),
                position='Below' if v < lo else 'Above')

def lift_quality(v):
    return next((q, e) for t, q, e in LIFT_QUALITY if v < t)


def analyze_weighted(df, s):
    avg = df.adoption_rate.mean() * 100
    r = THEORETICAL_DATA[s]['weighted_avg']
    return dict(value=avg, range=r, alignment=check_alignment(avg, r))


def analyze_brackets(df, s):
    df = df.assign(bracket=df.income_bin.map(income_bracket))
    return {
        b: (lambda d, r: None if d.empty else {
            'avg': d.adoption_rate.mean()*100,
            'n': len(d),
            'range': r,
            'alignment': check_alignment(d.adoption_rate.mean()*100, r)
        })(df[df.bracket == b], r)
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
    return {s: _one(heatmap[heatmap.scenario == s], s) for s in SCENARIOS}
