"""
CSV Data Analysis: Comparison with Theoretical Targets
Analyzes heatmap_grid.csv and prim_boxes.csv to validate alignment with theoretical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

# ============================================================================
# THEORETICAL DATA DEFINITION
# ============================================================================

THEORETICAL_DATA = {
    'NI': {  # No Incentive
        'weighted_avg': (14.0, 20.0),
        'income_brackets': {
            'Low (0-20K)': {'range': (0, 20), 'adoption': (8.0, 15.0)},
            'Middle (20-50K)': {'range': (20, 50), 'adoption': (15.0, 22.0)},
            'High (50-100K)': {'range': (50, 100), 'adoption': (20.0, 28.0)},
        },
        'expected_target': 'High (50-100K)'
    },
    'EI': {  # Economic Incentive
        'weighted_avg': (29.0, 40.0),
        'income_brackets': {
            'Low (0-20K)': {'range': (0, 20), 'adoption': (30.0, 40.0)},
            'Middle (20-50K)': {'range': (20, 50), 'adoption': (28.0, 38.0)},
            'High (50-100K)': {'range': (50, 100), 'adoption': (30.0, 42.0)},
        },
        'expected_target': 'Low (0-20K)'
    },
    'SI': {  # Services Incentive
        'weighted_avg': (20.0, 30.0),
        'income_brackets': {
            'Low (0-20K)': {'range': (0, 20), 'adoption': (12.0, 20.0)},
            'Middle (20-50K)': {'range': (20, 50), 'adoption': (18.0, 28.0)},
            'High (50-100K)': {'range': (50, 100), 'adoption': (35.0, 48.0)},
        },
        'expected_target': 'High (50-100K)'
    }
}

LIFT_QUALITY = [
    (1.10, 'Poor', '❌'),
    (1.50, 'Moderate', '⚠️'),
    (2.00, 'Good', '✅'),
    (float('inf'), 'Excellent', '⭐')
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def classify_income_bracket(income_value: float) -> str:
    """Classify income value into theoretical brackets"""
    if income_value < 20:
        return 'Low (0-20K)'
    elif income_value < 50:
        return 'Middle (20-50K)'
    else:
        return 'High (50-100K)'


def get_lift_quality(lift_value: float) -> Tuple[str, str]:
    """Get quality assessment for LIFT value"""
    for threshold, quality, emoji in LIFT_QUALITY:
        if lift_value < threshold:
            return quality, emoji
    return 'Unknown', '❓'


def check_alignment(value: float, theoretical_range: Tuple[float, float]) -> Dict:
    """Check if value aligns with theoretical range"""
    min_val, max_val = theoretical_range
    aligned = min_val <= value <= max_val
    
    if value < min_val:
        gap = value - min_val
        position = 'Below'
    elif value > max_val:
        gap = value - max_val
        position = 'Above'
    else:
        gap = 0
        # Determine position within range
        range_size = max_val - min_val
        rel_position = (value - min_val) / range_size
        if rel_position < 0.33:
            position = 'Lower third'
        elif rel_position < 0.67:
            position = 'Center'
        else:
            position = 'Upper third'
    
    return {
        'aligned': aligned,
        'gap': gap,
        'position': position,
        'value': value,
        'range': theoretical_range
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(base_path: str = 'data/montecarlo_calibrated_fixed'):
    """Load CSV files"""
    heatmap_path = Path(base_path) / 'heatmap_grid.csv'
    prim_path = Path(base_path) / 'prim_boxes.csv'
    
    heatmap_df = pd.read_csv(heatmap_path)
    prim_df = pd.read_csv(prim_path)
    
    return heatmap_df, prim_df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_weighted_averages(heatmap_df: pd.DataFrame) -> Dict:
    """Analyze weighted average adoption rates"""
    results = {}
    
    for scenario in ['NI', 'EI', 'SI']:
        scenario_data = heatmap_df[heatmap_df['scenario'] == scenario]
        
        # Calculate weighted average
        weighted_avg = scenario_data['adoption_rate'].mean()
        
        # Get theoretical range
        theoretical_range = THEORETICAL_DATA[scenario]['weighted_avg']
        
        # Check alignment
        alignment = check_alignment(weighted_avg * 100, theoretical_range)
        
        results[scenario] = {
            'weighted_avg': weighted_avg * 100,
            'theoretical_range': theoretical_range,
            'alignment': alignment
        }
    
    return results


def analyze_income_brackets(heatmap_df: pd.DataFrame) -> Dict:
    """Analyze adoption rates by income brackets"""
    results = {}
    
    for scenario in ['NI', 'EI', 'SI']:
        scenario_data = heatmap_df[heatmap_df['scenario'] == scenario].copy()
        
        # Classify income bins into brackets
        scenario_data['bracket'] = scenario_data['income_bin'].apply(classify_income_bracket)
        
        # Calculate average adoption per bracket
        bracket_results = {}
        for bracket_name, bracket_info in THEORETICAL_DATA[scenario]['income_brackets'].items():
            bracket_data = scenario_data[scenario_data['bracket'] == bracket_name]
            
            if len(bracket_data) > 0:
                avg_adoption = bracket_data['adoption_rate'].mean() * 100
                theoretical_range = bracket_info['adoption']
                
                alignment = check_alignment(avg_adoption, theoretical_range)
                
                bracket_results[bracket_name] = {
                    'avg_adoption': avg_adoption,
                    'theoretical_range': theoretical_range,
                    'n_samples': len(bracket_data),
                    'alignment': alignment
                }
            else:
                bracket_results[bracket_name] = {
                    'avg_adoption': None,
                    'theoretical_range': bracket_info['adoption'],
                    'n_samples': 0,
                    'alignment': None
                }
        
        results[scenario] = bracket_results
    
    return results


def analyze_prim_boxes(prim_df: pd.DataFrame, heatmap_df: pd.DataFrame) -> Dict:
    """Analyze PRIM boxes alignment with theoretical targets"""
    results = {}
    
    for scenario in ['NI', 'EI', 'SI']:
        prim_box = prim_df[prim_df['scenario'] == scenario].iloc[0]
        
        # Get overall average for lift calculation
        scenario_data = heatmap_df[heatmap_df['scenario'] == scenario]
        overall_avg = scenario_data['adoption_rate'].mean() * 100
        
        # Estimate adoption in box
        adoption_in_box = overall_avg * prim_box['lift']
        
        # Identify which bracket the PRIM box targets
        income_range = (prim_box['income_min'], prim_box['income_max'])
        
        # Determine target bracket
        if income_range[1] <= 30:
            target_bracket = 'Low (0-20K)'
        elif income_range[0] >= 50:
            target_bracket = 'High (50-100K)'
        elif income_range == (0, 100):
            target_bracket = 'All brackets (no segmentation)'
        else:
            target_bracket = 'Middle (20-50K)'
        
        # Get expected target
        expected_target = THEORETICAL_DATA[scenario]['expected_target']
        
        # Get theoretical range for the identified bracket
        if target_bracket in THEORETICAL_DATA[scenario]['income_brackets']:
            theoretical_range = THEORETICAL_DATA[scenario]['income_brackets'][target_bracket]['adoption']
            alignment = check_alignment(adoption_in_box, theoretical_range)
        else:
            theoretical_range = None
            alignment = None
        
        # LIFT quality assessment
        lift_quality, lift_emoji = get_lift_quality(prim_box['lift'])
        
        results[scenario] = {
            'income_range': income_range,
            'target_bracket': target_bracket,
            'expected_target': expected_target,
            'correct_targeting': target_bracket == expected_target,
            'coverage': prim_box['coverage'] * 100,
            'density': prim_box['density'] * 100,
            'lift': prim_box['lift'],
            'lift_quality': lift_quality,
            'lift_emoji': lift_emoji,
            'adoption_in_box': adoption_in_box,
            'theoretical_range': theoretical_range,
            'alignment': alignment,
            'trust_range': (prim_box['trust_min'], prim_box['trust_max'])
        }
    
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(weighted_results: Dict, bracket_results: Dict, prim_results: Dict):
    """Generate comprehensive analysis report"""
    
    scenario_names = {
        'NI': 'No Incentive',
        'EI': 'Economic Incentive',
        'SI': 'Services Incentive'
    }
    
    print("="*80)
    print("📊 COMPREHENSIVE CSV DATA ANALYSIS vs THEORETICAL TARGETS")
    print("="*80)
    print()
    
    # ========================================================================
    # SECTION 1: WEIGHTED AVERAGES
    # ========================================================================
    print("─"*80)
    print("✅ SECTION 1: WEIGHTED AVERAGE ALIGNMENT")
    print("─"*80)
    print()
    
    for scenario in ['NI', 'SI', 'EI']:
        result = weighted_results[scenario]
        alignment = result['alignment']
        
        status = '✅' if alignment['aligned'] else '❌'
        
        print(f"📌 {scenario_names[scenario]} ({scenario})")
        print(f"   Observed:    {result['weighted_avg']:.1f}%")
        print(f"   Theoretical: {result['theoretical_range'][0]:.1f}% - {result['theoretical_range'][1]:.1f}%")
        print(f"   Status:      {status} {alignment['position']}")
        if not alignment['aligned']:
            print(f"   Gap:         {alignment['gap']:.1f} percentage points")
        print()
    
    # ========================================================================
    # SECTION 2: INCOME BRACKETS
    # ========================================================================
    print("─"*80)
    print("⚠️ SECTION 2: INCOME BRACKET ANALYSIS")
    print("─"*80)
    print()
    
    for scenario in ['NI', 'SI', 'EI']:
        print(f"📌 {scenario_names[scenario]} ({scenario})")
        print()
        
        for bracket_name, bracket_data in bracket_results[scenario].items():
            if bracket_data['avg_adoption'] is not None:
                alignment = bracket_data['alignment']
                status = '✅' if alignment['aligned'] else '❌'
                
                print(f"   {bracket_name}:")
                print(f"      Observed:    {bracket_data['avg_adoption']:.1f}%")
                print(f"      Theoretical: {bracket_data['theoretical_range'][0]:.1f}% - {bracket_data['theoretical_range'][1]:.1f}%")
                print(f"      Status:      {status} {alignment['position']}")
                if not alignment['aligned']:
                    print(f"      Gap:         {alignment['gap']:.1f} percentage points")
                print(f"      Samples:     {bracket_data['n_samples']}")
                print()
        print()
    
    # ========================================================================
    # SECTION 3: PRIM BOX ANALYSIS
    # ========================================================================
    print("─"*80)
    print("📦 SECTION 3: PRIM BOX TARGETING ANALYSIS")
    print("─"*80)
    print()
    
    for scenario in ['NI', 'SI', 'EI']:
        result = prim_results[scenario]
        
        targeting_status = '✅' if result['correct_targeting'] else '❌'
        
        print(f"📌 {scenario_names[scenario]} ({scenario})")
        print(f"   Income Range:       {result['income_range'][0]:.0f} - {result['income_range'][1]:.0f}")
        print(f"   Target Bracket:     {result['target_bracket']}")
        print(f"   Expected Target:    {result['expected_target']}")
        print(f"   Correct Targeting:  {targeting_status}")
        print(f"   Coverage:           {result['coverage']:.1f}%")
        print(f"   Density:            {result['density']:.1f}%")
        print(f"   LIFT:               {result['lift']:.2f}x {result['lift_emoji']} ({result['lift_quality']})")
        print(f"   Adoption in Box:    {result['adoption_in_box']:.1f}%")
        
        if result['theoretical_range'] and result['alignment']:
            alignment = result['alignment']
            status = '✅' if alignment['aligned'] else '❌'
            print(f"   Theoretical Range:  {result['theoretical_range'][0]:.1f}% - {result['theoretical_range'][1]:.1f}%")
            print(f"   Alignment:          {status} {alignment['position']}")
            if not alignment['aligned']:
                print(f"   Gap:                {alignment['gap']:.1f} percentage points")
        print()
    
    # ========================================================================
    # SECTION 4: LIFT QUALITY SUMMARY
    # ========================================================================
    print("─"*80)
    print("📈 SECTION 4: LIFT QUALITY ASSESSMENT")
    print("─"*80)
    print()
    
    print("Lift Quality Scale:")
    print("   < 1.10  = Poor      ❌ (Box barely better than random)")
    print("   1.10-1.50 = Moderate  ⚠️  (Decent targeting advantage)")
    print("   1.50-2.00 = Good      ✅ (Strong targeting benefit)")
    print("   > 2.00  = Excellent ⭐ (Outstanding performance)")
    print()
    
    for scenario in ['NI', 'SI', 'EI']:
        result = prim_results[scenario]
        print(f"   {scenario_names[scenario]:20} LIFT = {result['lift']:.2f}x  {result['lift_emoji']} {result['lift_quality']}")
    print()
    
    # ========================================================================
    # SECTION 5: SUMMARY TABLE
    # ========================================================================
    print("─"*80)
    print("📋 SECTION 5: COMPREHENSIVE SUMMARY TABLE")
    print("─"*80)
    print()
    
    print(f"{'Scenario':<20} {'Avg Align':<12} {'Target OK':<12} {'LIFT':<10} {'Overall':<10}")
    print("─"*80)
    
    for scenario in ['NI', 'SI', 'EI']:
        avg_align = '✅' if weighted_results[scenario]['alignment']['aligned'] else '❌'
        target_ok = '✅' if prim_results[scenario]['correct_targeting'] else '❌'
        lift_emoji = prim_results[scenario]['lift_emoji']
        lift_val = prim_results[scenario]['lift']
        
        # Calculate overall score
        score = 0
        if weighted_results[scenario]['alignment']['aligned']:
            score += 1
        if prim_results[scenario]['correct_targeting']:
            score += 1
        if lift_val >= 1.50:
            score += 1
            overall = '✅ Good'
        elif lift_val >= 1.10:
            if score >= 1:
                overall = '⚠️ Partial'
            else:
                overall = '❌ Poor'
        else:
            overall = '❌ Poor'
        
        print(f"{scenario_names[scenario]:<20} {avg_align:<12} {target_ok:<12} {lift_emoji} {lift_val:.2f}x   {overall}")
    
    print()
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis execution"""
    
    # Load data
    print("Loading CSV files...")
    heatmap_df, prim_df = load_data()
    print(f"✅ Loaded {len(heatmap_df)} heatmap records and {len(prim_df)} PRIM boxes")
    print()
    
    # Perform analyses
    print("Analyzing data...")
    weighted_results = analyze_weighted_averages(heatmap_df)
    bracket_results = analyze_income_brackets(heatmap_df)
    prim_results = analyze_prim_boxes(prim_df, heatmap_df)
    print("✅ Analysis complete")
    print()
    
    # Generate report
    generate_report(weighted_results, bracket_results, prim_results)


if __name__ == "__main__":
    main()