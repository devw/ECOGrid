"""
Report generation utilities for adoption rate analysis.
Provides textual summaries of adoption rates and PRIM box analysis.
"""

from typing import Dict
import pandas as pd
import numpy as np


class AdoptionReportGenerator:
    """Generates textual reports for adoption rate analysis."""
    
    @staticmethod
    def _interpret_coverage(coverage: float) -> str:
        """Interpret population coverage percentage."""
        if coverage < 0.15:
            return "Very limited reach (elite only)"
        elif coverage < 0.30:
            return "Limited reach (excludes majority)"
        elif coverage < 0.50:
            return "Moderate reach (covers minority)"
        else:
            return "Good reach (covers majority)"
    
    @staticmethod
    def _interpret_lift(lift: float) -> str:
        """Interpret effectiveness lift factor."""
        if lift > 2.0:
            return "Highly effective in target region"
        elif lift > 1.5:
            return "Moderately effective in target region"
        else:
            return "Modest effectiveness in target region"
    
    @staticmethod
    def _format_scenario_header(title: str) -> str:
        """Format scenario section header."""
        return f"\n{'─'*80}\n📌 {title.upper()}\n{'─'*80}"
    
    @staticmethod
    def _format_adoption_stats(grid: dict) -> str:
        """Format adoption rate statistics section."""
        adoption_data = grid['adoption'].flatten()
        lines = [
            "\n  📈 Adoption Rate Statistics:",
            f"     • Average (α):  {grid['avg_adoption']:.1%}",
            f"     • Minimum:      {adoption_data.min():.1%}",
            f"     • Maximum:      {adoption_data.max():.1%}",
            f"     • Std Dev (σ):  {adoption_data.std():.3f}",
            f"     • Replications: {grid['n_replications']:,}"
        ]
        return "\n".join(lines)
    
    @staticmethod
    def _format_prim_box(prim: pd.Series, avg_adoption: float) -> str:
        """Format PRIM box analysis section."""
        if prim is None:
            return "\n  ⚠️  No PRIM box identified (low variance in adoption)"
        
        estimated_adoption = avg_adoption * prim['lift']
        
        lines = [
            "\n  🎯 PRIM Box (High-Adoption Region):",
            f"     • Coverage:  {prim['coverage']:.1%} of population",
            f"     • Density:   {prim['density']:.1%} of high-adoption cases",
            f"     • Lift:      {prim['lift']:.2f}x above average",
            f"     • Est. adoption in box: ~{estimated_adoption:.1%}",
            "\n  📦 Box Boundaries:",
            f"     • Trust:  [{prim['trust_min']:.3f}, {prim['trust_max']:.3f}]",
            f"     • Income: [{prim['income_min']:.1f}, {prim['income_max']:.1f}]",
        ]
        return "\n".join(lines)
    
    @staticmethod
    def _format_interpretation(prim: pd.Series) -> str:
        """Format interpretation section."""
        if prim is None:
            return ""
        
        reach = AdoptionReportGenerator._interpret_coverage(prim['coverage'])
        effectiveness = AdoptionReportGenerator._interpret_lift(prim['lift'])
        
        lines = [
            "\n  💡 Interpretation:",
            f"     • Population reach: {reach}",
            f"     • Effectiveness: {effectiveness}"
        ]
        return "\n".join(lines)
    
    @classmethod
    def generate_scenario_report(cls, code: str, title: str, grid: dict) -> str:
        """Generate complete report for a single scenario."""
        sections = [
            cls._format_scenario_header(title),
            cls._format_adoption_stats(grid),
            cls._format_prim_box(grid.get('prim'), grid['avg_adoption']),
            cls._format_interpretation(grid.get('prim')),
        ]
        return "\n".join(sections) + "\n"
    
    @classmethod
    def generate_full_report(cls, grids: Dict[str, dict], scenarios: Dict[str, str]) -> str:
        """Generate complete analysis report for all scenarios."""
        header = "\n" + "="*80 + "\n📊 ADOPTION RATE & PRIM BOX ANALYSIS REPORT\n" + "="*80
        
        scenario_reports = [
            cls.generate_scenario_report(code, title, grids[code])
            for code, title in scenarios.items()
        ]
        
        footer = "="*80 + "\n"
        
        return header + "".join(scenario_reports) + footer


def print_analysis_report(grids: Dict[str, dict], scenarios: Dict[str, str]) -> None:
    """
    Print textual analysis of adoption rates and PRIM boxes.
    
    Args:
        grids: Dictionary mapping scenario codes to grid data
        scenarios: Dictionary mapping scenario codes to titles
    """
    report = AdoptionReportGenerator.generate_full_report(grids, scenarios)
    print(report)
