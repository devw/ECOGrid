"""
Utility functions for displaying standard CLI output
in a consistent and reusable way across scripts.
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
def show_header(title: str, version: str | None = None) -> None:
    """
    Display a standardized header for CLI tools.
    """
    line = "=" * 70
    print(line)
    header = f"🚀 {title}"
    if version:
        header += f" (v{version})"
    print(header)
    print(line)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
def show_config(config: dict[str, Any] | Any) -> None:
    """
    Display configuration values.

    Parameters
    ----------
    config : dict or object
        Either:
        - a dict of key → value pairs
        - any object with attributes to display
    """
    print("\nConfiguration:")

    if isinstance(config, dict):
        for key, value in config.items():
            print(f"  • {key}: {value}")
    else:
        # Generic object with attributes
        attrs = [a for a in dir(config) if not a.startswith("_")]
        for attr in attrs:
            value = getattr(config, attr)
            if not callable(value):
                print(f"  • {attr}: {value}")


# ---------------------------------------------------------
# STEP MESSAGES
# ---------------------------------------------------------
def show_step(message: str) -> None:
    """
    Display a mid-process step message (generic and reusable).
    """
    print(f"\n➡️  {message}")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
def show_summary(summary: dict[str, Any] | Any) -> None:
    """
    Display a standardized summary or list of outputs.

    Parameters
    ----------
    summary : dict or object
        - For dict: key → output/filepath/description
        - For objects: attributes will be inspected
    """
    line = "=" * 70
    print("\n" + line)
    print("✅ Process complete!")
    print(line)

    print("\n📊 Summary:")

    if isinstance(summary, dict):
        for key, value in summary.items():
            print(f"  • {key}: {value}")
    else:
        attrs = [a for a in dir(summary) if not a.startswith("_")]
        for attr in attrs:
            value = getattr(summary, attr)
            if not callable(value):
                print(f"  • {attr}: {value}")
