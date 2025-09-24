"""
Interface layer - User interfaces and external API adapters.

This module contains adapters for different user interfaces including CLI and web interfaces.
"""

from .cli import CLI, create_cli

__all__ = [
    "CLI",
    "create_cli",
]
