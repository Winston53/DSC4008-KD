# models/burstormer/__init__.py
"""
Burstormer subpackage.

Convenience re-exports so callers can write:
    from models.burstormer import create_teacher
instead of the longer:
    from models.burstormer.burstormer_teacher import create_teacher
"""

from .burstormer_teacher import BurstormerTeacher, create_teacher

__all__ = ["BurstormerTeacher", "create_teacher"]