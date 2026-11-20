# losses/__init__.py
"""
Loss functions for output-level KD.

Exports:
    OutputKDLoss:      L1(student_out, teacher_out)
    GroundTruthLoss:   L1(student_out, gt)
"""

from .distillation import OutputKDLoss, GroundTruthLoss

__all__ = ["OutputKDLoss", "GroundTruthLoss"]