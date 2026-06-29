"""Distillation trainers (Mean Flow, Consistency) for V3 students."""

from finflow.distillation.consistency import (
    ConsistencyDistillConfig,
    consistency_distill_step,
    train_consistency_distill,
)
from finflow.distillation.mean_flow import (
    MeanFlowDistillConfig,
    mean_flow_loss,
    mean_flow_loss_components,
    train_mean_flow_distill,
)

__all__ = [
    "ConsistencyDistillConfig",
    "MeanFlowDistillConfig",
    "consistency_distill_step",
    "mean_flow_loss",
    "mean_flow_loss_components",
    "train_consistency_distill",
    "train_mean_flow_distill",
]
