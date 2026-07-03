"""Generation policies for MIMIC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationPolicy:
    method: str = "smote"
    neighbour_mode: str = "normal"
    n_neighbors: int = 5
    lambda_range: tuple[float, float] = (0.0, 1.0)
    class_conditioned: bool = False
    cluster_conditioned: bool = False

    def validate(self):
        if self.method not in {"smote", "displacement"}:
            raise ValueError("method must be 'smote' or 'displacement'")
        if self.neighbour_mode not in {"normal", "mutual"}:
            raise ValueError("neighbour_mode must be 'normal' or 'mutual'")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        lo, hi = self.lambda_range
        if hi < lo:
            raise ValueError("lambda_range must be ordered as (low, high)")
        return self

