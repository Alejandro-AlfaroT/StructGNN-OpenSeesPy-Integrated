"""Engineering-facing metrics in physical response units."""

from __future__ import annotations

import math

import torch


class ResponseMetrics:
    def __init__(self):
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.value_count = 0
        self.peak_absolute_error = 0.0
        self.peak_value_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        prediction = prediction * target_std + target_mean
        target = target * target_std + target_mean
        valid = mask.unsqueeze(-1).expand_as(prediction)
        error = prediction - target
        self.squared_error += float(torch.square(error)[valid].sum().item())
        self.absolute_error += float(torch.abs(error)[valid].sum().item())
        self.value_count += int(valid.sum().item())
        for batch_index in range(prediction.shape[0]):
            length = int(mask[batch_index].sum().item())
            predicted_peak = prediction[batch_index, :length].abs().amax(dim=0)
            target_peak = target[batch_index, :length].abs().amax(dim=0)
            self.peak_absolute_error += float(
                torch.abs(predicted_peak - target_peak).sum().item()
            )
            self.peak_value_count += int(predicted_peak.numel())

    def compute(self) -> dict[str, float]:
        if self.value_count == 0:
            return {"rmse_in": math.nan, "mae_in": math.nan, "peak_mae_in": math.nan}
        return {
            "rmse_in": math.sqrt(self.squared_error / self.value_count),
            "mae_in": self.absolute_error / self.value_count,
            "peak_mae_in": self.peak_absolute_error / self.peak_value_count,
        }
