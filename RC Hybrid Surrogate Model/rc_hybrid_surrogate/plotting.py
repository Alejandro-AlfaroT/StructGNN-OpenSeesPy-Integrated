"""Dependency-free live training diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Sequence


class LossPlotter:
    """Write an auto-refreshing HTML/SVG loss chart after every epoch."""

    def __init__(self, output_path: str | Path, live: bool = False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.live = live

    @staticmethod
    def _polyline(values: Sequence[float], x_values: Sequence[float], scale_y) -> str:
        return " ".join(
            f"{x_value:.2f},{scale_y(value):.2f}"
            for x_value, value in zip(x_values, values)
        )

    def update(self, history: Sequence[dict]) -> None:
        if not history:
            return
        width, height = 920, 540
        left, right, top, bottom = 82, 28, 48, 70
        plot_width = width - left - right
        plot_height = height - top - bottom
        epochs = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train"]["loss"]) for row in history]
        validation_loss = [float(row["validation"]["loss"]) for row in history]
        all_loss = train_loss + validation_loss
        maximum = max(all_loss)
        minimum = min(0.0, min(all_loss))
        if maximum <= minimum:
            maximum = minimum + 1.0
        maximum += 0.08 * (maximum - minimum)

        if len(epochs) == 1:
            x_values = [left + plot_width / 2.0]
        else:
            x_values = [
                left + plot_width * index / (len(epochs) - 1)
                for index in range(len(epochs))
            ]

        def scale_y(value: float) -> float:
            return top + plot_height * (maximum - value) / (maximum - minimum)

        grid = []
        labels = []
        for index in range(6):
            value = minimum + (maximum - minimum) * index / 5.0
            y_value = scale_y(value)
            grid.append(
                f'<line x1="{left}" y1="{y_value:.2f}" x2="{width-right}" '
                f'y2="{y_value:.2f}" class="grid" />'
            )
            labels.append(
                f'<text x="{left-12}" y="{y_value+5:.2f}" text-anchor="end" '
                f'class="tick">{value:.4f}</text>'
            )

        train_points = self._polyline(train_loss, x_values, scale_y)
        validation_points = self._polyline(validation_loss, x_values, scale_y)
        refresh = '<meta http-equiv="refresh" content="3">' if self.live else ""
        latest = history[-1]
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {refresh}
  <title>RC Hybrid Surrogate Training</title>
  <style>
    body {{ margin: 0; background: #111827; color: #e5e7eb; font-family: Segoe UI, sans-serif; }}
    .wrap {{ max-width: 980px; margin: 24px auto; padding: 0 18px; }}
    .card {{ background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; }}
    svg {{ width: 100%; height: auto; }}
    .grid {{ stroke: #374151; stroke-width: 1; }}
    .axis {{ stroke: #9ca3af; stroke-width: 1.5; }}
    .tick {{ fill: #d1d5db; font-size: 13px; }}
    .label {{ fill: #e5e7eb; font-size: 15px; }}
    .title {{ fill: #f9fafb; font-size: 21px; font-weight: 600; }}
    .train {{ fill: none; stroke: #38bdf8; stroke-width: 3; }}
    .validation {{ fill: none; stroke: #fb7185; stroke-width: 3; }}
    .summary {{ margin: 10px 4px 0; color: #d1d5db; }}
  </style>
</head>
<body>
<div class="wrap"><div class="card">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training and validation loss curves">
  <text x="{width/2}" y="30" text-anchor="middle" class="title">Hybrid GNN-LSTM loss</text>
  {''.join(grid)}
  {''.join(labels)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis" />
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis" />
  <polyline points="{train_points}" class="train" />
  <polyline points="{validation_points}" class="validation" />
  <text x="{left}" y="{height-28}" class="tick">Epoch 1</text>
  <text x="{width-right}" y="{height-28}" text-anchor="end" class="tick">Epoch {epochs[-1]}</text>
  <text x="{width/2}" y="{height-10}" text-anchor="middle" class="label">Epoch</text>
  <text x="18" y="{height/2}" text-anchor="middle" class="label" transform="rotate(-90 18 {height/2})">Normalized Smooth L1 loss</text>
  <line x1="{width-245}" y1="25" x2="{width-210}" y2="25" class="train" />
  <text x="{width-200}" y="30" class="tick">Training</text>
  <line x1="{width-125}" y1="25" x2="{width-90}" y2="25" class="validation" />
  <text x="{width-80}" y="30" class="tick">Validation</text>
</svg>
<div class="summary">Latest epoch: {epochs[-1]} &nbsp;|&nbsp; Training loss: {latest['train']['loss']:.6f} &nbsp;|&nbsp; Validation loss: {latest['validation']['loss']:.6f}</div>
</div></div>
</body>
</html>
"""
        temporary = self.output_path.with_name(f".{self.output_path.name}.tmp")
        temporary.write_text(html, encoding="utf-8")
        delay = 0.05
        for attempt in range(10):
            try:
                os.replace(temporary, self.output_path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(delay)
                delay = min(delay * 2.0, 0.5)
