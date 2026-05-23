from __future__ import annotations

import math


def compute_trigger_decisions(
    current_val_losses: dict[str, float],
    previous_val_losses: dict[str, float],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node_id in sorted(current_val_losses):
        current_loss = float(current_val_losses[node_id])
        previous_loss = float(previous_val_losses.get(node_id, current_loss))
        if not math.isfinite(previous_loss) or abs(previous_loss) <= 1e-12:
            relative_change = 0.0
        else:
            relative_change = (current_loss - previous_loss) / previous_loss
        rows.append(
            {
                "node_id": node_id,
                "previous_val_loss": previous_loss,
                "current_val_loss": current_loss,
                "relative_change_pct": relative_change * 100.0,
                "triggered": relative_change > threshold,
            }
        )
    return rows
