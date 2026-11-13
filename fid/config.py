"""Configuration and shared constants for the Dead Stock Detector."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import pandas as pd

# Behaviour classes the engine can assign to an in-stock SKU.
CLASSES = [
    "active",
    "seasonal",
    "slow_moving",
    "temporarily_inactive",
    "forgotten",
    "obsolete",
]
DEAD_CLASSES = ("forgotten", "obsolete")

# Non-behavioural statuses (the SKU is not classified).
STATUS_OUT_OF_STOCK = "out_of_stock"
STATUS_INSUFFICIENT = "insufficient_history"

CLASS_LABELS = {
    "active": "Active",
    "seasonal": "Seasonal (expected quiet)",
    "slow_moving": "Slow-moving",
    "temporarily_inactive": "Temporarily inactive (watch)",
    "forgotten": "Forgotten",
    "obsolete": "Obsolete",
    STATUS_OUT_OF_STOCK: "Out of stock",
    STATUS_INSUFFICIENT: "Too new to judge",
}

CLASS_COLORS = {
    "active": "#2a9d8f",
    "seasonal": "#4c78a8",
    "slow_moving": "#e9c46a",
    "temporarily_inactive": "#f4a261",
    "forgotten": "#e63946",
    "obsolete": "#7b2d26",
    STATUS_OUT_OF_STOCK: "#adb5bd",
    STATUS_INSUFFICIENT: "#ced4da",
}


@dataclass(frozen=True)
class Config:
    """All tunable thresholds in one place.

    Every threshold is exposed in the Streamlit sidebar so a user can tune the
    engine to their own business (fast FMCG vs. spare parts, for example).
    """

    # --- analysis window -------------------------------------------------
    as_of: Optional[pd.Timestamp] = None  # default: last movement date
    lookback_days: int = 760  # daily matrix width (>= 2 years for seasonality)
    min_history_days: int = 90  # younger SKUs are "too new to judge"

    # --- classification --------------------------------------------------
    p_ok: float = 0.10  # idle spell more likely than this => normal for the SKU
    p_unusual: float = 0.05  # idle spell this unlikely or less => unusual
    watch_min_idle_days: int = 14  # in-stock idle days before a SKU can go on the watch list
    dead_min_idle_days: int = 60  # in-stock idle days before "dead" can be called
    never_moved_days: int = 120  # received but never picked for this long => forgotten
    active_pick_days_180: int = 12  # pick days in 180 days to count as "active"
    erosion_ratio_threshold: float = 0.6  # pre-stop rate ratio below this => obsolete
    life_used_obsolete: float = 0.8  # share of shelf life used => obsolete
    season_corr_min: float = 0.5  # year-on-year profile correlation for "seasonal"
    season_top3_min: float = 0.5  # share of demand in the 3 busiest 4-week blocks
    season_min_pick_days: int = 10  # minimum pick days over 2 years to assess seasonality

    # --- baseline / validation ------------------------------------------
    naive_days: int = 90  # the naive "no pick in N days" rule
    horizon_days: int = 90  # look-ahead for behavioural validation

    # --- finance ---------------------------------------------------------
    holding_cost_rate: float = 0.20  # annual holding cost as share of stock value
    currency: str = "₹"

    # --- engineering -----------------------------------------------------
    chunk_size: int = 3000  # SKUs per vectorised chunk (memory control)

    def replace(self, **kwargs) -> "Config":
        return replace(self, **kwargs)
