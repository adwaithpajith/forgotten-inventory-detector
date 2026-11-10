"""Turn per-SKU features into a behaviour class, a risk score, money figures and an action.

Decision order (first match wins), applied to SKUs that are in stock and old enough to judge:

1. **obsolete**    hard evidence the product has no future: stock that cannot be sold before
                   it expires, a successor SKU that has taken over, or demand that was
                   visibly fading before it stopped.
2. **seasonal**    the SKU has a strong yearly rhythm *and* it was equally quiet at this
                   point last year, so the silence is expected.
3. **forgotten**   a silence that is both long (``dead_min_idle_days`` in-stock days) and
                   statistically unusual for this SKU's own history, or stock that arrived
                   long ago and was never picked at all.
4. **temporarily_inactive**  an unusual silence that has not yet lasted long enough to call.
5. **slow_moving** picks are rare but the current gap is normal for that rhythm.
6. **active**      everything else.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    CLASS_LABELS,
    STATUS_INSUFFICIENT,
    STATUS_OUT_OF_STOCK,
    Config,
)
from .schema import ensure_sku_table

ACTIONS: Dict[str, str] = {
    "forgotten": "Investigate now: confirm the stock physically exists and is correctly located, "
                 "then promote, transfer, return to supplier or liquidate. Stop re-ordering.",
    "obsolete": "Liquidate, return to supplier or write down. Block further purchasing and "
                "redirect demand to the successor if there is one.",
    "temporarily_inactive": "Watch: re-check in 30 days and hold new purchase orders until demand returns.",
    "slow_moving": "Lower the reorder quantity (or switch to order-on-demand) so stock cover stops growing.",
    "seasonal": "Hold. Quiet is expected; plan replenishment ahead of the next season.",
    "active": "No action.",
    STATUS_OUT_OF_STOCK: "No stock on hand.",
    STATUS_INSUFFICIENT: "Too little history to judge; re-run in a few weeks.",
}


def _fmt_days(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{int(round(x))}"


def _reason(row: pd.Series, cfg: Config) -> str:
    cls = row["status"]
    idle = row["idle_instock_days"]
    if cls == "forgotten":
        if row["never_picked"]:
            return (f"Received and in stock for {_fmt_days(row['instock_days_win'])} days without a single pick.")
        return (f"No pick for {_fmt_days(idle)} in-stock days; a gap this long has only a "
                f"{row['p_idle']:.1%} chance given this SKU's own pick rhythm.")
    if cls == "obsolete":
        parts: List[str] = []
        if row["flag_expiry"]:
            parts.append("stock cannot be sold before its shelf life ends at the current pace")
        if row["flag_successor"]:
            parts.append("its successor SKU is now selling")
        if row["flag_eroded"]:
            parts.append(f"demand had already fallen to {row['erosion_ratio']:.0%} of its earlier rate before stopping")
        return "Obsolescence signals: " + "; ".join(parts) + "."
    if cls == "temporarily_inactive":
        return (f"Idle {_fmt_days(idle)} in-stock days, unusual for this SKU "
                f"(p={row['p_idle']:.2f}) but not yet long enough to call it dead.")
    if cls == "seasonal":
        return ("Strong yearly rhythm and equally quiet at this point last year; "
                f"idle {_fmt_days(idle)} in-stock days is expected.")
    if cls == "slow_moving":
        return (f"{int(row['pick_days_180'])} pick days in the last 180; the current gap is normal for its rhythm.")
    if cls == "active":
        return f"{int(row['pick_days_180'])} pick days in the last 180."
    if cls == STATUS_OUT_OF_STOCK:
        return "No stock on hand."
    return f"Only {int(row['history_days'])} days of history (need {cfg.min_history_days})."


def classify(features: pd.DataFrame, skus: Optional[pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Return ``features`` plus ``status``, flags, risk score, finance columns, reason and action."""
    f = features.copy()
    sk = ensure_sku_table(skus, pd.DataFrame({"sku": f.index.astype(str)})).reindex(f.index)

    on_hand = f["on_hand"].to_numpy()
    idle = f["idle_instock_days"].to_numpy()
    p = f["p_idle"].to_numpy()
    never = f["never_picked"].to_numpy()
    n = len(f)

    in_stock = on_hand > 0.5
    too_new = f["history_days"].to_numpy() < cfg.min_history_days

    # ---- seasonality: strong yearly rhythm and equally quiet this time last year
    sa = f["season_assessable"].to_numpy()
    with np.errstate(invalid="ignore"):
        season_pattern = (
            sa
            & (f["season_corr"].to_numpy() >= cfg.season_corr_min)
            & (f["season_top3"].to_numpy() >= cfg.season_top3_min)
        )
        quiet_last_year = f["ly_quiet_ratio"].to_numpy() <= 0.5
    seasonal_explained = season_pattern & quiet_last_year

    # ---- obsolescence signals
    with np.errstate(invalid="ignore"):
        life_used = f["life_used"].to_numpy()
        cover = f["cover_days"].to_numpy()
        days_to_exp = np.maximum(f["days_to_expiry"].to_numpy(), 0.0)
        weak_demand = (p <= cfg.p_ok) | (f["pick_days_180"].to_numpy() < cfg.active_pick_days_180)
        flag_expiry = (life_used >= cfg.life_used_obsolete) & (cover > days_to_exp) & weak_demand
        flag_expiry = flag_expiry | ((f["expired_qty"].to_numpy() > 0) & weak_demand)

        succ = sk["successor_sku"].to_numpy(dtype=object)
        succ_rate = f["pick_days_90"].reindex(pd.Index(succ[pd.notna(succ)])).to_numpy() if pd.notna(succ).any() else np.array([])
        succ_active = np.zeros(n, dtype=bool)
        if succ_rate.size:
            succ_active[pd.notna(succ)] = np.nan_to_num(succ_rate, nan=0.0) >= 3
        flag_successor = succ_active & (p <= cfg.p_ok) & (idle >= cfg.watch_min_idle_days)

        long_silent = (idle >= cfg.dead_min_idle_days) & (p <= cfg.p_unusual) & ~seasonal_explained
        flag_eroded = long_silent & (f["erosion_ratio"].to_numpy() < cfg.erosion_ratio_threshold)

    obsolete = flag_expiry | flag_successor | flag_eroded

    never_long = never & (f["instock_days_win"].to_numpy() >= cfg.never_moved_days)
    forgotten = (never_long | long_silent) & ~obsolete

    # currently in a trough: the last 90 days sold at under half the rate of the last year
    with np.errstate(invalid="ignore", divide="ignore"):
        trough = (f["pick_days_90"].to_numpy() / 90.0) < 0.5 * (f["pick_days_365"].to_numpy() / 365.0)
    seasonal = (
        seasonal_explained & trough
        & ((idle >= cfg.watch_min_idle_days) | (f["pick_days_30"].to_numpy() <= 1))
        & ~obsolete
    )
    temporarily = (
        (idle >= cfg.watch_min_idle_days) & (p <= cfg.p_ok) & ~seasonal_explained & ~obsolete & ~forgotten
    )
    slow = f["pick_days_180"].to_numpy() < cfg.active_pick_days_180

    status = np.select(
        [~in_stock, too_new, obsolete, seasonal, forgotten, temporarily, slow],
        [STATUS_OUT_OF_STOCK, STATUS_INSUFFICIENT, "obsolete", "seasonal", "forgotten",
         "temporarily_inactive", "slow_moving"],
        default="active",
    )
    f["status"] = status
    f["flag_expiry"] = flag_expiry & in_stock
    f["flag_successor"] = flag_successor & in_stock
    f["flag_eroded"] = flag_eroded & in_stock
    f["dead_flag"] = np.isin(status, ["forgotten", "obsolete"])

    # ---- master data and money
    cost = pd.to_numeric(sk["unit_cost"], errors="coerce").fillna(1.0).to_numpy()
    vol = pd.to_numeric(sk["volume_per_unit"], errors="coerce").to_numpy()
    f["description"] = sk["description"].to_numpy()
    f["category"] = sk["category"].to_numpy()
    f["location"] = sk["location"].to_numpy()
    f["unit_cost"] = cost
    f["capital_tied"] = np.where(in_stock, on_hand, 0.0) * cost
    f["holding_cost_per_year"] = f["capital_tied"] * cfg.holding_cost_rate
    f["space_m3"] = np.where(in_stock & np.isfinite(vol), on_hand * vol, np.nan if not np.isfinite(vol).any() else 0.0)
    f["expiry_exposure"] = np.where(in_stock, f["expired_qty"].to_numpy() * cost, 0.0)

    # ---- risk score 0-100 (transparent weighted sum)
    evidence = np.where(idle > 0, 1.0 - np.clip(p, 0, 1), 0.0)
    evidence = np.where(seasonal_explained, evidence * 0.3, evidence)
    evidence = np.where(never_long, 1.0, evidence)
    with np.errstate(invalid="ignore"):
        age_factor = np.nan_to_num(np.clip(f["stock_age_oldest"].to_numpy() / 365.0, 0, 1), nan=0.0)
        cover_factor = np.where(np.isfinite(cover), np.clip(cover / 730.0, 0, 1), 1.0)
        expiry_factor = np.where(
            f["expired_qty"].to_numpy() > 0, 1.0, np.nan_to_num(np.clip(life_used, 0, 1), nan=0.0)
        )
    risk = 100.0 * (0.45 * evidence + 0.25 * age_factor + 0.15 * cover_factor + 0.15 * expiry_factor)
    risk = np.where(np.isin(status, ["forgotten", "obsolete"]), np.maximum(risk, 60.0), risk)
    risk = np.where(np.isin(status, [STATUS_OUT_OF_STOCK, STATUS_INSUFFICIENT]), 0.0, risk)
    f["risk_score"] = np.clip(risk, 0, 100).round(1)
    f["value_at_risk"] = f["capital_tied"] * f["risk_score"] / 100.0

    # ---- replenishment conflict: still buying something nobody is picking
    f["replenishment_conflict"] = (
        np.isin(status, ["forgotten", "obsolete", "temporarily_inactive"]) & (f["receipts_90"].to_numpy() > 0)
    )

    f["status_label"] = f["status"].map(CLASS_LABELS)
    f["action"] = f["status"].map(ACTIONS)
    f.loc[f["replenishment_conflict"], "action"] = (
        "STOP PURCHASING: this item is still being replenished while it sits idle. " + f["action"]
    )
    f["reason"] = f.apply(lambda r: _reason(r, cfg), axis=1)
    return f


def summarize(result: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """Headline numbers for the dashboard."""
    judged = result[~result["status"].isin([STATUS_OUT_OF_STOCK, STATUS_INSUFFICIENT])]
    dead = result[result["dead_flag"]]
    total_cap = float(result["capital_tied"].sum())
    return {
        "n_skus": int(len(result)),
        "n_in_stock": int((result["status"] != STATUS_OUT_OF_STOCK).sum()),
        "n_judged": int(len(judged)),
        "n_forgotten": int((result["status"] == "forgotten").sum()),
        "n_obsolete": int((result["status"] == "obsolete").sum()),
        "n_watch": int((result["status"] == "temporarily_inactive").sum()),
        "n_conflict": int(result["replenishment_conflict"].sum()),
        "capital_total": total_cap,
        "capital_dead": float(dead["capital_tied"].sum()),
        "capital_dead_share": float(dead["capital_tied"].sum() / total_cap) if total_cap > 0 else 0.0,
        "holding_cost_dead": float(dead["holding_cost_per_year"].sum()),
        "space_dead_m3": float(dead["space_m3"].sum(skipna=True)),
        "expiry_exposure": float(result["expiry_exposure"].sum()),
    }
