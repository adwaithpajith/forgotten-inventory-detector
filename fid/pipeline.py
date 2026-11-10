"""One-call analysis: movements (+ optional SKU master) in, classified SKU table out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .classify import classify, summarize
from .config import Config
from .features import compute_features
from .schema import data_quality, ensure_sku_table, reconcile


@dataclass
class Analysis:
    movements: pd.DataFrame
    skus: pd.DataFrame
    features: pd.DataFrame
    result: pd.DataFrame
    summary: dict
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    as_of: pd.Timestamp = None
    cfg: Config = None


def analyze(
    movements: pd.DataFrame,
    skus: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> Analysis:
    """Run the full detector as of ``as_of`` (default: the last movement date)."""
    cfg = cfg or Config()
    if len(movements) == 0:
        raise ValueError("No movements to analyse.")
    as_of = pd.Timestamp(as_of or cfg.as_of or movements["date"].max()).normalize()
    skus = ensure_sku_table(skus, movements)
    movements, skus, notes = reconcile(movements, skus, as_of)
    feats = compute_features(movements, skus, as_of, cfg)
    result = classify(feats, skus, cfg)
    warnings = data_quality(movements, skus, as_of)
    return Analysis(
        movements=movements, skus=skus, features=feats, result=result,
        summary=summarize(result, cfg), notes=notes, warnings=warnings, as_of=as_of, cfg=cfg,
    )


def sku_timeline(movements: pd.DataFrame, sku: str, as_of: pd.Timestamp, days: int = 540) -> pd.DataFrame:
    """Daily picks, receipts and closing stock for one SKU (for the explorer chart)."""
    as_of = pd.Timestamp(as_of).normalize()
    mv = movements[(movements["sku"] == sku) & (movements["date"] <= as_of)]
    if len(mv) == 0:
        return pd.DataFrame(columns=["date", "picked", "received", "stock"])
    start = max(mv["date"].min(), as_of - pd.Timedelta(days=days - 1))
    idx = pd.date_range(start, as_of, freq="D")
    rec = mv[mv["kind"] == "receipt"].groupby("date")["qty"].sum()
    iss = mv[mv["kind"] == "issue"].groupby("date")["qty"].sum()
    full = pd.DataFrame(index=pd.date_range(mv["date"].min(), as_of, freq="D"))
    full["received"] = rec.reindex(full.index).fillna(0.0)
    full["picked"] = iss.reindex(full.index).fillna(0.0)
    full["stock"] = (full["received"] - full["picked"]).cumsum()
    out = full.reindex(idx)
    out.index.name = "date"
    return out.reset_index()
