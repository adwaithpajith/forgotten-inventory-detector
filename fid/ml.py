"""Optional ML second opinion: gradient boosting on the engine's own features.

The model is trained on **simulated** warehouses, where the planted labels are known.  It is a
secondary view: real warehouses can differ from the simulator, so the Streamlit app shows it
next to (never instead of) the statistical engine and highlights disagreement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

from .classify import classify
from .config import CLASSES, STATUS_OUT_OF_STOCK, Config
from .features import compute_features
from .schema import ensure_sku_table
from .simulate import make_warehouse

FEATURES: List[str] = [
    "idle_instock_days", "p_idle", "p_exp", "p_lognormal", "n_gaps", "mean_gap", "gap_cv", "adi", "cv2",
    "pick_days_30", "pick_days_90", "pick_days_180", "pick_days_365", "erosion_ratio", "season_corr",
    "season_top3", "ly_quiet_ratio", "ly_return_ratio", "stock_age_oldest", "stock_age_avg", "cover_days",
    "rate_per30", "receipt_events_90", "history_days", "days_since_pick", "life_used", "never_picked",
    "instock_days_win",
]


def _matrix(result: pd.DataFrame) -> pd.DataFrame:
    X = result[FEATURES].astype(float).replace([np.inf, -np.inf], 5000.0)
    return X


def build_training_set(seeds: Sequence[int], n_skus: int = 4000, cfg: Optional[Config] = None):
    cfg = cfg or Config()
    Xs, ys = [], []
    for s in seeds:
        wh = make_warehouse(n_skus=n_skus, seed=s)
        sk = ensure_sku_table(wh.skus, wh.movements)
        res = classify(compute_features(wh.movements, sk, wh.as_of, cfg), sk, cfg)
        t = wh.truth.reindex(res.index)
        keep = (res["status"] != STATUS_OUT_OF_STOCK) & t["truth_state"].isin(CLASSES)
        Xs.append(_matrix(res[keep]))
        ys.append(t.loc[keep, "truth_state"])
    return pd.concat(Xs), pd.concat(ys)


def train(seeds: Sequence[int] = (101, 102, 103), n_skus: int = 4000, cfg: Optional[Config] = None):
    X, y = build_training_set(seeds, n_skus, cfg)
    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                           l2_regularization=1.0, random_state=0)
    model.fit(X, y)
    return model


def evaluate(model, seeds: Sequence[int] = (201,), n_skus: int = 6000, cfg: Optional[Config] = None) -> Dict:
    """Score the model *and the engine* on a fresh simulated warehouse the model never saw."""
    cfg = cfg or Config()
    X, y = build_training_set(seeds, n_skus, cfg)
    pred = model.predict(X)
    # engine predictions on the same rows
    wh = make_warehouse(n_skus=n_skus, seed=seeds[0])
    sk = ensure_sku_table(wh.skus, wh.movements)
    res = classify(compute_features(wh.movements, sk, wh.as_of, cfg), sk, cfg)
    t = wh.truth.reindex(res.index)
    keep = (res["status"] != STATUS_OUT_OF_STOCK) & t["truth_state"].isin(CLASSES)
    eng = res.loc[keep, "status"]
    pi = permutation_importance(model, X.iloc[:3000], y.iloc[:3000], n_repeats=3, random_state=0, scoring="f1_macro")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    dead = {"forgotten", "obsolete"}
    return {
        "ml_accuracy": float(accuracy_score(y, pred)),
        "ml_macro_f1": float(f1_score(y, pred, average="macro")),
        "engine_accuracy": float((eng == y).mean()),
        "engine_macro_f1": float(f1_score(y, eng, average="macro")),
        "ml_dead_f1": float(f1_score(y.isin(dead), pd.Series(pred, index=y.index).isin(dead))),
        "engine_dead_f1": float(f1_score(y.isin(dead), eng.isin(dead))),
        "importance": imp,
        "n_test": int(len(y)),
    }


def save(model, path: str | Path) -> None:
    joblib.dump({"model": model, "features": FEATURES, "sklearn": sklearn.__version__}, path)


def load(path: str | Path):
    """Return the model, or ``None`` if the file is missing or was built with another sklearn."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        blob = joblib.load(p)
    except Exception:
        return None
    if blob.get("sklearn") != sklearn.__version__:
        return None
    return blob["model"]


def predict(model, result: pd.DataFrame) -> pd.DataFrame:
    """ML class and confidence for every SKU (only meaningful for judged, in-stock SKUs)."""
    X = _matrix(result)
    proba = model.predict_proba(X)
    out = pd.DataFrame({"ml_class": model.classes_[proba.argmax(1)], "ml_confidence": proba.max(1)}, index=result.index)
    return out
