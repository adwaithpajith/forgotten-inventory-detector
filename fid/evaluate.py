"""Evaluation: planted-truth scoring, the naive baseline, detection lead time and a label-free backtest.

Three complementary views:

* :func:`planted_eval`  scores the engine against the *planted* ground truth of a simulated
  warehouse (confusion matrix, per-class precision/recall, dead-vs-alive F1) next to the
  naive "no pick in N days" rule, and checks flagged SKUs against what happened in the
  held-out future.
* :func:`lead_time_eval`  how many days after a planted stop each method first flags the SKU.
* :func:`behavioural_backtest`  needs **no labels**: classify as of ``horizon`` days ago and
  see whether the SKUs flagged dead really stayed dormant afterwards.  This is what the
  Streamlit app runs on a user's own data.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .classify import classify
from .config import CLASSES, DEAD_CLASSES, STATUS_INSUFFICIENT, STATUS_OUT_OF_STOCK, Config
from .features import compute_features
from .pipeline import analyze
from .schema import ensure_sku_table

NOT_JUDGED = (STATUS_OUT_OF_STOCK, STATUS_INSUFFICIENT)


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def naive_flag(result: pd.DataFrame, naive_days: int) -> np.ndarray:
    """The rule most warehouses use today: in stock and no pick for N calendar days."""
    return ((result["on_hand"] > 0.5) & (result["days_since_pick"] >= naive_days)).to_numpy()


def _future_pick_days(future: pd.DataFrame) -> pd.Series:
    iss = future[future["kind"] == "issue"]
    return iss.groupby("sku")["date"].nunique()


# --------------------------------------------------------------------------- planted truth
def planted_eval(wh, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    a = analyze(wh.movements, wh.skus, cfg, as_of=wh.as_of)
    r = a.result
    t = wh.truth.reindex(r.index)
    in_stock = (r["status"] != STATUS_OUT_OF_STOCK).to_numpy()
    judged_truth = t["truth_state"].ne("insufficient_history").to_numpy()
    keep = in_stock & judged_truth
    rr, tt = r[keep], t[keep]

    labels = [c for c in CLASSES if c in set(tt["truth_state"])]
    conf = pd.crosstab(tt["truth_state"], rr["status"]).reindex(index=labels, columns=CLASSES + [STATUS_INSUFFICIENT], fill_value=0)
    per_class = []
    for c in labels:
        tp = int(((tt["truth_state"] == c) & (rr["status"] == c)).sum())
        sup = int((tt["truth_state"] == c).sum())
        pred = int((rr["status"] == c).sum())
        pr = tp / pred if pred else 0.0
        rc = tp / sup if sup else 0.0
        per_class.append({"class": c, "support": sup, "precision": pr, "recall": rc,
                          "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0})
    per_class = pd.DataFrame(per_class).set_index("class")
    lab = tt["truth_state"].isin(CLASSES).to_numpy()
    accuracy = float((rr["status"][lab] == tt["truth_state"][lab]).mean()) if lab.any() else float("nan")

    eng_dead = rr["dead_flag"].to_numpy()
    nv = naive_flag(rr, cfg.naive_days)
    nv_same = naive_flag(rr, cfg.dead_min_idle_days)  # naive rule at the engine's own idle threshold
    truth_dead = tt["truth_dead"].to_numpy()
    dead = {"engine": prf(truth_dead[lab], eng_dead[lab]), "naive": prf(truth_dead[lab], nv[lab]),
            "naive_same_idle": prf(truth_dead[lab], nv_same[lab])}

    fp = _future_pick_days(wh.future).reindex(rr.index).fillna(0)
    def outcome(mask):
        m = np.asarray(mask, bool)
        return {"flagged": int(m.sum()),
                "picked_again_in_future": int((fp[m] > 0).sum()),
                "share_picked_again": float((fp[m] > 0).mean()) if m.any() else 0.0}
    outcomes = {"engine_dead": outcome(eng_dead), "naive": outcome(nv)}
    seasonal = (tt["truth_state"] == "seasonal").to_numpy()
    seasonal_false_alarms = {"engine": int((eng_dead & seasonal).sum()), "naive": int((nv & seasonal).sum()),
                             "n_seasonal": int(seasonal.sum())}
    if (~lab).any():  # real-demand warehouses have unlabelled SKUs
        outcomes["engine_dead_unlabelled"] = outcome(eng_dead & ~lab)
        outcomes["naive_unlabelled"] = outcome(nv & ~lab)
        outcomes["naive_same_idle_unlabelled"] = outcome(nv_same & ~lab)
        outcomes["n_unlabelled"] = int((~lab).sum())
    return {
        "analysis": a, "confusion": conf, "per_class": per_class, "accuracy": accuracy, "dead": dead,
        "outcomes": outcomes, "seasonal_false_alarms": seasonal_false_alarms, "n_evaluated": int(keep.sum()),
        "n_out_of_stock": int((~in_stock).sum()),
    }


def threshold_sweep(wh, cfg: Optional[Config] = None, p_grid: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
                    idle_grid: Iterable[int] = (30, 45, 60, 90)) -> pd.DataFrame:
    """Dead-vs-alive precision/recall as the two main thresholds move (features computed once)."""
    cfg = cfg or Config()
    sk = ensure_sku_table(wh.skus, wh.movements)
    feats = compute_features(wh.movements, sk, wh.as_of, cfg)
    t = wh.truth.reindex(feats.index)
    keep = ((feats["on_hand"] > 0.5) & t["truth_state"].isin(CLASSES)).to_numpy()
    rows = []
    for pu in p_grid:
        for idl in idle_grid:
            c = cfg.replace(p_unusual=pu, dead_min_idle_days=idl)
            res = classify(feats, sk, c)
            m = prf(t["truth_dead"].to_numpy()[keep], res["dead_flag"].to_numpy()[keep])
            rows.append({"p_unusual": pu, "dead_min_idle_days": idl, **m})
    return pd.DataFrame(rows)


def unlabelled_tradeoff(wh, cfg: Optional[Config] = None, idle_grid: Iterable[int] = (45, 60, 90, 120, 180)) -> Optional[pd.DataFrame]:
    """Real-demand warehouses only: false alarms on untouched real SKUs against recall of planted dead SKUs.

    Unlabelled real SKUs keep their real demand, so (almost all of) them are alive: every dead flag on one is
    a likely false alarm.  ``*_per_100`` is flags per 100 unlabelled in-stock SKUs; ``*_picked_again_pct``
    is the share of those flags picked again in the held-out window.  Planted recall shows what each
    threshold buys.  Features are computed once.
    """
    cfg = cfg or Config()
    sk = ensure_sku_table(wh.skus, wh.movements)
    feats = compute_features(wh.movements, sk, wh.as_of, cfg)
    t = wh.truth.reindex(feats.index)
    in_stock = (feats["on_hand"] > 0.5).to_numpy()
    unl = (t["truth_state"] == "unlabelled").to_numpy() & in_stock
    if not unl.any():
        return None
    planted_dead = t["truth_dead"].to_numpy().astype(bool) & in_stock
    fp = _future_pick_days(wh.future).reindex(feats.index).fillna(0).to_numpy()
    rows = []
    for idl in idle_grid:
        res = classify(feats, sk, cfg.replace(dead_min_idle_days=idl))
        e, n = res["dead_flag"].to_numpy(), naive_flag(res, idl)
        def stats(flag):
            m = flag & unl
            return int(m.sum()), 100.0 * m.sum() / unl.sum(), (100.0 * (fp[m] > 0).mean() if m.any() else float("nan"))
        es, ns = stats(e), stats(n)
        rows.append({
            "idle_days": idl, "engine_flags": es[0], "engine_per_100": es[1], "engine_picked_again_pct": es[2],
            "naive_flags": ns[0], "naive_per_100": ns[1], "naive_picked_again_pct": ns[2],
            "engine_planted_recall": float((e & planted_dead).sum() / max(planted_dead.sum(), 1)),
            "naive_planted_recall": float((n & planted_dead).sum() / max(planted_dead.sum(), 1)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- lead time
def lead_time_eval(wh, cfg: Optional[Config] = None, step: int = 15, span: int = 330) -> Dict:
    """Days between a planted abrupt stop and the first flag, engine vs naive rule."""
    cfg = cfg or Config()
    t = wh.truth
    sub = t[(t["truth_state"] == "forgotten") & ~t["never_picked"] & t["event_date"].notna()]
    if sub.empty:
        return {"n": 0}
    mv = wh.movements[wh.movements["sku"].isin(sub.index)]
    sk = ensure_sku_table(wh.skus.reindex(sub.index), mv)
    dates = list(pd.date_range(wh.as_of - pd.Timedelta(days=span), wh.as_of, freq=f"{step}D"))
    if dates[-1] != wh.as_of:
        dates.append(wh.as_of)
    first_eng: Dict[str, pd.Timestamp] = {}
    first_nv: Dict[str, pd.Timestamp] = {}
    first_nv_same: Dict[str, pd.Timestamp] = {}
    ev = sub["event_date"]
    for d in dates:
        f = compute_features(mv, sk, d, cfg)
        res = classify(f, sk, cfg)
        for first, flags in (
            (first_eng, res["dead_flag"].to_numpy()),
            (first_nv, naive_flag(res, cfg.naive_days)),
            (first_nv_same, naive_flag(res, cfg.dead_min_idle_days)),
        ):
            for sku in res.index[flags]:
                if d >= ev[sku] and sku not in first:
                    first[sku] = d
    def delays(first):
        return pd.Series({k: (v - ev[k]).days for k, v in first.items()}, dtype=float)
    de, dn, dns = delays(first_eng), delays(first_nv), delays(first_nv_same)
    def summ(d):
        return {"detected": int(len(d)), "share": float(len(d) / len(sub)),
                "median_days": float(d.median()) if len(d) else float("nan"),
                "p90_days": float(d.quantile(0.9)) if len(d) else float("nan")}
    return {"n": int(len(sub)), "engine": summ(de), "naive": summ(dn), "naive_same_idle": summ(dns),
            "engine_delays": de, "naive_delays": dn, "naive_same_idle_delays": dns, "step_days": step}


# --------------------------------------------------------------------------- label-free backtest
def behavioural_backtest(
    movements: pd.DataFrame, skus: Optional[pd.DataFrame], cfg: Optional[Config] = None,
    as_of: Optional[pd.Timestamp] = None, horizon: Optional[int] = None, exclude: Optional[Iterable[str]] = None,
) -> Optional[Dict]:
    """Classify as of ``horizon`` days ago; did the flagged SKUs stay dormant since?

    Needs no labels, only enough history.  Returns ``None`` when there is not enough.
    ``exclude`` drops SKUs from scoring: pass the planted SKUs of a real-demand warehouse, whose
    demand was edited, so that only untouched real behaviour is judged.
    """
    cfg = cfg or Config()
    horizon = int(horizon or cfg.horizon_days)
    as_of = pd.Timestamp(as_of or movements["date"].max()).normalize()
    t0 = as_of - pd.Timedelta(days=horizon)
    start = movements["date"].min()
    if (t0 - start).days < cfg.min_history_days + 120:
        return None
    sk = ensure_sku_table(skus, movements)
    f0 = compute_features(movements, sk, t0, cfg)
    r0 = classify(f0, sk, cfg)
    win = movements[(movements["date"] > t0) & (movements["date"] <= as_of) & (movements["kind"] == "issue")]
    picks = win.groupby("sku")["date"].nunique().reindex(r0.index).fillna(0)
    judged = ~r0["status"].isin(NOT_JUDGED)
    if exclude is not None:
        judged &= ~r0.index.isin(list(exclude))
    base = r0[judged]
    pk = picks[judged]

    def row(name: str, mask: np.ndarray) -> Dict:
        m = np.asarray(mask, bool)
        n = int(m.sum())
        return {"group": name, "n_skus": n,
                "stayed_dormant_pct": float((pk[m] == 0).mean() * 100) if n else float("nan"),
                "picked_again_pct": float((pk[m] > 0).mean() * 100) if n else float("nan"),
                "capital_tied": float(base["capital_tied"][m].sum())}

    nv = naive_flag(base, cfg.naive_days)
    rows = [
        row("Engine: forgotten + obsolete", base["dead_flag"].to_numpy()),
        row("Engine: forgotten", (base["status"] == "forgotten").to_numpy()),
        row("Engine: obsolete", (base["status"] == "obsolete").to_numpy()),
        row("Engine: watch", (base["status"] == "temporarily_inactive").to_numpy()),
        row("Engine: seasonal", (base["status"] == "seasonal").to_numpy()),
        row("Engine: slow-moving", (base["status"] == "slow_moving").to_numpy()),
        row("Engine: active", (base["status"] == "active").to_numpy()),
        row(f"Naive rule: no pick in {cfg.naive_days} days", nv),
        row("All judged SKUs", np.ones(len(base), bool)),
    ]
    table = pd.DataFrame(rows)
    eng = base["dead_flag"].to_numpy()
    only_naive = nv & ~eng
    return {
        "t0": t0, "as_of": as_of, "horizon": horizon, "table": table,
        "naive_false_alarms_avoided": int((only_naive & (pk.to_numpy() > 0)).sum()),
        "naive_only_flagged": int(only_naive.sum()),
        "engine_only_flagged": int((eng & ~nv).sum()),
    }