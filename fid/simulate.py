"""Warehouse simulator with planted ground truth.

No public dataset records which stock was *forgotten*, so we plant it.  The
simulator has two halves that can be used together or apart:

1. **Demand** (units demanded per SKU per day).  Either generated synthetically from
   six behaviour archetypes, or taken from *real* sales history (see ``realdata.py``)
   with forgotten / obsolete / paused behaviour planted on a random subset.
2. **Supply side** (:func:`simulate_ledger`).  A reorder-point replenishment policy
   with lead times and pack sizes turns demand into the receipts and picks a real
   warehouse management system would record.  Stock-outs lose sales, an exponentially
   weighted demand forecast drives re-ordering (so it lags when demand dies, exactly
   as real forecasts do), and some dead SKUs keep receiving stock from standing
   purchase orders.

The archetypes (and the state each is in on the ``as_of`` date):

========================  =====================================================
``active``                steady demand, from fast movers to intermittent ones
``seasonal``              demand concentrated in a few weeks of the year
``slow_moving``           a few picks a year, regular
``temporarily_inactive``  normally active, paused for a spell that ends after as_of
``forgotten``             steady demand that stops abruptly and never returns, or
                          stock that was never picked at all
``obsolete``              demand that fades away over months (some with a successor
                          SKU, some with a shelf life that runs out)
``new``                   launched recently (too new to judge)
========================  =====================================================

A held-out *future* window after ``as_of`` is simulated too, so the engine's flags
can be checked against what really happened next.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ARCHETYPES = ["active", "seasonal", "slow_moving", "temporarily_inactive", "forgotten", "obsolete", "new"]
DEFAULT_MIX = {
    "active": 0.42, "seasonal": 0.15, "slow_moving": 0.15, "temporarily_inactive": 0.08,
    "forgotten": 0.10, "obsolete": 0.07, "new": 0.03,
}
CATEGORIES = ["Fasteners", "Electrical", "Packaging", "Chemicals", "Seasonal goods", "Tools",
              "Consumables", "Spare parts"]


@dataclass
class Warehouse:
    movements: pd.DataFrame  # standard ledger up to and including ``as_of``
    future: pd.DataFrame     # ledger after ``as_of`` (held out, for validation only)
    skus: pd.DataFrame       # SKU master (indexed by sku)
    truth: pd.DataFrame      # planted ground truth (indexed by sku)
    as_of: pd.Timestamp
    source: str = "synthetic"


def _logu(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


# --------------------------------------------------------------------------- demand
def generate_demand(
    n_skus: int, n_hist: int, n_future: int, start: pd.Timestamp, rng: np.random.Generator,
    mix: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Synthetic demand in units per day, shape (T, n_skus), plus a truth/parameter table."""
    mix = mix or DEFAULT_MIX
    T = n_hist + n_future
    asi = n_hist - 1  # index of the as_of day
    names = list(mix.keys())
    probs = np.array([mix[k] for k in names], dtype=float)
    arche = rng.choice(names, size=n_skus, p=probs / probs.sum())

    days = pd.date_range(start, periods=T, freq="D")
    doy = days.dayofyear.to_numpy()
    t = np.arange(T)[:, None].astype(np.float32)
    lam = np.zeros((T, n_skus), dtype=np.float32)

    truth = pd.DataFrame({"archetype": arche})
    truth["event_idx"] = -1
    truth["launch_idx"] = 0
    truth["base_rate"] = np.nan
    truth["never_picked"] = False
    truth["peak_doy"] = np.nan

    for name in names:
        ix = np.flatnonzero(arche == name)
        k = ix.size
        if k == 0:
            continue
        if name == "active":
            # "active" means at least ~2.4 pick events a month; slower SKUs are 'slow_moving'.
            # The boundary is a business definition, so the archetypes are kept apart to
            # leave the interesting confusions (dead vs paused vs slow vs seasonal) intact.
            rate = _logu(rng, 0.08, 3.0, k)
            lam[:, ix] = rate[None, :]
            truth.loc[ix, "base_rate"] = rate
        elif name == "seasonal":
            peak = _logu(rng, 0.3, 3.0, k)
            pdoy = rng.uniform(1, 365, k)
            w = rng.uniform(22, 40, k)
            dd = np.abs(doy[:, None] - pdoy[None, :])
            dd = np.minimum(dd, 365 - dd)
            lam[:, ix] = (peak[None, :] * (np.exp(-0.5 * (dd / w[None, :]) ** 2) + 0.02)).astype(np.float32)
            truth.loc[ix, "base_rate"] = peak
            truth.loc[ix, "peak_doy"] = pdoy
        elif name == "slow_moving":
            rate = _logu(rng, 0.006, 0.035, k)
            lam[:, ix] = rate[None, :]
            truth.loc[ix, "base_rate"] = rate
        elif name == "temporarily_inactive":
            rate = _logu(rng, 0.2, 1.0, k)
            p0 = (asi - rng.integers(10, 81, k))
            p1 = (asi + rng.integers(15, 71, k))
            inside = (t >= p0[None, :]) & (t < p1[None, :])
            lam[:, ix] = np.where(inside, 0.0, rate[None, :]).astype(np.float32)
            truth.loc[ix, "base_rate"] = rate
            truth.loc[ix, "event_idx"] = p0
        elif name == "forgotten":
            rate = _logu(rng, 0.05, 1.5, k)
            e = asi - rng.integers(75, 301, k)
            never = rng.random(k) < 0.25
            lam[:, ix] = np.where(t > e[None, :], 0.0, rate[None, :]).astype(np.float32)
            lam[:, ix[never]] = 0.0
            e = np.where(never, asi - rng.integers(130, 331, k), e)  # for never-picked: the receipt day
            truth.loc[ix, "base_rate"] = rate
            truth.loc[ix, "event_idx"] = e
            truth.loc[ix[never], "never_picked"] = True
            truth.loc[ix[never], "launch_idx"] = e[never]
        elif name == "obsolete":
            rate = _logu(rng, 0.1, 1.5, k)
            dur = rng.integers(180, 301, k)
            e = asi - rng.integers(75, 261, k)
            fade = np.clip((e[None, :] - t) / dur[None, :], 0.0, 1.0)
            lam[:, ix] = (rate[None, :] * fade).astype(np.float32)
            truth.loc[ix, "base_rate"] = rate
            truth.loc[ix, "event_idx"] = e
        elif name == "new":
            rate = _logu(rng, 0.1, 1.5, k)
            launch = asi - rng.integers(20, 76, k)
            lam[:, ix] = np.where(t >= launch[None, :], rate[None, :], 0.0).astype(np.float32)
            truth.loc[ix, "base_rate"] = rate
            truth.loc[ix, "launch_idx"] = launch

    # State on the as_of day: a seasonal item counts as 'seasonal' only while it is in its
    # trough (pick rate below the 0.08/day 'active' line); while it is still selling at an
    # active rate it is simply 'active'.
    state = pd.Series(arche).copy()
    seas = np.flatnonzero(arche == "seasonal")
    if seas.size:
        state.iloc[seas] = np.where(lam[asi, seas] < 0.08, "seasonal", "active")
    state = state.replace({"new": "insufficient_history"})
    truth["truth_state"] = state.to_numpy()

    # units per pick event, then draw counts and units
    msize = rng.choice([1, 2, 3, 5, 8, 12], size=n_skus, p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05])
    truth["pick_size"] = msize
    counts = rng.poisson(lam)
    units = counts + rng.poisson(counts * (msize[None, :] - 1))
    return units.astype(np.float32), truth


# --------------------------------------------------------------------------- supply side
def simulate_ledger(
    units: np.ndarray,
    launch: np.ndarray,
    rng: np.random.Generator,
    init_qty: Optional[np.ndarray] = None,
    extra: Optional[np.ndarray] = None,
    alpha: float = 1.0 / 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reorder-point replenishment on a (T, n) demand matrix.

    Returns receipt and issue event arrays ``(t, sku_index, qty)`` each as three arrays.
    ``extra`` is an optional (T, n) matrix of unplanned receipts (standing purchase orders).
    """
    T, n = units.shape
    lead = rng.integers(5, 22, n)
    pack = rng.choice([1, 6, 12, 24], size=n, p=[0.4, 0.3, 0.2, 0.1]).astype(float)
    launch = np.asarray(launch, dtype=int)
    arange = np.arange(n)

    lam_hat = np.zeros(n)
    for i0 in range(0, n, 2000):  # initial demand estimate from the first 60 days after launch
        for i in range(i0, min(n, i0 + 2000)):
            lam_hat[i] = units[launch[i]:launch[i] + 60, i].mean() if launch[i] < T else 0.0
    if init_qty is None:
        init_qty = np.ceil(lam_hat * (lead + 44) / pack) * pack
        init_qty = np.maximum(init_qty, pack)
    free = np.zeros((T, n), dtype=np.float32) if extra is None else extra.astype(np.float32).copy()
    free[launch, arange] += init_qty.astype(np.float32)
    pipe = np.zeros((T, n), dtype=np.float32)

    on_hand = np.zeros(n)
    on_order = np.zeros(n)
    rt, ri, rq, it, ii, iq = [], [], [], [], [], []
    for t in range(T):
        arr_p = pipe[t]
        arr = arr_p + free[t]
        on_hand += arr
        on_order -= arr_p
        nz = np.flatnonzero(arr > 0)
        if nz.size:
            rt.append(np.full(nz.size, t)); ri.append(nz); rq.append(arr[nz].astype(float))
        d = units[t]
        issue = np.minimum(d, on_hand)
        on_hand -= issue
        nz = np.flatnonzero(issue > 0)
        if nz.size:
            it.append(np.full(nz.size, t)); ii.append(nz); iq.append(issue[nz].astype(float))
        lam_hat += alpha * (d - lam_hat)
        pos = on_hand + on_order
        s = lam_hat * (lead + 14)
        idx = np.flatnonzero((t >= launch) & (pos <= s) & (lam_hat > 0.005))
        if idx.size:
            target = s[idx] + 30.0 * lam_hat[idx]
            q = np.ceil(np.maximum(target - pos[idx], 1.0) / pack[idx]) * pack[idx]
            ta = t + lead[idx]
            ok = ta < T
            pipe[ta[ok], idx[ok]] += q[ok].astype(np.float32)
            on_order[idx] += q
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.array([], dtype=dt)
    return (cat(rt, int), cat(ri, int), cat(rq, float), cat(it, int), cat(ii, int), cat(iq, float))


def _events_to_frame(t, i, q, kind: str, ids: np.ndarray, start: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "sku": ids[i], "date": start + pd.to_timedelta(t, unit="D"), "kind": kind, "qty": q,
    })


def _assemble(rt, ri, rq, it, ii, iq, ids, start, asi) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mv = pd.concat([
        _events_to_frame(rt, ri, rq, "receipt", ids, start),
        _events_to_frame(it, ii, iq, "issue", ids, start),
    ], ignore_index=True)
    mv = mv.sort_values(["date", "sku", "kind"], kind="stable").reset_index(drop=True)
    cut = start + pd.Timedelta(days=int(asi))
    return mv[mv["date"] <= cut].reset_index(drop=True), mv[mv["date"] > cut].reset_index(drop=True)


def _master(n: int, truth: pd.DataFrame, rng: np.random.Generator, ids: np.ndarray, as_idx: int,
            start: pd.Timestamp) -> pd.DataFrame:
    arche = truth["archetype"].to_numpy()
    cat = rng.choice(CATEGORIES, n)
    cost = np.round(_logu(rng, 40, 4000, n), 2)
    vol = np.round(_logu(rng, 0.0008, 0.06, n), 5)
    shelf = np.full(n, np.nan)
    rate = truth["base_rate"].to_numpy()
    has = ((arche == "active") & (rate >= 0.3) & (rng.random(n) < 0.3))
    shelf[has] = rng.integers(120, 366, has.sum())
    has_o = (arche == "obsolete") & (rng.random(n) < 0.35)
    shelf[has_o] = rng.integers(150, 401, has_o.sum())
    loc = [f"{chr(65 + a)}-{b:02d}-{c}" for a, b, c in zip(rng.integers(0, 12, n), rng.integers(1, 41, n), rng.integers(1, 6, n))]
    succ = np.full(n, None, dtype=object)
    obs = np.flatnonzero(arche == "obsolete")
    act = np.flatnonzero(arche == "active")
    if obs.size and act.size:
        use = obs[rng.random(obs.size) < 0.6]
        succ[use] = ids[rng.choice(act, use.size)]
    return pd.DataFrame({
        "sku": ids, "description": [f"{c} item {s}" for c, s in zip(cat, ids)], "category": cat,
        "unit_cost": cost, "shelf_life_days": shelf, "volume_per_unit": vol, "location": loc,
        "on_hand_qty": np.nan, "first_receipt_date": pd.NaT, "successor_sku": succ,
    }).set_index("sku")


def make_warehouse(
    n_skus: int = 2000, n_hist: int = 1095, n_future: int = 120, seed: int = 7,
    end_date: str = "2026-10-31", mix: Optional[Dict[str, float]] = None,
) -> Warehouse:
    """A fully synthetic warehouse: demand archetypes -> replenishment -> ledger + truth."""
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(end_date).normalize()
    start = as_of - pd.Timedelta(days=n_hist - 1)
    units, truth = generate_demand(n_skus, n_hist, n_future, start, rng, mix)
    ids = np.array([f"SKU{i:05d}" for i in range(n_skus)])
    asi = n_hist - 1
    T = n_hist + n_future

    launch = truth["launch_idx"].to_numpy().astype(int)
    arche = truth["archetype"].to_numpy()
    init_qty = None
    extra = np.zeros((T, n_skus), dtype=np.float32)
    # never-picked forgotten stock: one receipt, never picked
    never = truth["never_picked"].to_numpy()
    base_q = np.where(never, rng.integers(20, 201, n_skus), 0).astype(float)
    # standing purchase orders that keep delivering to some dead SKUs
    auto = np.isin(arche, ["forgotten", "obsolete"]) & ~never & (rng.random(n_skus) < 0.35)
    interval = rng.integers(35, 70, n_skus)
    for i in np.flatnonzero(auto):
        e = int(truth["event_idx"].iat[i])
        qty = max(6.0, np.ceil(truth["base_rate"].iat[i] * truth["pick_size"].iat[i] * 20 / 6) * 6)
        extra[np.arange(e + 20, T, interval[i]), i] += qty
    truth["auto_replen"] = auto

    rt, ri, rq, it, ii, iq = simulate_ledger(units, launch, rng, extra=extra)
    # replace never-picked SKUs' policy-sized opening stock with the planted single receipt
    if never.any():
        keep_r = ~np.isin(ri, np.flatnonzero(never))
        rt, ri, rq = rt[keep_r], ri[keep_r], rq[keep_r]
        nv = np.flatnonzero(never)
        rt = np.concatenate([rt, launch[nv]]); ri = np.concatenate([ri, nv]); rq = np.concatenate([rq, base_q[nv]])
    mv, fut = _assemble(rt, ri, rq, it, ii, iq, ids, start, asi)
    sk = _master(n_skus, truth, rng, ids, asi, start)

    truth = truth.copy()
    truth.index = pd.Index(ids, name="sku")
    ev = truth["event_idx"].to_numpy()
    ev_dates = (start + pd.to_timedelta(np.maximum(ev, 0), unit="D")).to_numpy().astype("datetime64[ns]")
    truth["event_date"] = pd.to_datetime(np.where(ev >= 0, ev_dates, np.datetime64("NaT", "ns")))
    truth["truth_dead"] = truth["truth_state"].isin(["forgotten", "obsolete"])
    return Warehouse(movements=mv, future=fut, skus=sk, truth=truth.drop(columns=["event_idx", "launch_idx"]),
                     as_of=as_of, source="synthetic")


# --------------------------------------------------------------------------- real demand
def make_warehouse_from_sales(
    sales: pd.DataFrame, seed: int = 7, n_future: int = 120, planted_share: Optional[Dict[str, float]] = None,
    min_pick_days: int = 25, currency_cost: Optional[pd.Series] = None,
) -> Warehouse:
    """Put a supply side and planted events underneath **real** sales history.

    ``sales`` has columns ``sku``, ``date``, ``qty`` (one row per sale; returns removed).
    The last ``n_future`` days are held out as the validation window.  A random subset of
    well-selling SKUs is turned into forgotten / obsolete / temporarily-inactive SKUs by
    editing their *future* demand (stop it abruptly, fade it out, or pause it); everything
    else keeps its real demand and stays unlabelled (``truth_state == 'unlabelled'``).
    """
    share = planted_share or {"forgotten": 0.08, "obsolete": 0.06, "temporarily_inactive": 0.06}
    rng = np.random.default_rng(seed)
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s = s.groupby(["sku", "date"], as_index=False)["qty"].sum()
    d0, d1 = s["date"].min(), s["date"].max()
    T = (d1 - d0).days + 1
    asi = T - 1 - n_future
    if asi < 540:
        raise ValueError("Need at least ~540 days before the held-out window to plant events.")
    ids = np.array(sorted(s["sku"].astype(str).unique()))
    col = pd.Index(ids).get_indexer(s["sku"].astype(str))
    units = np.zeros((T, len(ids)), dtype=np.float32)
    np.add.at(units, ((s["date"] - d0).dt.days.to_numpy(), col), s["qty"].to_numpy(np.float32))

    pick_days = (units[:asi - 200] > 0).sum(0)
    first = np.where((units > 0).any(0), (units > 0).argmax(0), T)
    eligible = np.flatnonzero((pick_days >= min_pick_days) & (first < asi - 400))
    rng.shuffle(eligible)
    n = len(ids)
    arche = np.full(n, "unlabelled", dtype=object)
    event = np.full(n, -1)
    cursor = 0
    for name, frac in share.items():
        k = int(round(frac * len(eligible)))
        grp = eligible[cursor:cursor + k]
        cursor += k
        arche[grp] = name
        for i in grp:
            if name == "forgotten":
                e = asi - int(rng.integers(75, 241)); units[e + 1:, i] = 0.0; event[i] = e
            elif name == "obsolete":
                e = asi - int(rng.integers(75, 221)); dur = int(rng.integers(180, 301))
                fade = np.clip((e - np.arange(T)) / dur, 0.0, 1.0).astype(np.float32)
                units[:, i] *= fade; event[i] = e
            else:
                p0 = asi - int(rng.integers(10, 81)); p1 = asi + int(rng.integers(15, 71))
                units[p0:p1, i] = 0.0; event[i] = p0
    launch = np.maximum(first - 7, 0).clip(max=T - 1)
    truth = pd.DataFrame({"archetype": arche}, index=pd.Index(ids, name="sku"))
    truth["truth_state"] = truth["archetype"].replace({"unlabelled": "unlabelled"})
    truth["event_date"] = pd.to_datetime([d0 + pd.Timedelta(days=int(e)) if e >= 0 else pd.NaT for e in event])
    truth["truth_dead"] = truth["truth_state"].isin(["forgotten", "obsolete"])
    truth["never_picked"] = False
    truth["auto_replen"] = False

    rt, ri, rq, it, ii, iq = simulate_ledger(units, launch, rng)
    mv, fut = _assemble(rt, ri, rq, it, ii, iq, ids, d0, asi)
    cost = (currency_cost.reindex(ids) if currency_cost is not None else pd.Series(np.nan, index=ids)).to_numpy()
    sk = pd.DataFrame({"sku": ids, "unit_cost": cost}).set_index("sku")
    return Warehouse(movements=mv, future=fut, skus=sk, truth=truth,
                     as_of=d0 + pd.Timedelta(days=int(asi)), source="real-demand")
