"""Vectorised per-SKU feature engineering.

Everything is computed from the movement ledger for one ``as_of`` date, so the
same code powers the live analysis and the behavioural backtest (which simply
calls it again with an earlier date and never lets the future leak in).

Key ideas
---------
* **In-stock time, not calendar time.**  A SKU that was out of stock for two
  months did not get "neglected" for two months.  Idle spells and pick gaps are
  measured in days the SKU was actually on the shelf.
* **FIFO stock age.**  The units on hand are assumed to be the most recent
  receipts, so the *oldest* unit's age is honest even when a SKU keeps being
  replenished.
* **Seasonality from the SKU's own history.**  Year-on-year profile correlation,
  peak concentration, and "was it also quiet at this point last year?".
* **Demand pattern.**  Syntetos-Boylan classification (smooth / erratic /
  intermittent / lumpy) from ADI and CV^2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .schema import ensure_sku_table
from .survival import idle_pvalue

BLOCK = 28  # seasonality block length in days (13 blocks ~ one year)
SEASON_WINDOW = BLOCK * 26  # two years


def _group_stat(codes: np.ndarray, values: np.ndarray, n: int, how: str) -> np.ndarray:
    """Group min/max of ``values`` by integer ``codes`` into an array of length n (NaT/NaN if empty)."""
    out = pd.Series(values).groupby(codes).agg(how)
    res = np.full(n, np.nan)
    res[out.index.to_numpy()] = out.to_numpy()
    return res


def _fifo_age(mv: pd.DataFrame, codes: np.ndarray, ledger: np.ndarray, shelf: np.ndarray,
              as_of: pd.Timestamp, n: int):
    """FIFO age of units on hand: average, oldest, and quantity already past shelf life."""
    isr = (mv["kind"] == "receipt").to_numpy()
    rc = codes[isr]
    rq = mv["qty"].to_numpy(float)[isr]
    rage = (as_of - mv["date"][isr]).dt.days.to_numpy(float)
    order = np.lexsort((rage, rc))  # by sku, then newest (smallest age) first
    rc, rq, rage = rc[order], rq[order], rage[order]
    on_hand = np.maximum(ledger, 0.0)
    cum = pd.Series(rq).groupby(rc).cumsum().to_numpy()
    later = cum - rq
    attr = np.clip(on_hand[rc] - later, 0.0, rq)
    tot = np.bincount(rc, weights=attr, minlength=n)
    wsum = np.bincount(rc, weights=attr * rage, minlength=n)
    avg_age = np.where(tot > 0, wsum / np.maximum(tot, 1e-12), np.nan)
    oldest = np.full(n, np.nan)
    m = attr > 0
    if m.any():
        s = pd.Series(rage[m]).groupby(rc[m]).max()
        oldest[s.index.to_numpy()] = s.to_numpy()
    sh = shelf[rc]
    exp_mask = (attr > 0) & np.isfinite(sh) & (rage > sh)
    expired = np.bincount(rc[exp_mask], weights=attr[exp_mask], minlength=n)
    return avg_age, oldest, expired


def compute_features(
    movements: pd.DataFrame,
    skus: Optional[pd.DataFrame],
    as_of: pd.Timestamp,
    cfg: Config,
) -> pd.DataFrame:
    """Compute the per-SKU feature table as of ``as_of`` (movements after it are ignored)."""
    as_of = pd.Timestamp(as_of).normalize()
    mv = movements[movements["date"] <= as_of]
    skus = ensure_sku_table(skus, mv)
    ids = skus.index
    n = len(ids)
    codes_all = ids.get_indexer(mv["sku"])
    keep = codes_all >= 0
    mv = mv[keep]
    codes = codes_all[keep]
    qty = mv["qty"].to_numpy(float)
    isr = (mv["kind"] == "receipt").to_numpy()
    back = (as_of - mv["date"]).dt.days.to_numpy()

    tot_r = np.bincount(codes[isr], weights=qty[isr], minlength=n)
    tot_i = np.bincount(codes[~isr], weights=qty[~isr], minlength=n)
    ledger = tot_r - tot_i
    on_hand = np.maximum(ledger, 0.0)

    dates_ns = mv["date"].to_numpy("datetime64[ns]").astype("int64").astype(float)
    first_move = _group_stat(codes, dates_ns, n, "min")
    last_pick = _group_stat(codes[~isr], dates_ns[~isr], n, "max")
    last_rec = _group_stat(codes[isr], dates_ns[isr], n, "max")
    n_pick_days_all = np.bincount(codes[~isr], minlength=n)

    def _to_ts(a):
        return pd.to_datetime(np.where(np.isnan(a), np.nan, a).astype("float64"), unit="ns", errors="coerce")

    first_move_ts, last_pick_ts, last_rec_ts = _to_ts(first_move), _to_ts(last_pick), _to_ts(last_rec)
    history_days = np.where(np.isnan(first_move), 0, (as_of - first_move_ts).days + 1)
    days_since_pick = np.where(np.isnan(last_pick), np.nan, (as_of - last_pick_ts).days)
    days_since_rec = np.where(np.isnan(last_rec), np.nan, (as_of - last_rec_ts).days)

    shelf = skus["shelf_life_days"].to_numpy(float)
    avg_age, oldest_age, expired_qty = _fifo_age(mv, codes, ledger, shelf, as_of, n)

    # ------------------------------------------------------------------ daily window
    data_start = mv["date"].min() if len(mv) else as_of
    start = max(data_start, as_of - pd.Timedelta(days=cfg.lookback_days - 1))
    L = int((as_of - start).days) + 1
    order = np.argsort(codes, kind="stable")
    codes_s, qty_s, isr_s = codes[order], qty[order], isr[order]
    col_s = L - 1 - back[order]

    cols_out = {k: np.zeros(n) for k in [
        "idle_instock_days", "n_pick_days_win", "n_gaps", "mean_gap", "max_gap", "gap_cv",
        "adi", "cv2", "pick_days_30", "pick_days_90", "pick_days_180", "pick_days_365",
        "units_90", "units_365", "receipts_90", "receipt_events_90", "instock_days_365",
        "hist_instock_days", "instock_days_win"]}
    p_idle = np.ones(n)
    p_exp = np.ones(n)
    p_logn = np.ones(n)
    erosion = np.full(n, np.nan)
    season_corr = np.full(n, np.nan)
    season_top3 = np.full(n, np.nan)
    ly_quiet = np.full(n, np.nan)
    ly_return = np.full(n, np.nan)
    season_assessable = np.zeros(n, dtype=bool)
    n_pick_days_2y = np.zeros(n)

    for c0 in range(0, n, cfg.chunk_size):
        c1 = min(n, c0 + cfg.chunk_size)
        nc = c1 - c0
        lo, hi = np.searchsorted(codes_s, [c0, c1])
        rows = codes_s[lo:hi] - c0
        cols = col_s[lo:hi]
        q = qty_s[lo:hi]
        r_flag = isr_s[lo:hi]
        pre = cols < 0
        sign = np.where(r_flag, 1.0, -1.0)
        stock0 = np.bincount(rows[pre], weights=(sign * q)[pre], minlength=nc)
        w = ~pre
        R = np.bincount((rows * L + cols)[w & r_flag], weights=q[w & r_flag], minlength=nc * L).reshape(nc, L)
        I = np.bincount((rows * L + cols)[w & ~r_flag], weights=q[w & ~r_flag], minlength=nc * L).reshape(nc, L)

        B = stock0[:, None] + np.cumsum(R - I, axis=1)
        instock = (B + I) > 0.5
        S = np.cumsum(instock, axis=1).astype(np.int32)
        pick = I > 0
        cp = np.cumsum(pick, axis=1).astype(np.int32)
        npick = pick.sum(1)
        has = npick > 0
        rr = np.arange(nc)
        lastp = np.where(has, L - 1 - np.argmax(pick[:, ::-1], axis=1), -1)
        S_T = S[:, -1].astype(float)
        S_last = np.where(has, S[rr, np.maximum(lastp, 0)], 0).astype(float)
        idle = S_T - S_last

        sl = slice(c0, c1)
        cols_out["idle_instock_days"][sl] = idle
        cols_out["n_pick_days_win"][sl] = npick
        cols_out["instock_days_win"][sl] = S_T
        cols_out["hist_instock_days"][sl] = S_last

        # ---- gaps between consecutive pick days, in in-stock days
        r_idx, c_idx = np.nonzero(pick)
        s_at = S[r_idx, c_idx].astype(float)
        same = r_idx[1:] == r_idx[:-1]
        g = np.maximum((s_at[1:] - s_at[:-1])[same], 1.0)
        gr = r_idx[1:][same]
        ng = np.bincount(gr, minlength=nc).astype(float)
        lg = np.log(g)
        sum_g = np.bincount(gr, weights=g, minlength=nc)
        sum_g2 = np.bincount(gr, weights=g * g, minlength=nc)
        sum_lg = np.bincount(gr, weights=lg, minlength=nc)
        sum_lg2 = np.bincount(gr, weights=lg * lg, minlength=nc)
        with np.errstate(divide="ignore", invalid="ignore"):
            mg = np.where(ng > 0, sum_g / ng, np.nan)
            vg = np.where(ng > 0, sum_g2 / ng - mg**2, np.nan)
            lm = np.where(ng > 0, sum_lg / ng, 0.0)
            lv = np.where(ng > 0, np.maximum(sum_lg2 / ng - lm**2, 0.0), 0.0)
        mx = np.zeros(nc)
        if len(g):
            np.maximum.at(mx, gr, g)
        cols_out["n_gaps"][sl] = ng
        cols_out["mean_gap"][sl] = mg
        cols_out["max_gap"][sl] = np.where(ng > 0, mx, np.nan)
        cols_out["gap_cv"][sl] = np.where(ng > 1, np.sqrt(np.maximum(vg, 0)) / np.maximum(mg, 1e-9), np.nan)

        # ---- demand size variability (Syntetos-Boylan)
        qv = I[r_idx, c_idx]
        cnt = np.bincount(r_idx, minlength=nc).astype(float)
        s1 = np.bincount(r_idx, weights=qv, minlength=nc)
        s2 = np.bincount(r_idx, weights=qv * qv, minlength=nc)
        with np.errstate(divide="ignore", invalid="ignore"):
            mq = np.where(cnt > 0, s1 / cnt, np.nan)
            vq = np.where(cnt > 0, np.maximum(s2 / cnt - mq**2, 0), np.nan)
            cols_out["cv2"][sl] = np.where(cnt > 1, vq / mq**2, np.nan)
            cols_out["adi"][sl] = np.where(cnt > 0, S_T / cnt, np.nan)

        # ---- survival p-value for the current idle spell
        p, pe, pl = idle_pvalue(idle, npick, S_last, ng, lm, lv)
        p_idle[sl], p_exp[sl], p_logn[sl] = p, pe, pl

        # ---- recent activity
        for wdw, name in ((30, "pick_days_30"), (90, "pick_days_90"), (180, "pick_days_180"), (365, "pick_days_365")):
            cols_out[name][sl] = pick[:, -wdw:].sum(1)
        cols_out["units_90"][sl] = I[:, -90:].sum(1)
        cols_out["units_365"][sl] = I[:, -365:].sum(1)
        cols_out["instock_days_365"][sl] = instock[:, -365:].sum(1)
        cols_out["receipts_90"][sl] = R[:, -90:].sum(1)
        cols_out["receipt_events_90"][sl] = (R[:, -90:] > 0).sum(1)

        # ---- erosion: was demand fading before the current silence began?
        def at(arr, idx):
            return np.where(idx < 0, 0, arr[rr, np.clip(idx, 0, L - 1)])

        e = np.where(has, lastp, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pa = at(cp, e) - at(cp, e - 180)
            ia = at(S, e) - at(S, e - 180)
            pb = at(cp, e - 180) - at(cp, e - 360)
            ib = at(S, e - 180) - at(S, e - 360)
            ok = has & (ia >= 30) & (ib >= 30) & (pb >= 4)
            erosion[sl] = np.where(ok, (pa / np.maximum(ia, 1)) / (pb / np.maximum(ib, 1)), np.nan)

        # ---- seasonality
        if L >= SEASON_WINDOW:
            n2y = pick[:, -SEASON_WINDOW:].sum(1)
            n_pick_days_2y[sl] = n2y
            blocks = I[:, -SEASON_WINDOW:].reshape(nc, 26, BLOCK).sum(2)[:, ::-1]  # block 0 = most recent
            y2, y1 = blocks[:, :13], blocks[:, 13:]
            a = y1 - y1.mean(1, keepdims=True)
            b = y2 - y2.mean(1, keepdims=True)
            den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
            with np.errstate(divide="ignore", invalid="ignore"):
                season_corr[sl] = np.where(den > 0, (a * b).sum(1) / den, np.nan)
                comb = y1 + y2
                tot = comb.sum(1)
                top3 = np.sort(comb, axis=1)[:, ::-1][:, :3].sum(1)
                season_top3[sl] = np.where(tot > 0, top3 / tot, np.nan)
                nxt = blocks[:, 10:13].sum(1) + blocks[:, 23:26].sum(1)
                ly_return[sl] = np.where(tot > 0, nxt / (tot * 168.0 / 728.0), np.nan)
            season_assessable[sl] = n2y >= cfg.season_min_pick_days
            cI = np.cumsum(I, axis=1)
            b1 = L - 1 - 364
            a1 = np.clip(np.where(has, lastp, 0) - 364, 0, b1)
            pre_cum = np.where(a1 > 0, cI[rr, np.maximum(a1 - 1, 0)], 0.0)
            ly_units = cI[:, b1] - pre_cum
            ly_rate = ly_units / (b1 - a1 + 1)
            avg_rate = cI[:, b1] / (b1 + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ly_quiet[sl] = np.where(has & (avg_rate > 0), ly_rate / avg_rate, np.nan)

    f = pd.DataFrame(index=ids)
    f["on_hand"] = on_hand
    f["first_move_date"] = first_move_ts
    f["last_pick_date"] = last_pick_ts
    f["last_receipt_date"] = last_rec_ts
    f["history_days"] = history_days
    f["never_picked"] = n_pick_days_all == 0
    f["n_pick_days_all"] = n_pick_days_all
    f["days_since_pick"] = np.where(np.isnan(days_since_pick), history_days, days_since_pick)
    f["days_since_receipt"] = days_since_rec
    for k, v in cols_out.items():
        f[k] = v
    f["p_idle"] = p_idle
    f["p_exp"] = p_exp
    f["p_lognormal"] = p_logn
    f["erosion_ratio"] = erosion
    f["season_assessable"] = season_assessable
    f["season_corr"] = season_corr
    f["season_top3"] = season_top3
    f["ly_quiet_ratio"] = ly_quiet
    f["ly_return_ratio"] = ly_return
    f["n_pick_days_2y"] = n_pick_days_2y
    f["stock_age_avg"] = avg_age
    f["stock_age_oldest"] = oldest_age
    f["expired_qty"] = expired_qty

    with np.errstate(divide="ignore", invalid="ignore"):
        f["rate_per30"] = np.where(f["instock_days_365"] > 0, f["pick_days_365"] / f["instock_days_365"] * 30.0, 0.0)
        daily = np.where(f["instock_days_365"] > 0, f["units_365"] / f["instock_days_365"], 0.0)
        f["avg_daily_units"] = daily
        f["cover_days"] = np.where(daily > 0, f["on_hand"] / daily, np.inf)
        f["life_used"] = f["stock_age_oldest"] / shelf
        f["days_to_expiry"] = shelf - f["stock_age_oldest"]

    adi, cv2 = f["adi"].to_numpy(), f["cv2"].to_numpy()
    pattern = np.full(n, "no_demand", dtype=object)
    have = np.isfinite(adi)
    cv2f = np.where(np.isfinite(cv2), cv2, 0.0)
    pattern[have & (adi < 1.32) & (cv2f < 0.49)] = "smooth"
    pattern[have & (adi >= 1.32) & (cv2f < 0.49)] = "intermittent"
    pattern[have & (adi < 1.32) & (cv2f >= 0.49)] = "erratic"
    pattern[have & (adi >= 1.32) & (cv2f >= 0.49)] = "lumpy"
    f["demand_pattern"] = pattern
    f["window_days"] = L
    return f
