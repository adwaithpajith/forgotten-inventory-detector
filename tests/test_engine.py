"""Hand-built SKUs whose correct label is obvious."""
import numpy as np
import pandas as pd

from fid import Config, analyze
from tests.conftest import ledger, steady_then_silent

AS_OF = pd.Timestamp("2026-06-30")


def build():
    rows = []
    # active: picked every 2 days right up to as_of, 400 days of history
    rows += steady_then_silent("ACTIVE", "2025-05-01", 425, 0)
    # forgotten: picked every 2 days for 300 days, then silent 110 days before as_of
    rows += steady_then_silent("FORGOTTEN", "2025-05-01", 300, 110)
    # paused briefly: picked every 2 days, silent for only 20 days
    rows += steady_then_silent("PAUSED", "2025-05-01", 405, 20)
    # never picked: received 200 days ago, nothing since
    rows += [("NEVER", pd.Timestamp("2025-12-12"), "receipt", 100)]
    # out of stock: received then fully picked
    rows += [("EMPTY", pd.Timestamp("2025-05-01"), "receipt", 10), ("EMPTY", pd.Timestamp("2025-05-20"), "issue", 10)]
    # too new: received 30 days ago, picked once
    rows += [("NEW", pd.Timestamp("2026-05-31"), "receipt", 50), ("NEW", pd.Timestamp("2026-06-10"), "issue", 2)]
    # anchor so the data starts well before every SKU
    rows += [("ANCHOR", pd.Timestamp("2025-05-01"), "receipt", 5)]
    return ledger(rows)


def test_clear_cases():
    a = analyze(build(), None, Config(), as_of=AS_OF)
    s = a.result["status"]
    assert s["ACTIVE"] == "active"
    assert s["FORGOTTEN"] == "forgotten"
    assert s["PAUSED"] == "temporarily_inactive"
    assert s["NEVER"] == "forgotten"
    assert s["EMPTY"] == "out_of_stock"
    assert s["NEW"] == "insufficient_history"


def test_risk_and_money_ordering():
    skus = pd.DataFrame({"sku": ["ACTIVE", "FORGOTTEN"], "unit_cost": [10.0, 50.0]}).set_index("sku")
    a = analyze(build(), skus, Config(), as_of=AS_OF)
    r = a.result
    assert r.loc["FORGOTTEN", "risk_score"] > r.loc["ACTIVE", "risk_score"]
    assert r.loc["FORGOTTEN", "capital_tied"] == r.loc["FORGOTTEN", "on_hand"] * 50.0
    assert a.summary["n_forgotten"] >= 2
    assert (r["risk_score"].between(0, 100)).all()


def test_idle_time_counts_only_days_in_stock():
    """A SKU that was out of stock during its quiet spell must not look neglected."""
    rows = [("S", "2025-05-01", "receipt", 30)]
    d = pd.Timestamp("2025-05-01")
    for k in range(1, 31):  # sell all 30 units, so the shelf is empty until the restock
        rows.append(("S", d + pd.Timedelta(days=k), "issue", 1))
    rows.append(("S", "2026-05-01", "receipt", 30))  # restocked 60 days before as_of
    rows += [("ANCHOR", "2025-05-01", "receipt", 5)]
    a = analyze(ledger(rows), None, Config(), as_of=AS_OF)
    assert a.result.loc["S", "idle_instock_days"] < 70
    assert a.result.loc["S", "days_since_pick"] > 300


def test_obsolete_by_expiry():
    rows = steady_then_silent("EXP", "2025-05-01", 200, 0, every=10, qty=1, opening=500)
    rows += [("ANCHOR", "2025-05-01", "receipt", 5)]
    skus = pd.DataFrame({"sku": ["EXP"], "shelf_life_days": [120.0]}).set_index("sku")
    a = analyze(ledger(rows), skus, Config(), as_of=AS_OF)
    assert a.result.loc["EXP", "status"] == "obsolete"
    assert a.result.loc["EXP", "flag_expiry"]


def test_naive_rule_would_misfire_where_engine_does_not():
    """A fast mover quiet for 30 days is *not* flagged dead; the 90-day naive rule is irrelevant here."""
    a = analyze(build(), None, Config(), as_of=AS_OF)
    assert not a.result.loc["PAUSED", "dead_flag"]
