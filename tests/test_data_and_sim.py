import io

import numpy as np
import pandas as pd
import pytest

from fid import Config, analyze
from fid.export import fmt_money, friendly_table, read_table, to_excel_bytes
from fid.realdata import load_online_retail
from fid.schema import guess_kind_values, standardize_movements, standardize_skus, suggest_mapping, MOVEMENT_ALIASES
from fid.simulate import make_warehouse, make_warehouse_from_sales
from tests.conftest import ROOT


# ---------------------------------------------------------------- simulator
def test_simulator_is_deterministic():
    a = make_warehouse(n_skus=80, seed=1, n_hist=800)
    b = make_warehouse(n_skus=80, seed=1, n_hist=800)
    pd.testing.assert_frame_equal(a.movements, b.movements)


def test_stock_never_negative_and_truth_complete(small_wh):
    mv = small_wh.movements
    signed = np.where(mv["kind"] == "receipt", mv["qty"], -mv["qty"])
    bal = pd.Series(signed, index=mv["sku"]).groupby(level=0).sum()
    assert (bal >= -1e-6).all()
    assert set(small_wh.truth.index) == set(small_wh.skus.index)
    assert small_wh.movements["date"].max() <= small_wh.as_of
    assert small_wh.future["date"].min() > small_wh.as_of
    states = set(small_wh.truth["truth_state"])
    assert {"forgotten", "obsolete", "active", "slow_moving"} <= states


def test_planted_forgotten_really_have_no_future_demand(small_wh):
    fut = small_wh.future[small_wh.future["kind"] == "issue"]
    forg = small_wh.truth.index[small_wh.truth["truth_state"] == "forgotten"]
    assert fut["sku"].isin(forg).sum() == 0


# ---------------------------------------------------------------- engine on simulated data
def test_engine_beats_naive_same_threshold(small_wh):
    from fid.evaluate import planted_eval
    ev = planted_eval(small_wh, Config())
    assert ev["dead"]["engine"]["recall"] > 0.9
    assert ev["dead"]["engine"]["precision"] > ev["dead"]["naive_same_idle"]["precision"]
    assert ev["seasonal_false_alarms"]["engine"] <= ev["seasonal_false_alarms"]["naive"]


def test_backtest_is_label_free_and_sane(small_wh):
    from fid.evaluate import behavioural_backtest
    bt = behavioural_backtest(small_wh.movements, small_wh.skus, Config(), small_wh.as_of)
    t = bt["table"].set_index("group")
    assert t.loc["Engine: forgotten + obsolete", "stayed_dormant_pct"] > t.loc["Engine: active", "stayed_dormant_pct"] + 50


def test_backtest_returns_none_on_short_history(small_wh):
    from fid.evaluate import behavioural_backtest
    short = small_wh.movements[small_wh.movements["date"] > small_wh.as_of - pd.Timedelta(days=150)]
    assert behavioural_backtest(short, small_wh.skus, Config(), small_wh.as_of) is None


# ---------------------------------------------------------------- cleaning
def test_three_movement_modes():
    raw = pd.DataFrame({"Item": ["a", "a", "b", "b"], "When": ["2025-01-01", "2025-01-05", "2025-01-01", "bad"],
                        "Qty": [10, -4, 5, 1], "T": ["GRN", "DISPATCH", "GRN", "DISPATCH"]})
    m = {"sku": "Item", "date": "When", "qty": "Qty", "kind": "T"}
    signed, rep = standardize_movements(raw, m, "signed")
    assert set(signed["kind"]) == {"receipt", "issue"} and rep["rows_dropped_bad_date"] == 1
    typed, _ = standardize_movements(raw, m, "type_column", ["GRN"], ["DISPATCH"])
    assert typed["qty"].min() > 0 and len(typed) == 3
    sales, rep2 = standardize_movements(raw, m, "issues_only")
    assert set(sales["kind"]) == {"issue"} and rep2["notes"]


def test_suggestions_and_kind_guess():
    sug = suggest_mapping(["Item Code", "Posting Date", "Quantity", "Movement Type"], MOVEMENT_ALIASES)
    assert sug["sku"] == "Item Code" and sug["date"] == "Posting Date" and sug["qty"] == "Quantity"
    rec, iss = guess_kind_values(["GRN", "Dispatch", "Receipt", "Pick"])
    assert {"GRN", "Receipt"} <= set(rec) and {"Dispatch", "Pick"} <= set(iss)


def test_templates_run_end_to_end():
    mv = pd.read_csv(ROOT / "templates" / "movements_template.csv")
    m, _ = standardize_movements(mv, {"sku": "sku", "date": "date", "qty": "quantity", "kind": "type"}, "type_column")
    sk = standardize_skus(pd.read_csv(ROOT / "templates" / "sku_master_template.csv"), {"sku": "sku", **{c: c for c in
                          ["description", "category", "unit_cost", "shelf_life_days", "volume_per_unit", "location", "on_hand_qty", "successor_sku"]}})
    a = analyze(m, sk, Config())
    assert len(a.result) == 3


def test_sample_data_files_load_and_analyse():
    mv = pd.read_csv(ROOT / "data" / "sample" / "movements.csv.gz", parse_dates=["date"])
    sk = pd.read_csv(ROOT / "data" / "sample" / "sku_master.csv").set_index("sku")
    a = analyze(mv, sk, Config())
    assert a.summary["n_forgotten"] > 0


# ---------------------------------------------------------------- export / io
def test_read_table_csv_semicolon_and_xlsx(tmp_path):
    csv = "sku;date;qty\nA;2025-01-01;3\nB;2025-01-02;4\n".encode()
    df = read_table(csv, "x.csv")
    assert list(df.columns) == ["sku", "date", "qty"] and len(df) == 2
    p = tmp_path / "x.xlsx"
    df.to_excel(p, index=False)
    assert len(read_table(p.read_bytes(), "x.xlsx")) == 2


def test_excel_report_and_money(small_wh):
    a = analyze(small_wh.movements, small_wh.skus, Config(), as_of=small_wh.as_of)
    data = to_excel_bytes(a.result, a.summary, a.cfg)
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert set(sheets) == {"Action list", "All SKUs", "Summary"}
    assert len(friendly_table(a.result)) == len(a.result)
    assert fmt_money(25_000_000) == "₹2.50 Cr" and fmt_money(250_000) == "₹2.50 L" and fmt_money(1_500_000, "$") == "$1.50M"


# ---------------------------------------------------------------- real-data plumbing
def test_online_retail_loader_filters_and_aggregates(tmp_path):
    raw = pd.DataFrame({
        "Invoice": ["1", "2", "C3", "4", "5"], "StockCode": ["85123A", "85123A", "85123A", "POST", "22423"],
        "Description": ["LAMP"] * 5, "Quantity": [6, 6, -6, 1, 4],
        "InvoiceDate": ["2010-01-04 09:00", "2010-01-04 15:00", "2010-01-05 10:00", "2010-01-05 11:00", "2010-01-06 12:00"],
        "Price": [2.5, 2.5, 2.5, 18.0, 9.0], "Customer ID": [1, 2, 3, 4, 5], "Country": ["UK"] * 5})
    p = tmp_path / "retail.csv"
    raw.to_csv(p, index=False)
    sales, master = load_online_retail(p)
    assert set(sales["sku"]) == {"85123A", "22423"}           # POST and cancellation removed
    assert sales.loc[sales["sku"] == "85123A", "qty"].iloc[0] == 12  # same-day invoices summed
    assert master.loc["22423", "unit_cost"] > 0


def test_real_demand_layer_plants_events(small_wh):
    """Use a simulated demand history as a stand-in for a real sales file."""
    big = make_warehouse(n_skus=300, seed=9, n_hist=1300, n_future=0)
    sales = big.movements[big.movements["kind"] == "issue"][["sku", "date", "qty"]]
    wh = make_warehouse_from_sales(sales, seed=2, n_future=120)
    states = wh.truth["truth_state"].value_counts()
    assert states.get("forgotten", 0) > 0 and states.get("obsolete", 0) > 0 and states.get("unlabelled", 0) > 0
    assert wh.future["date"].min() > wh.as_of


def test_m5_loader_with_standin_files(tmp_path):
    days = [f"d_{i}" for i in range(1, 6)]
    pd.DataFrame({"id": ["A_CA_1", "A_TX_1", "B_CA_1"], "item_id": ["A", "A", "B"], "dept_id": ["D"] * 3,
                  "cat_id": ["FOODS"] * 3, "state_id": ["CA", "TX", "CA"], "store_id": ["CA_1", "TX_1", "CA_1"],
                  **{d: v for d, v in zip(days, [[0, 1, 0], [2, 0, 0], [0, 0, 3], [1, 1, 0], [0, 0, 0]])}}
                 ).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    pd.DataFrame({"d": days, "date": pd.date_range("2015-01-01", periods=5).strftime("%Y-%m-%d")}).to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame({"store_id": ["CA_1", "CA_1", "TX_1"], "item_id": ["A", "B", "A"], "wm_yr_wk": [1, 1, 1],
                  "sell_price": [2.0, 4.0, 9.0]}).to_csv(tmp_path / "sell_prices.csv", index=False)
    from fid.realdata import load_m5
    sales, master = load_m5(tmp_path, "CA_1")
    assert set(sales["sku"]) == {"A", "B"} and sales["qty"].sum() == 6  # CA_1 only: A=2+1, B=3; the TX_1 row is excluded
    assert master.loc["A", "unit_cost"] == pytest.approx(1.4)
