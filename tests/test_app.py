"""Drive the Streamlit app headlessly."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

APP = str(ROOT / "app" / "streamlit_app.py")


def test_demo_mode_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    assert any("Forgotten" in m.label for m in at.metric)


def _own_data_app():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("My own data").run()
    return at


def _sample_csv(rename=None):
    import pandas as pd
    mv = pd.read_csv(ROOT / "data" / "sample" / "movements.csv.gz")
    return mv.rename(columns=rename or {}).to_csv(index=False).encode()


def test_upload_flow_with_unfamiliar_column_names():
    at = _own_data_app()
    csv = _sample_csv({"sku": "Item Code", "date": "Posting Date", "kind": "Movement Type", "qty": "Quantity"})
    master = (ROOT / "data" / "sample" / "sku_master.csv").read_bytes()
    at.sidebar.file_uploader(key="mv").set_value(("stock.csv", csv, "text/csv"))
    at.sidebar.file_uploader(key="sk").set_value(("master.csv", master, "text/csv"))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("movements for" in s.value for s in at.success), [s.value for s in at.success]
    assert len(at.tabs) == 5
    assert any("Forgotten + obsolete" in m.label for m in at.metric)


def test_sales_only_without_on_hand_is_blocked_with_a_clear_message():
    at = _own_data_app()
    csv = _sample_csv({"sku": "sku", "date": "date", "kind": "kind", "qty": "qty"})
    at.sidebar.file_uploader(key="mv").set_value(("stock.csv", csv, "text/csv"))
    at.run()
    at.radio[0].set_value("Sales / picks only (every row is an outbound movement)").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("on-hand quantity" in e.value for e in at.error), [e.value for e in at.error]


def test_backtest_ml_excel_and_explorer_interactions():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    if at.sidebar.checkbox:  # ML second opinion, present when models/ml_classifier.joblib loads
        at.sidebar.checkbox[0].check().run()
    assert not at.exception, [e.value for e in at.exception]
    next(b for b in at.button if b.label == "Run backtest").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Classified as of" in s.value for s in at.success), [s.value for s in at.success]
    next(b for b in at.button if b.label == "Build Excel report").click().run()
    assert not at.exception, [e.value for e in at.exception]
    typed = next(t for t in at.text_input if t.label.startswith("…or type"))
    typed.set_value("SKU00001").run()
    assert not at.exception, [e.value for e in at.exception]


def test_own_data_mode_waits_for_upload():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("My own data").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Bring your own data" in s.value for s in at.subheader)
