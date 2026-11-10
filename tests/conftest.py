import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fid.simulate import make_warehouse  # noqa: E402


@pytest.fixture(scope="session")
def small_wh():
    return make_warehouse(n_skus=600, seed=5)


def ledger(rows):
    """rows: (sku, 'YYYY-MM-DD', kind, qty) -> standard movements frame."""
    df = pd.DataFrame(rows, columns=["sku", "date", "kind", "qty"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "sku"]).reset_index(drop=True)


def steady_then_silent(sku, start, n_days_active, silence_days, every=2, qty=3, opening=2000):
    """Picked every ``every`` days for ``n_days_active`` days, then nothing."""
    d0 = pd.Timestamp(start)
    rows = [(sku, d0, "receipt", opening)]
    for k in range(0, n_days_active, every):
        rows.append((sku, d0 + pd.Timedelta(days=k + 1), "issue", qty))
    return rows
