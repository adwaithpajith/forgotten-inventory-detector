"""Loaders that turn public retail datasets into the ``sales`` table used by the simulator.

Both loaders return ``(sales, master)``:

``sales``   columns ``sku``, ``date``, ``qty``  (one row per sale, returns removed)
``master``  index ``sku`` with ``description``, ``category`` and ``unit_cost`` (a price proxy)

Supported sources
-----------------
* **M5 Walmart** (Kaggle ``m5-forecasting-accuracy``): 5+ years of daily unit sales, most series
  intermittent.  Long enough for the seasonality logic.  Pick one store so each row is a SKU
  in one warehouse.
* **Online Retail II** (UCI, CC BY 4.0): two years of invoice lines for a UK gift-ware wholesaler,
  strongly seasonal.  Two years is *not* enough to assess seasonality once a validation window is
  held out, so prefer M5 for the full pipeline.

Neither file can be fetched from inside the project's sandbox; ``scripts/prepare_real_demand.py``
downloads them on your own machine.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_online_retail(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read Online Retail II from the UCI ``.xlsx`` (both sheets) or a ``.csv`` export."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        parts = pd.read_excel(path, sheet_name=None)
        raw = pd.concat(parts.values(), ignore_index=True)
    else:
        raw = pd.read_csv(path)
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"Column not found (looked for {names}); have {list(raw.columns)}")

    inv = raw[pick("invoice", "invoiceno")].astype(str)
    sku = raw[pick("stockcode")].astype(str).str.strip().str.upper()
    desc = raw[pick("description")]
    qty = pd.to_numeric(raw[pick("quantity")], errors="coerce")
    date = pd.to_datetime(raw[pick("invoicedate")], errors="coerce")
    price = pd.to_numeric(raw[pick("price", "unitprice")], errors="coerce")

    ok = (
        ~inv.str.upper().str.startswith("C")        # cancellations
        & (qty > 0) & (price > 0) & date.notna()
        & sku.str.match(r"^\d{5}")                   # real products only (drops POST, DOT, M, BANK CHARGES, ...)
    )
    df = pd.DataFrame({"sku": sku[ok], "date": date[ok].dt.normalize(), "qty": qty[ok], "price": price[ok],
                       "description": desc[ok]})
    sales = df.groupby(["sku", "date"], as_index=False)["qty"].sum()
    master = df.groupby("sku").agg(description=("description", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""),
                                   unit_cost=("price", "median"))
    master["category"] = "Gift-ware"
    master["unit_cost"] = (master["unit_cost"] * 0.6).round(2)  # sale price -> rough cost proxy
    return sales, master


def load_m5(directory: str | Path, store: str = "CA_1") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read M5 (``sales_train_evaluation.csv``, ``calendar.csv``, ``sell_prices.csv``) for one store."""
    d = Path(directory)
    sales_wide = pd.read_csv(d / "sales_train_evaluation.csv")
    sales_wide = sales_wide[sales_wide["store_id"] == store]
    if sales_wide.empty:
        raise ValueError(f"Store {store!r} not found in M5 sales file.")
    cal = pd.read_csv(d / "calendar.csv", usecols=["d", "date"])
    cal["date"] = pd.to_datetime(cal["date"])
    dcols = [c for c in sales_wide.columns if re.fullmatch(r"d_\d+", c)]
    dates = cal.set_index("d").loc[dcols, "date"].to_numpy()
    vals = sales_wide[dcols].to_numpy(dtype=np.float32)
    r, c = np.nonzero(vals > 0)
    sales = pd.DataFrame({"sku": sales_wide["item_id"].to_numpy()[r], "date": dates[c], "qty": vals[r, c]})
    prices = pd.read_csv(d / "sell_prices.csv")
    prices = prices[prices["store_id"] == store].groupby("item_id")["sell_price"].mean()
    master = pd.DataFrame({"sku": sales_wide["item_id"].to_numpy(), "category": sales_wide["cat_id"].to_numpy(),
                           "description": sales_wide["item_id"].to_numpy()}).drop_duplicates("sku").set_index("sku")
    master["unit_cost"] = (prices.reindex(master.index) * 0.7).round(2).to_numpy()
    return sales, master


def window(sales: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Restrict sales to a date window (e.g. to cap M5 at the last three years for speed)."""
    out = sales
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out
