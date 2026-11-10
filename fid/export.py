"""Helpers shared by the app and the tests: reading uploads, money formatting, tidy tables, Excel export."""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .classify import ACTIONS  # noqa: F401  (re-exported for convenience)
from .config import Config

TABLE_COLUMNS: Dict[str, str] = {
    "description": "Description",
    "category": "Category",
    "location": "Location",
    "status_label": "Status",
    "risk_score": "Risk score",
    "on_hand": "On hand",
    "unit_cost": "Unit cost",
    "capital_tied": "Capital tied up",
    "holding_cost_per_year": "Holding cost / yr",
    "idle_instock_days": "Idle days (in stock)",
    "days_since_pick": "Days since last pick",
    "last_pick_date": "Last pick",
    "last_receipt_date": "Last receipt",
    "stock_age_oldest": "Oldest stock (days)",
    "cover_days": "Cover (days)",
    "replenishment_conflict": "Still being replenished",
    "reason": "Why",
    "action": "Recommended action",
}


def read_table(data: bytes, filename: str, sheet: Optional[str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read an uploaded CSV / TSV / Excel file into a DataFrame (all columns as read, no cleaning)."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        xl = pd.ExcelFile(io.BytesIO(data))
        return xl.parse(sheet or xl.sheet_names[0], nrows=nrows)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            head = data[:8192].decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, nrows=nrows, low_memory=False)


def list_sheets(data: bytes, filename: str) -> List[str]:
    if filename.lower().endswith((".xlsx", ".xlsm")):
        return pd.ExcelFile(io.BytesIO(data)).sheet_names
    return []


def fmt_money(x: float, currency: str = "₹") -> str:
    """Compact money: lakh/crore for rupees, K/M/B otherwise."""
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if currency == "₹":
        if a >= 1e7:
            return f"{sign}₹{a / 1e7:,.2f} Cr"
        if a >= 1e5:
            return f"{sign}₹{a / 1e5:,.2f} L"
        return f"{sign}₹{a:,.0f}"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{sign}{currency}{a / div:,.2f}{suf}"
    return f"{sign}{currency}{a:,.0f}"


def friendly_table(result: pd.DataFrame) -> pd.DataFrame:
    """The columns a warehouse manager wants, with readable names (index = SKU)."""
    t = result[list(TABLE_COLUMNS)].copy()
    t["cover_days"] = t["cover_days"].replace([np.inf, -np.inf], np.nan)
    for c in ("last_pick_date", "last_receipt_date"):
        t[c] = pd.to_datetime(t[c]).dt.date
    t = t.rename(columns=TABLE_COLUMNS)
    t.index.name = "SKU"
    return t


def to_excel_bytes(result: pd.DataFrame, summary: Dict[str, float], cfg: Config, action_status: Optional[List[str]] = None) -> bytes:
    """A three-sheet workbook: action list, all SKUs, headline summary."""
    action_status = action_status or ["forgotten", "obsolete", "temporarily_inactive"]
    tbl = friendly_table(result.sort_values("value_at_risk", ascending=False))
    act = tbl[result.sort_values("value_at_risk", ascending=False)["status"].isin(action_status).to_numpy()]
    summ = pd.DataFrame({
        "Metric": ["SKUs analysed", "In stock", "Forgotten", "Obsolete", "Watch list", "Still being replenished",
                   f"Capital in dead stock ({cfg.currency})", "Share of all stock value", f"Holding cost / year ({cfg.currency})",
                   "Space in dead stock (m3)", f"Expired-stock exposure ({cfg.currency})"],
        "Value": [summary["n_skus"], summary["n_in_stock"], summary["n_forgotten"], summary["n_obsolete"], summary["n_watch"],
                  summary["n_conflict"], round(summary["capital_dead"], 2), f"{summary['capital_dead_share']:.1%}",
                  round(summary["holding_cost_dead"], 2), round(summary["space_dead_m3"], 2), round(summary["expiry_exposure"], 2)],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        act.to_excel(xw, sheet_name="Action list")
        tbl.to_excel(xw, sheet_name="All SKUs")
        summ.to_excel(xw, sheet_name="Summary", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                letter = col[0].column_letter
                width = max((len(str(c.value)) if c.value is not None else 0) for c in col[:200])
                ws.column_dimensions[letter].width = min(max(10, width + 2), 70)
    return buf.getvalue()
