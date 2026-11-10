"""Input handling: column mapping, cleaning, stock reconciliation, data-quality report.

The engine works on two tables:

``movements``  one row per stock event
    sku, date, kind ('receipt' | 'issue'), qty (> 0)

``skus``  optional master data, indexed by sku
    description, category, unit_cost, shelf_life_days, volume_per_unit,
    location, on_hand_qty, first_receipt_date, successor_sku

Only ``movements`` is required.  Everything else improves the answer but has a
sensible default.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

MOVEMENT_ALIASES: Dict[str, List[str]] = {
    "sku": ["sku", "item", "item_id", "itemid", "item_code", "product", "product_id",
            "productid", "product_code", "stockcode", "stock_code", "article", "part_number", "material"],
    "date": ["date", "timestamp", "datetime", "invoicedate", "invoice_date", "posting_date",
             "transaction_date", "movement_date", "txn_date", "created_at", "time"],
    "kind": ["kind", "type", "event", "event_type", "movement_type", "transaction_type",
             "txn_type", "direction", "movement", "mvmt_type", "mvt_type", "movement_code", "mvmt_code"],
    "qty": ["qty", "quantity", "units", "amount", "qty_moved", "quantity_moved", "moved_qty"],
}

SKU_ALIASES: Dict[str, List[str]] = {
    "sku": MOVEMENT_ALIASES["sku"],
    "description": ["description", "name", "item_name", "product_name", "desc", "title"],
    "category": ["category", "family", "product_group", "class", "dept", "department", "group", "cat"],
    "unit_cost": ["unit_cost", "cost", "unitcost", "cost_price", "standard_cost", "std_cost", "price", "unit_price"],
    "on_hand_qty": ["on_hand_qty", "on_hand", "onhand", "stock", "stock_qty", "qty_on_hand",
                    "quantity_on_hand", "current_stock", "stock_on_hand", "soh"],
    "shelf_life_days": ["shelf_life_days", "shelf_life", "shelflife", "expiry_days", "life_days"],
    "volume_per_unit": ["volume_per_unit", "volume", "unit_volume", "cbm", "m3", "space_per_unit"],
    "location": ["location", "bin", "slot", "shelf", "bin_location", "storage_location", "bin_id"],
    "successor_sku": ["successor_sku", "replaced_by", "successor", "superseded_by", "replacement_sku"],
    "first_receipt_date": ["first_receipt_date", "received_date", "first_received", "intro_date"],
}

RECEIPT_WORDS = {"receipt", "receive", "received", "receiving", "inbound", "in", "purchase", "grn",
                 "putaway", "put_away", "goods_in", "po_receipt", "replenishment", "goods_receipt", "r", "+"}
ISSUE_WORDS = {"issue", "issued", "pick", "picked", "picking", "outbound", "out", "sale", "sold",
               "sales", "dispatch", "dispatched", "shipped", "ship", "shipment", "goods_out", "goods_issue", "gi", "consumption", "i", "-"}

SKU_COLUMNS = ["description", "category", "unit_cost", "shelf_life_days", "volume_per_unit",
               "location", "on_hand_qty", "first_receipt_date", "successor_sku"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


_CURRENCY_SUFFIX = re.compile(r"_(inr|usd|eur|gbp|aud|cad|jpy|cny|rs)$")


def _norm_col(name: str) -> str:
    """Normalise a column header and drop a trailing currency code (``Std Cost (INR)`` -> ``std_cost``)."""
    return _CURRENCY_SUFFIX.sub("", _norm(name))


def suggest_mapping(columns: Iterable[str], aliases: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """Guess which raw column maps to each standard field (case/spacing insensitive)."""
    by_norm = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    for c in columns:
        by_norm.setdefault(_norm_col(c), c)
    out: Dict[str, Optional[str]] = {}
    used = set()
    for field, names in aliases.items():
        hit = None
        for nm in names:
            c = by_norm.get(nm)
            if c is not None and c not in used:
                hit = c
                break
        out[field] = hit
        if hit is not None:
            used.add(hit)
    return out


def guess_kind_values(values: Iterable) -> Tuple[List[str], List[str]]:
    """Split distinct raw type values into (receipt-like, issue-like) lists."""
    rec, iss = [], []
    for v in pd.unique(pd.Series(list(values)).dropna().astype(str)):
        n = _norm(v)
        if n in RECEIPT_WORDS:
            rec.append(v)
        elif n in ISSUE_WORDS:
            iss.append(v)
    return rec, iss


def standardize_movements(
    raw: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    mode: str = "type_column",
    receipt_values: Optional[Iterable[str]] = None,
    issue_values: Optional[Iterable[str]] = None,
    dayfirst: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Clean a raw movement table into the standard schema.

    ``mode`` is one of:

    * ``"type_column"``  a column says whether each row is a receipt or an issue;
    * ``"signed"``       positive quantity = receipt, negative = issue;
    * ``"issues_only"``  every positive row is a pick/sale (needs ``on_hand_qty``
      in the SKU master to know the stock position).
    """
    report: dict = {"rows_in": int(len(raw)), "notes": []}
    for need in ("sku", "date", "qty"):
        if not mapping.get(need) or mapping[need] not in raw.columns:
            raise ValueError(f"Missing required column mapping for '{need}'.")
    df = pd.DataFrame({
        "sku": raw[mapping["sku"]].astype(str).str.strip().str.upper(),
        "date": pd.to_datetime(raw[mapping["date"]], errors="coerce", dayfirst=dayfirst),
        "qty": pd.to_numeric(raw[mapping["qty"]], errors="coerce"),
    })
    if mode == "type_column":
        if not mapping.get("kind") or mapping["kind"] not in raw.columns:
            raise ValueError("Mode 'type_column' needs a mapped movement-type column.")
        df["raw_kind"] = raw[mapping["kind"]].astype(str).str.strip()
        if receipt_values is None or issue_values is None:
            auto_r, auto_i = guess_kind_values(df["raw_kind"].unique())
            receipt_values = auto_r if receipt_values is None else receipt_values
            issue_values = auto_i if issue_values is None else issue_values
        rset, iset = set(map(str, receipt_values)), set(map(str, issue_values))
        kind = np.where(df["raw_kind"].isin(rset), "receipt",
                        np.where(df["raw_kind"].isin(iset), "issue", ""))
        df["kind"] = kind
        df["qty"] = df["qty"].abs()
        df = df.drop(columns="raw_kind")
    elif mode == "signed":
        df["kind"] = np.where(df["qty"] > 0, "receipt", np.where(df["qty"] < 0, "issue", ""))
        df["qty"] = df["qty"].abs()
    elif mode == "issues_only":
        neg = int((df["qty"] < 0).sum())
        if neg:
            report["notes"].append(f"{neg} negative rows (returns/credits) were ignored in sales-only mode.")
        df["kind"] = np.where(df["qty"] > 0, "issue", "")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    bad_date = int(df["date"].isna().sum())
    bad_qty = int(df["qty"].isna().sum())
    unknown = int(((df["kind"] == "") & df["date"].notna() & df["qty"].notna()).sum())
    zero = int(((df["qty"] == 0) & (df["kind"] != "")).sum())
    df = df[df["date"].notna() & df["qty"].notna() & (df["kind"] != "") & (df["qty"] > 0)].copy()
    df["date"] = df["date"].dt.normalize()
    df = df[(df["sku"] != "") & (df["sku"].str.lower() != "nan")]
    df = df.groupby(["sku", "date", "kind"], as_index=False)["qty"].sum()
    df = df.sort_values(["date", "sku"], kind="stable").reset_index(drop=True)

    report.update({
        "rows_dropped_bad_date": bad_date,
        "rows_dropped_bad_qty": bad_qty,
        "rows_dropped_unknown_type": unknown,
        "rows_dropped_zero_qty": zero,
        "rows_kept": int(len(df)),
        "n_skus": int(df["sku"].nunique()),
        "date_min": df["date"].min() if len(df) else None,
        "date_max": df["date"].max() if len(df) else None,
        "n_receipts": int((df["kind"] == "receipt").sum()),
        "n_issues": int((df["kind"] == "issue").sum()),
    })
    if unknown:
        report["notes"].append(f"{unknown} rows had a movement type that was not mapped to receipt or issue and were ignored.")
    return df, report


def standardize_skus(raw: Optional[pd.DataFrame], mapping: Optional[Dict[str, Optional[str]]] = None) -> pd.DataFrame:
    """Clean a raw SKU master into a table indexed by ``sku``."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=SKU_COLUMNS, index=pd.Index([], name="sku"), dtype=object)
    mapping = mapping or suggest_mapping(raw.columns, SKU_ALIASES)
    if not mapping.get("sku") or mapping["sku"] not in raw.columns:
        raise ValueError("SKU master needs a column mapped to 'sku'.")
    out = pd.DataFrame({"sku": raw[mapping["sku"]].astype(str).str.strip().str.upper()})
    for col in SKU_COLUMNS:
        src = mapping.get(col)
        out[col] = raw[src].to_numpy() if src and src in raw.columns else np.nan
    for col in ("unit_cost", "shelf_life_days", "volume_per_unit", "on_hand_qty"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["first_receipt_date"] = pd.to_datetime(out["first_receipt_date"], errors="coerce").dt.normalize()
    out["successor_sku"] = out["successor_sku"].map(lambda v: str(v).strip().upper() if pd.notna(v) and str(v).strip() else np.nan)
    out = out.drop_duplicates("sku", keep="last").set_index("sku")
    return out


def ensure_sku_table(skus: Optional[pd.DataFrame], movements: pd.DataFrame) -> pd.DataFrame:
    """Return a SKU table covering every SKU seen in movements or the master."""
    base = skus.copy() if skus is not None else pd.DataFrame(index=pd.Index([], name="sku"))
    for col in SKU_COLUMNS:
        if col not in base.columns:
            base[col] = np.nan
    all_ids = pd.Index(sorted(set(base.index.astype(str)) | set(movements["sku"].astype(str))), name="sku")
    base.index = base.index.astype(str)
    base = base.reindex(all_ids)
    return base[SKU_COLUMNS]


def reconcile(
    movements: pd.DataFrame, skus: pd.DataFrame, as_of: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Make the ledger agree with an optional ``on_hand_qty`` snapshot.

    If the master says more stock is on hand than the ledger implies (always the
    case in sales-only data) we add an *opening balance receipt* for the
    difference, dated at ``first_receipt_date`` if known, otherwise at the start
    of the data.  The assumption is reported to the user.
    """
    notes: List[str] = []
    skus = ensure_sku_table(skus, movements)
    mv = movements[movements["date"] <= as_of]
    rec = mv[mv["kind"] == "receipt"].groupby("sku")["qty"].sum()
    iss = mv[mv["kind"] == "issue"].groupby("sku")["qty"].sum()
    ledger = rec.reindex(skus.index).fillna(0.0) - iss.reindex(skus.index).fillna(0.0)

    oh = skus["on_hand_qty"]
    has = oh.notna()
    if not has.any():
        return movements, skus, notes

    need = (oh - ledger).where(has, 0.0)
    add_mask = has & (need > 0.5)
    lower = int((has & (need < -0.5)).sum())
    if lower:
        notes.append(f"{lower} SKUs have a master on-hand figure below the ledger balance; the ledger was kept for those.")
    if add_mask.any():
        first_seen = mv.groupby("sku")["date"].min().reindex(skus.index)
        global_start = mv["date"].min() if len(mv) else as_of
        fr = skus["first_receipt_date"]
        when = fr.where(fr.notna(), first_seen).fillna(global_start)
        when = when.where(when <= as_of, as_of)
        extra = pd.DataFrame({
            "sku": skus.index[add_mask].to_numpy(),
            "date": when[add_mask].to_numpy(),
            "kind": "receipt",
            "qty": need[add_mask].to_numpy(),
        })
        n_assumed = int((~fr[add_mask].notna()).sum())
        notes.append(
            f"Added opening-balance receipts for {int(add_mask.sum())} SKUs to match the on-hand snapshot; "
            f"{n_assumed} of them were dated at the start of the data because no first-receipt date was given "
            "(stock age for those SKUs is a lower bound)."
        )
        movements = pd.concat([movements, extra], ignore_index=True)
        movements["date"] = pd.to_datetime(movements["date"])
        movements = movements.sort_values(["date", "sku"], kind="stable").reset_index(drop=True)
    return movements, skus, notes


def data_quality(movements: pd.DataFrame, skus: pd.DataFrame, as_of: pd.Timestamp) -> List[str]:
    """Human-readable warnings the user should see before trusting results."""
    w: List[str] = []
    if len(movements) == 0:
        return ["No usable movement rows."]
    span = (movements["date"].max() - movements["date"].min()).days
    if span < 180:
        w.append(f"Only {span} days of history: most SKUs will be 'too new to judge' and seasonality cannot be assessed.")
    elif span < 700:
        w.append(f"{span} days of history: seasonality needs about two years, so strongly seasonal SKUs may be "
                 "mistaken for neglected ones. Treat 'forgotten' flags on seasonal categories with care.")
    lag = (pd.Timestamp(as_of) - movements["date"].max()).days
    if lag > 14:
        w.append(f"The as-of date is {lag} days after the last movement; stock may have moved since the extract.")
    mv = movements[movements["date"] <= as_of]
    rec = mv[mv["kind"] == "receipt"].groupby("sku")["qty"].sum()
    iss = mv[mv["kind"] == "issue"].groupby("sku")["qty"].sum()
    ledger = rec.reindex(skus.index).fillna(0.0) - iss.reindex(skus.index).fillna(0.0)
    neg = int((ledger < -0.5).sum())
    if neg:
        w.append(f"{neg} SKUs issued more units than they received in this extract (history before the extract is missing). "
                 "Provide an on-hand snapshot or a longer history for those SKUs.")
    if skus["unit_cost"].isna().all():
        w.append("No unit costs supplied: financial figures are shown in units (cost assumed 1.0 per unit).")
    elif skus["unit_cost"].isna().any():
        w.append(f"{int(skus['unit_cost'].isna().sum())} SKUs have no unit cost; cost assumed 1.0 for them.")
    if skus["shelf_life_days"].isna().all():
        w.append("No shelf-life data: expiry risk is not assessed.")
    return w
