"""Build a warehouse on REAL demand: download or read a public retail dataset, then put a supply side
and planted forgotten / obsolete / paused behaviour underneath it.

    # M5 Walmart (recommended: 5 years, enough for seasonality). Download from Kaggle first:
    #   kaggle competitions download -c m5-forecasting-accuracy   (needs a free Kaggle account)
    python scripts/prepare_real_demand.py --m5-dir path/to/m5 --store CA_1 --since 2013-01-01

    # Online Retail II (UCI, CC BY 4.0). Two years only: seasonality cannot be assessed after the hold-out.
    python scripts/prepare_real_demand.py --online-retail path/to/online_retail_II.xlsx
    python scripts/prepare_real_demand.py --download-online-retail      # tries the UCI URL

Outputs land in ``data/real/`` as ``sales.csv``, ``movements.csv.gz``, ``sku_master.csv`` and
``planted_truth.csv``.  Evaluate on it with:

    python scripts/run_evaluation.py --real-sales data/real/sales.csv --out reports/real

Note: this script needs internet access to the data source, so it is run on your own machine.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from fid.realdata import load_m5, load_online_retail, window
from fid.simulate import make_warehouse_from_sales

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5-dir", type=str)
    ap.add_argument("--store", type=str, default="CA_1")
    ap.add_argument("--online-retail", type=str)
    ap.add_argument("--download-online-retail", action="store_true")
    ap.add_argument("--since", type=str, default=None, help="drop sales before this date (e.g. 2013-01-01 for M5)")
    ap.add_argument("--n-future", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "real"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.m5_dir:
        sales, master = load_m5(a.m5_dir, a.store)
    elif a.online_retail or a.download_online_retail:
        path = a.online_retail
        if a.download_online_retail:
            print("downloading Online Retail II from UCI ...")
            blob = urllib.request.urlopen(UCI_URL, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".csv")))
                path = out / Path(name).name
                path.write_bytes(z.read(name))
        sales, master = load_online_retail(path)
    else:
        ap.error("give --m5-dir, --online-retail or --download-online-retail")

    sales = window(sales, a.since)
    print(f"{len(sales):,} sale rows, {sales['sku'].nunique():,} SKUs, {sales['date'].min().date()} to {sales['date'].max().date()}")
    wh = make_warehouse_from_sales(sales, seed=a.seed, n_future=a.n_future, currency_cost=master["unit_cost"])
    sku_master = master.reindex(wh.skus.index)
    sales.to_csv(out / "sales.csv", index=False)
    wh.movements.to_csv(out / "movements.csv.gz", index=False)
    sku_master.reset_index().to_csv(out / "sku_master.csv", index=False)
    wh.truth.reset_index().to_csv(out / "planted_truth.csv", index=False)
    print("planted:", wh.truth["truth_state"].value_counts().to_dict())
    print(f"as-of {wh.as_of.date()}; wrote {out}")


if __name__ == "__main__":
    main()
