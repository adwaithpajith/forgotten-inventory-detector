"""Run the full evaluation and write ``reports/evaluation.md``, ``reports/metrics.json`` and figures.

    python scripts/run_evaluation.py                 # 10,000-SKU synthetic warehouse
    python scripts/run_evaluation.py --n-skus 3000   # quicker
    python scripts/run_evaluation.py --real-sales sales.csv   # real demand underneath (see prepare_real_demand.py)

Everything in the report is computed here; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fid import ml
from fid.config import CLASS_COLORS, CLASS_LABELS, CLASSES, Config
from fid.evaluate import behavioural_backtest, lead_time_eval, planted_eval, threshold_sweep, unlabelled_tradeoff
from fid.simulate import DEFAULT_MIX, make_warehouse, make_warehouse_from_sales

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
ENGINE, NAIVE, NAIVE60 = "#2a6f97", "#c9a227", "#9aa5b1"


COUNT_COLS = {"tp", "fp", "fn", "support", "SKUs detected", "n_skus", "engine_flags", "naive_flags"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for c, v in zip(df.columns, row):
            if c in COUNT_COLS and pd.notna(v):
                cells.append(f"{int(round(float(v))):,}")
            elif c == "capital_tied" and pd.notna(v):
                cells.append(f"{float(v):,.0f}")
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_confusion(conf: pd.DataFrame, path: Path) -> None:
    cols = [c for c in conf.columns if conf[c].sum() > 0]
    norm = conf[cols].div(conf[cols].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [CLASS_LABELS.get(c, c).split(" (")[0] for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(norm)), [CLASS_LABELS.get(c, c).split(" (")[0] for c in norm.index])
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm.iat[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > 0.5 else "#222", fontsize=9)
    ax.set_xlabel("Engine says"); ax.set_ylabel("Planted truth")
    ax.set_title("Where the engine agrees with the planted truth (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(path); plt.close(fig)


def fig_dead(dead: dict, path: Path) -> None:
    names = [("naive", "Naive: no pick\nin 90 days", NAIVE), ("naive_same_idle", "Naive: no pick\nin 60 days", NAIVE60),
             ("engine", "Engine", ENGINE)]
    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    w = 0.26
    for k, (key, label, col) in enumerate(names):
        vals = [dead[key][m] for m in metrics]
        bars = ax.bar(np.arange(3) + (k - 1) * w, vals, w, label=label, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), ["Precision", "Recall", "F1"]); ax.set_ylim(0, 1.12)
    ax.set_title("Finding dead stock (forgotten + obsolete)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.savefig(path); plt.close(fig)


def fig_lead(lt: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    bins = np.arange(0, 135, 7.5)
    ax.hist(lt["naive_delays"], bins=bins, alpha=0.8, color=NAIVE, label="Naive: no pick in 90 days")
    ax.hist(lt["engine_delays"], bins=bins, alpha=0.8, color=ENGINE, label="Engine")
    ax.set_xlabel("Days from planted stop to first flag"); ax.set_ylabel("SKUs")
    ax.set_title("How early is a stopped SKU flagged?"); ax.legend(frameon=False)
    fig.savefig(path); plt.close(fig)


def fig_backtest(bt: dict, path: Path) -> None:
    t = bt["table"].iloc[[0, 3, 4, 5, 6, 7]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = [ENGINE, "#e9a23b", "#4c78a8", "#c9b458", "#2a9d8f", NAIVE]
    ax.barh(t["group"][::-1], t["stayed_dormant_pct"][::-1], color=cols[::-1])
    for y, v in enumerate(t["stayed_dormant_pct"][::-1]):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 110); ax.set_xlabel(f"% with no pick in the following {bt['horizon']} days")
    ax.set_title("Backtest: did flagged SKUs really stay dormant?")
    fig.savefig(path); plt.close(fig)


def fig_sweep(sw: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for pu, col in ((0.01, "#6b8ea8"), (0.05, ENGINE), (0.20, "#b0c4d4")):
        s = sw[sw["p_unusual"] == pu].sort_values("dead_min_idle_days")
        ax.plot(s["dead_min_idle_days"], s["precision"], "-o", color=col, label=f"precision, p<={pu}")
        ax.plot(s["dead_min_idle_days"], s["recall"], "--s", color=col, alpha=0.7)
    ax.set_xlabel("Minimum in-stock idle days before 'dead' can be called"); ax.set_ylabel("score")
    ax.set_title("Threshold sensitivity (dashed = recall)"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(path); plt.close(fig)


def fig_importance(imp: pd.Series, path: Path) -> None:
    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(top.index, top.values, color=ENGINE)
    ax.set_xlabel("Drop in macro-F1 when the feature is shuffled"); ax.set_title("ML second opinion: top features")
    fig.savefig(path); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skus", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--real-sales", type=str, default=None, help="CSV with sku,date,qty to put under the simulator")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--out", type=str, default=str(ROOT / "reports"))
    args = ap.parse_args()
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = Config()
    t0 = time.time()

    if args.real_sales:
        sales = pd.read_csv(args.real_sales, parse_dates=["date"])
        wh = make_warehouse_from_sales(sales, seed=args.seed)
    else:
        wh = make_warehouse(n_skus=args.n_skus, seed=args.seed)
    print(f"warehouse ready ({wh.source}); {len(wh.movements):,} ledger rows")

    ev = planted_eval(wh, cfg)
    lt = lead_time_eval(wh, cfg)
    sw = threshold_sweep(wh, cfg)
    real_mode = wh.source != "synthetic"
    planted_ids = wh.truth.index[wh.truth["truth_state"] != "unlabelled"] if real_mode else None
    bt = behavioural_backtest(wh.movements, wh.skus, cfg, wh.as_of, exclude=planted_ids)
    ut = unlabelled_tradeoff(wh, cfg) if real_mode else None
    print("core evaluation done")

    fig_confusion(ev["confusion"], out / "figures" / "confusion.png")
    fig_dead(ev["dead"], out / "figures" / "dead_detection.png")
    if lt.get("n"):
        fig_lead(lt, out / "figures" / "lead_time.png")
    if bt:
        fig_backtest(bt, out / "figures" / "backtest.png")
    fig_sweep(sw, out / "figures" / "threshold_sweep.png")

    ml_res = None
    if not args.no_ml:
        print("training ML second opinion ...")
        model = ml.train()
        (ROOT / "models").mkdir(exist_ok=True)
        ml.save(model, ROOT / "models" / "ml_classifier.joblib")
        ml_res = ml.evaluate(model, cfg=cfg)
        fig_importance(ml_res["importance"], out / "figures" / "ml_importance.png")

    # ----------------------------------------------------------------- metrics.json
    metrics = {
        "source": wh.source, "n_skus": int(len(wh.truth)), "as_of": str(wh.as_of.date()), "seed": args.seed,
        "n_evaluated": ev["n_evaluated"], "accuracy_6class": ev["accuracy"],
        "dead": ev["dead"], "per_class": ev["per_class"].round(4).to_dict("index"),
        "outcomes": ev["outcomes"], "seasonal_false_alarms": ev["seasonal_false_alarms"],
        "lead_time": {k: v for k, v in lt.items() if not k.endswith("delays")},
        "backtest": bt["table"].round(2).to_dict("records") if bt else None,
        "unlabelled_tradeoff": ut.round(3).to_dict("records") if ut is not None else None,
        "ml": {k: v for k, v in (ml_res or {}).items() if k != "importance"},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ----------------------------------------------------------------- report
    d = ev["dead"]
    truth_counts = wh.truth["truth_state"].value_counts()
    L = []
    L.append("# Evaluation report\n")
    L.append(f"*Generated by `scripts/run_evaluation.py` in {time.time() - t0:.0f}s. Every number below is computed, not typed.*\n")
    L.append("## Setup\n")
    kind = ("a **simulated** warehouse (synthetic demand archetypes, reorder-point replenishment)"
            if wh.source == "synthetic" else "**real sales history** with planted forgotten / obsolete / paused behaviour")
    L.append(f"- Data: {kind}; {len(wh.truth):,} SKUs, as-of date {wh.as_of.date()}, seed {args.seed}.")
    L.append(f"- Ledger: {len(wh.movements):,} movements up to the as-of date, plus {len(wh.future):,} held-out movements after it.")
    L.append("- Planted truth on the as-of date: " + ", ".join(f"{k} {v:,}" for k, v in truth_counts.items()) + ".")
    L.append(f"- Scored: {ev['n_evaluated']:,} in-stock SKUs ({ev['n_out_of_stock']:,} out-of-stock excluded; too-new SKUs excluded from the class scores).")
    L.append("- Default thresholds (`fid/config.py`) were set from first principles and a first look at one simulation; they are "
             "**not** fitted to optimise the scores below. Change them in the app's sidebar.\n")
    if wh.source == "synthetic":
        L.append("> **Read this first.** On simulated data the planted truth is the generator's own labels, so these scores measure whether the "
                 "engine recovers the behaviours it was designed to separate, not how it will fare on a real warehouse. "
                 "Use the label-free backtest (section 4) on real data.\n")
    else:
        L.append("> **Read this first.** Demand here is real, but only the *planted* SKUs have labels. Precision/recall below are computed over "
                 "those planted SKUs alone, so false alarms on the unlabelled real SKUs (where a naive rule tends to misfire on slow and seasonal "
                 "items) are not counted in them. Section 1 therefore adds a label-free comparison on the unlabelled SKUs, and section 4 "
                 "(the backtest) is the fairer test.\n")

    L.append("## 1. Finding dead stock (forgotten + obsolete)\n")
    t1 = pd.DataFrame({
        "Naive: no pick in 90 days": d["naive"], "Naive: no pick in 60 days": d["naive_same_idle"], "Engine": d["engine"],
    }).T[["precision", "recall", "f1", "tp", "fp", "fn"]]
    t1.index.name = "Method"
    L.append(md_table(t1) + "\n")
    L.append("![dead detection](figures/dead_detection.png)\n")
    o = ev["outcomes"]
    sf = ev["seasonal_false_alarms"]
    L.append(f"- Of the SKUs the engine flagged dead, **{o['engine_dead']['share_picked_again']:.1%}** were picked again in the held-out future; "
             f"for the naive 90-day rule it was **{o['naive']['share_picked_again']:.1%}** ({o['naive']['flagged']:,} flagged).")
    if sf["n_seasonal"]:
        L.append(f"- Seasonal SKUs wrongly called dead: engine **{sf['engine']}**, naive 90-day rule **{sf['naive']}** (of {sf['n_seasonal']:,} seasonal SKUs).")
    de, dn, ds = d["engine"], d["naive"], d["naive_same_idle"]
    if de["precision"] > ds["precision"] + 0.02:
        L.append(f"- At the *same* {cfg.dead_min_idle_days}-day idle threshold the naive rule has precision {ds['precision']:.2f} against the "
                 f"engine's {de['precision']:.2f} (recall {ds['recall']:.2f} vs {de['recall']:.2f}), so here the gain comes from testing each SKU "
                 "against its own rhythm and its seasonality, not from flagging earlier.")
    else:
        L.append(f"- At the same {cfg.dead_min_idle_days}-day idle threshold the engine's precision ({de['precision']:.2f}) is not clearly "
                 f"better than the naive rule's ({ds['precision']:.2f}) on the labelled SKUs.")
    if dn["precision"] > de["precision"]:
        L.append(f"- Against the naive **{cfg.naive_days}-day** rule the engine's precision on the labelled SKUs is lower "
                 f"({de['precision']:.2f} vs {dn['precision']:.2f}) because it flags earlier; see the label-free comparison below.")
    if "engine_dead_unlabelled" in o:
        eu, nu, ns_ = o["engine_dead_unlabelled"], o["naive_unlabelled"], o["naive_same_idle_unlabelled"]
        L.append(f"- **Unlabelled real SKUs (label-free):** the engine flagged {eu['flagged']:,} as dead, of which "
                 f"{eu['share_picked_again']:.1%} were picked again in the held-out window; the naive {cfg.naive_days}-day rule flagged "
                 f"{nu['flagged']:,} ({nu['share_picked_again']:.1%} picked again) and the naive {cfg.dead_min_idle_days}-day rule "
                 f"{ns_['flagged']:,} ({ns_['share_picked_again']:.1%} picked again). These SKUs keep their real demand, so a high share "
                 "picked again means most of those flags are false alarms; see section 6 for the trade-off.")
    L.append("")

    L.append("## 2. Six-class behaviour\n")
    L.append(f"Overall accuracy on the six classes: **{ev['accuracy']:.1%}**"
             + (" (planted SKUs only; unlabelled real SKUs have no true class).\n" if real_mode else ".\n"))
    L.append(md_table(ev["per_class"]) + "\n")
    L.append("![confusion](figures/confusion.png)\n")
    L.append("Honest reading of the confusions:\n")
    L.append("- *Temporarily inactive → forgotten.* A fast mover silent for 60+ in-stock days looks exactly like a dead one at that moment; "
             "only the future tells them apart. The engine deliberately surfaces these for investigation. This is the main source of false alarms.")
    L.append("- *Slow-moving → forgotten / watch.* With p<=0.05 about one in twenty healthy slow movers is flagged by chance. "
             "The threshold sweep (section 5) shows the trade-off.")
    L.append("- *Obsolete → forgotten.* Both are dead, so the action is the same; the fade is sometimes too noisy to see.")
    L.append("- *Seasonal → active.* The engine reacts after a trough has started, so some SKUs are still called active early in the off-season.\n")

    L.append("## 3. How early is a stopped SKU caught?\n")
    if lt.get("n"):
        t3 = pd.DataFrame({k: lt[k] for k in ("naive", "naive_same_idle", "engine")}).T
        t3.index = ["Naive: no pick in 90 days", "Naive: no pick in 60 days", "Engine"]; t3.index.name = "Method"
        t3 = t3.rename(columns={"detected": "SKUs detected", "share": "share of stopped SKUs", "median_days": "median days after stop", "p90_days": "90th pct days"})
        L.append(f"{lt['n']:,} SKUs whose demand stopped abruptly; checked every {lt['step_days']} days (so delays have ±{lt['step_days']}-day resolution).\n")
        L.append(md_table(t3) + "\n")
        L.append("![lead time](figures/lead_time.png)\n")
        ee, nn, ss = lt["engine"], lt["naive"], lt["naive_same_idle"]
        L.append(f"Median delay: engine {ee['median_days']:.0f} days, naive {cfg.naive_days}-day rule {nn['median_days']:.0f} days, "
                 f"naive {cfg.dead_min_idle_days}-day rule {ss['median_days']:.0f} days. Whatever lead the engine has over the "
                 f"{cfg.naive_days}-day rule comes from its lower idle threshold ({cfg.dead_min_idle_days} in-stock days); "
                 f"it is not earlier than a naive rule set to that same threshold.\n")
    else:
        L.append("No abruptly stopped SKUs in this dataset.\n")

    L.append("## 4. Label-free backtest (works on any real data)\n")
    if bt:
        L.append(f"Classify as of {bt['t0'].date()}, then check what happened in the next {bt['horizon']} days. No labels needed."
                 + (" Planted SKUs are excluded, so only untouched real behaviour is judged." if real_mode else "") + "\n")
        tb = bt["table"].set_index("group")[["n_skus", "stayed_dormant_pct", "picked_again_pct", "capital_tied"]]
        tb.index.name = "Group at the earlier date"
        L.append(md_table(tb, "{:,.1f}") + "\n")
        L.append("![backtest](figures/backtest.png)\n")
        L.append(f"- The naive rule flagged {bt['naive_only_flagged']:,} SKUs the engine did not; "
                 f"**{bt['naive_false_alarms_avoided']:,}** of them were picked again, i.e. false alarms the engine avoided.\n")
    else:
        L.append("Not enough history for a backtest.\n")

    L.append("## 5. Threshold sensitivity\n")
    sw_t = sw.assign(setting=[f"p<={a}, idle>={b}d" for a, b in zip(sw["p_unusual"], sw["dead_min_idle_days"])]).set_index("setting")
    L.append(md_table(sw_t[["precision", "recall", "f1"]]) + "\n")
    L.append("![sweep](figures/threshold_sweep.png)\n")
    L.append("**Do not read the 90-day row as an optimum.** Planted pauses in this simulation last up to 80 idle days, so any threshold above 80 "
             "trivially removes them. In a real warehouse the right value depends on how long your products legitimately sit quiet; "
             "use the backtest on your own data to choose.\n")

    if ml_res:
        L.append("## 6. ML second opinion (trained on simulated warehouses only)\n")
        t6 = pd.DataFrame({"Engine": [ml_res["engine_accuracy"], ml_res["engine_macro_f1"], ml_res["engine_dead_f1"]],
                           "Gradient boosting": [ml_res["ml_accuracy"], ml_res["ml_macro_f1"], ml_res["ml_dead_f1"]]},
                          index=["Accuracy (6 classes)", "Macro-F1", "Dead-vs-alive F1"])
        t6.index.name = "Metric"
        L.append(f"Fresh simulated warehouse never seen in training ({ml_res['n_test']:,} SKUs).\n")
        L.append(md_table(t6) + "\n")
        L.append("![importance](figures/ml_importance.png)\n")
        L.append("The model learns the *simulator's* labels from the engine's features, so a score above the engine here shows it fits "
                 "that generator, not that it will beat the engine on your data. It ships as an optional, clearly labelled second view; "
                 "disagreement between the two is a useful prompt to look at a SKU by hand.\n")

    if ut is not None:
        L.append("## 6. False alarms on untouched real SKUs vs planted recall\n")
        L.append("Idle threshold swept for both the engine and the naive rule. `per_100` = dead flags per 100 unlabelled in-stock SKUs "
                 "(nearly all are alive, so these are mostly false alarms); `picked_again` = share of those flags picked again in the held-out window; "
                 "`recall` = share of planted dead SKUs caught. The better method gives fewer false alarms at the same recall.\n")
        tu = ut.set_index("idle_days")
        tu.index.name = "idle days"
        tu = tu[["engine_flags", "engine_per_100", "engine_picked_again_pct", "engine_planted_recall",
                 "naive_flags", "naive_per_100", "naive_picked_again_pct", "naive_planted_recall"]]
        L.append(md_table(tu, "{:,.2f}") + "\n")
    L.append("## Limitations\n")
    L.append("- Synthetic truth is the generator's own; real warehouses have behaviours not simulated here (promotions, "
             "kitting, customer-specific stock, data-entry errors, multi-location transfers).")
    L.append("- Demand is counted in pick *days*, not order lines; a single huge order looks like one pick.")
    L.append("- Seasonality needs about two years of history; with less, strongly seasonal SKUs can be mistaken for neglected ones (the app warns).")
    L.append("- Money figures use the unit cost you supply; without it the app reports units (cost 1.0).")
    (out / "evaluation.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'evaluation.md'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()