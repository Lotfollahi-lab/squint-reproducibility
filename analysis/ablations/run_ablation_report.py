#!/usr/bin/env python
"""
ONE self-contained entry-point for the SQUINT ablation report.

    multiseed per_seed_*.csv  ─▶  all axis figures  +  one combined metrics CSV

It reads the multi-seed per-seed metric files DIRECTLY, renders every ablation
axis figure (reusing `plots/plot_ablations.py`'s axis specs + renderer so the
figures are byte-for-byte the paper figures), and writes one combined metrics
CSV (long + wide, with per-seed mean / sem / 95%-CI / significance-vs-default,
plus the discrete-vs-continuous comparison folded in as a pseudo-axis).

WHY a rewrite (the footgun this kills): there used to be THREE scripts with two
INCOMPATIBLE "summary" formats —
  * compare_variants.py        -> summary_long.csv          (variant + `table`)
  * summarize_ablation_multiseed.py -> ablation_summary_long.csv (variant + prefix
                                   + metric + mean/std/sem/ci95/p_vs_ref/stars)
`plot_ablations.py` consumes the FIRST; the multiseed summarizer writes the
SECOND. Chaining the summarizer into the plotter therefore failed with
"summary_long.csv must have 'variant' and 'table' columns". This script removes
the intermediary entirely: it builds the `variant`+`table` long frame in memory
straight from the per-seed CSVs, so there is no filename / schema to mismatch.

Inputs (per variant prefix referenced by the axis specs):
    <artifacts>/<dataset>/<prefix>*__multiseed/<latest_TS>/metrics/
        per_seed_niche_identification.csv   (split, code_key, label_key, NMI, ARI)
        per_seed_batch_integration.csv      (emb_key, metric, score)

Outputs (to --out-dir, default <artifacts>/<dataset>/ablations):
    axis_*.{svg,png}                  one figure per ablation axis
    axis_*.csv                        per-axis point values (parity w/ old plotter)
    ablations_all_metrics_long.csv    every (axis, variant, metric): value, n,
                                      mean, std, sem, ci95, p_vs_default, stars
    ablations_all_metrics_wide.csv    table-ready: variants x metrics (default first)

Usage:
    python analysis/ablations/run_ablation_report.py
    python analysis/ablations/run_ablation_report.py --dataset chl59-2b_1p
    python analysis/ablations/run_ablation_report.py --dry-run   # resolve dirs only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Reuse the existing axis specs + renderer (plot_ablations) and the proven
# multiseed-dir resolver + stats helpers (summarize_ablation_multiseed). We
# IMPORT them so the figures + significance math stay identical to the
# standalone tools — this script only adds the glue.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "plots"))
import plot_ablations as pa                      # noqa: E402
import summarize_ablation_multiseed as sm        # noqa: E402
import aggregate_ablation_csvs as agg            # noqa: E402

DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"


# ---------------------------------------------------------------------------
# Build the variant+table long frame straight from the multiseed per_seed CSVs
# ---------------------------------------------------------------------------
def _canonicalize_code_keys(d: pd.DataFrame) -> pd.DataFrame:
    """Map a single-level (L=1) variant's BARE code_key to the canonical
    `..._code_indices[level_0]` key the axis specs filter on. Mirrors
    summarize_ablation_multiseed._effective_code_key so the residual-VQ-levels
    axis (s57_v30/31/32) resolves instead of showing n/a."""
    if "code_key" not in d.columns:
        return d
    keys = set(d["code_key"].astype(str).unique())
    out = d
    for canon, stem in ((pa.CELL_CODE_KEY, "cell_code_ind"),
                        (pa.NICHE_CODE_KEY, "neighborhood_code_ind")):
        if canon in keys:
            continue
        bare = sorted(k for k in keys
                      if k.startswith(stem) and "[" not in k)
        if bare:
            out = out.copy()
            out.loc[out["code_key"].astype(str) == bare[0], "code_key"] = canon
    return out


def build_long_df(prefixes, artifacts_root, dataset):
    """Concatenate every variant's per-seed niche-ID + batch-integration rows
    into ONE long frame tagged with `variant` (the multiseed dir name) and
    `table`, i.e. exactly the schema plot_ablations._extract_metric expects."""
    rows = []
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for prefix in prefixes:
        md = sm._resolve_metrics_dir(prefix, artifacts_root, dataset)
        if md is None:
            missing.append(prefix)
            continue
        full = md.parent.parent.name          # <prefix>...__multiseed
        resolved[prefix] = full

        nf = md / "per_seed_niche_identification.csv"
        if nf.is_file():
            d = pd.read_csv(nf)
            if "split" not in d.columns:       # _extract_metric filters split=="all"
                d["split"] = "all"
            d = _canonicalize_code_keys(d)
            d["variant"] = full
            d["table"] = "niche_identification"
            rows.append(d)

        bf = md / "per_seed_batch_integration.csv"
        if bf.is_file():
            d = pd.read_csv(bf)
            d["variant"] = full
            d["table"] = "batch_integration"
            rows.append(d)

    if not rows:
        raise SystemExit(
            "No multiseed per_seed_*.csv found for any axis prefix under "
            f"{artifacts_root}/{dataset}. Checked prefixes: {prefixes}")
    return pd.concat(rows, ignore_index=True, sort=False), resolved, missing


def _metric_array(df, full_variant, metric, cell_label_key, niche_label_key):
    """Per-seed values for (variant, metric) — the array whose .mean() equals
    plot_ablations._extract_metric. Used for sem / CI / significance."""
    if metric in ("Cell NMI", "Cell ARI", "Niche NMI", "Niche ARI"):
        is_cell = metric.startswith("Cell")
        code_key = pa.CELL_CODE_KEY if is_cell else pa.NICHE_CODE_KEY
        label_key = cell_label_key if is_cell else niche_label_key
        if label_key is None:
            return np.array([])
        sub = df[(df.get("table") == "niche_identification")
                 & (df["variant"] == full_variant)
                 & (df.get("split") == "all")
                 & (df.get("code_key") == code_key)
                 & (df.get("label_key") == label_key)]
        col = "NMI" if metric.endswith("NMI") else "ARI"
        if sub.empty or col not in sub.columns:
            return np.array([])
        return pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()

    if metric in ("Cell iLISI", "Cell MMD", "Niche iLISI", "Niche MMD"):
        is_cell = metric.startswith("Cell")
        emb_key = pa.CELL_EMB_KEY if is_cell else pa.NICHE_EMB_KEY
        tag = "iLISI" if metric.endswith("iLISI") else "MMD"
        sub = df[(df.get("table") == "batch_integration")
                 & (df["variant"] == full_variant)
                 & (df.get("emb_key") == emb_key)
                 & (df.get("metric") == tag)]
        if sub.empty or "score" not in sub.columns:
            return np.array([])
        return pd.to_numeric(sub["score"], errors="coerce").dropna().to_numpy()

    raise ValueError(f"unknown metric {metric!r}")


# ---------------------------------------------------------------------------
# Discretization (discrete-codes vs clustered-embeddings) pseudo-axis. The
# producer (analysis/discretization_ablation/compare_discrete_vs_continuous.py)
# writes BOTH a per-seed file (richer: lets us recompute sem/CI/significance in
# the SAME unified schema as the real axes) and a pre-aggregated summary. Prefer
# the per-seed file; fall back to the summary.
# ---------------------------------------------------------------------------
DISC_DEFAULT_CONDITION = "discrete codes"   # COND_CODES, the significance baseline


def _resolve_discretization(out_dir: Path, explicit):
    """Locate a discretization CSV. Prefers `discretization_per_seed.csv` over
    `discretization_summary.csv`, and checks the ablations dir itself (the new
    location) before the original `*/comparison_vs_discrete/` dirs."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    names = ("discretization_per_seed.csv", "discretization_summary.csv")
    for nm in names:                                   # 1. directly in out_dir
        if (out_dir / nm).is_file():
            return out_dir / nm
    cands = []                                         # 2. comparison_vs_discrete dirs
    for root in (out_dir.parent, out_dir.parent.parent):
        if root.is_dir():
            for nm in names:
                cands += list(root.glob(f"*/*/comparison_vs_discrete/{nm}"))
                cands += list(root.glob(f"*/comparison_vs_discrete/{nm}"))
    cands = list(dict.fromkeys(cands))
    per_seed = [c for c in cands if c.name == "discretization_per_seed.csv"]
    pool = per_seed or cands
    return max(pool, key=lambda p: p.stat().st_mtime) if pool else None


def _load_discretization(path: Path) -> pd.DataFrame:
    """Load a discretization CSV into the unified combined-metrics schema
    (axis/variant/label/is_default/metric/value + n/mean/std/sem/ci95/
    p_vs_default/stars). Per-seed input -> full stats + significance vs the
    "Discrete codes" default; summary input -> value/std/n only (no per-seed
    array to test). `metric` is normalised to "Cell NMI" / "Niche iLISI" / ..."""
    d = pd.read_csv(path)
    cols = set(d.columns)
    is_per_seed = {"condition", "branch", "metric", "value"}.issubset(cols) \
        and "seed_idx" in cols and "mean" not in cols
    if not is_per_seed:
        # pre-aggregated summary: defer to the (already-validated) aggregate loader.
        return agg._load_discretization(path)

    d = d.copy()
    d["branch"] = d["branch"].astype(str).str.strip().str.capitalize()  # cell->Cell
    d["metric_full"] = d["branch"] + " " + d["metric"].astype(str).str.strip()
    conds = list(dict.fromkeys(d["condition"].astype(str)))
    default_cond = next(
        (c for c in conds if c.strip().lower() == DISC_DEFAULT_CONDITION), conds[0])
    metric_fulls = list(dict.fromkeys(d["metric_full"]))

    def _arr(cond, mf):
        sub = d[(d["condition"].astype(str) == cond) & (d["metric_full"] == mf)]
        return pd.to_numeric(sub["value"], errors="coerce").dropna().to_numpy()

    rows = []
    for cond in conds:
        is_def = (cond == default_cond)
        for mf in metric_fulls:
            a = _arr(cond, mf)
            if a.size == 0:
                continue
            n = int(a.size)
            std = float(a.std(ddof=1)) if n > 1 else 0.0
            sem = float(std / np.sqrt(n)) if n > 1 else 0.0
            pval = (float("nan") if is_def
                    else sm._pvalue(_arr(default_cond, mf), a, "ttest"))
            rows.append({
                "axis": agg.DISCRETIZATION_AXIS, "prefix": "", "variant": cond,
                "label": cond, "is_default": is_def, "metric": mf,
                "value": float(a.mean()), "n": n, "mean": float(a.mean()),
                "std": std, "sem": sem, "ci95": sm._ci95(a),
                "p_vs_default": pval, "stars": "" if is_def else sm._stars(pval),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--out-dir", default=None,
                   help="Figures + per-axis CSVs + combined metrics CSV "
                        "(default <artifacts>/<dataset>/ablations).")
    p.add_argument("--cell-label-keys", default=",".join(pa.DEFAULT_CELL_LABEL_KEYS),
                   help="Cell-type label preference order (first present wins).")
    p.add_argument("--niche-label-keys", default=",".join(pa.DEFAULT_NICHE_LABEL_KEYS),
                   help="Niche label preference order (first present wins).")
    p.add_argument("--no-discretization", action="store_true",
                   help="Skip folding the discrete-vs-continuous comparison in.")
    p.add_argument("--discretization-summary", default=None,
                   help="Explicit discretization_summary.csv (else auto-discovered).")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve + print the per-prefix multiseed dirs and planned "
                        "outputs; read nothing else, write nothing.")
    args = p.parse_args(argv)

    art = args.artifacts_root
    out_dir = Path(args.out_dir or f"{art}/{args.dataset}/ablations")
    # Every variant prefix any axis references.
    prefixes = sorted({e.prefix for axis in pa.AXES for e in axis.entries})

    if args.dry_run:
        print(f"dataset        : {args.dataset}")
        print(f"artifacts root : {art}")
        print(f"out dir        : {out_dir}")
        print(f"axes           : {len(pa.AXES)}   prefixes: {len(prefixes)}")
        for pref in prefixes:
            md = sm._resolve_metrics_dir(pref, art, args.dataset)
            print(f"  {pref:<10} -> {md if md else 'MISSING'}")
        print("\nwould write:")
        print(f"  {out_dir}/axis_*.{{svg,png,csv}}")
        print(f"  {out_dir}/ablations_all_metrics_{{long,wide}}.csv")
        return 0

    long_df, resolved, missing = build_long_df(prefixes, art, args.dataset)
    if missing:
        print(f"NOTE: {len(missing)} prefix(es) had no multiseed dir "
              f"(axes using them are skipped): {missing}")
    all_variants = sorted(long_df["variant"].unique().tolist())
    niche_df = long_df[long_df.get("table") == "niche_identification"]
    cell_pref = tuple(k.strip() for k in args.cell_label_keys.split(",") if k.strip())
    niche_pref = tuple(k.strip() for k in args.niche_label_keys.split(",") if k.strip())

    pa._apply_nature_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nReading {len(resolved)} variant(s); rendering into {out_dir}")

    combined_records: list[dict] = []
    n_axes_rendered = 0
    for axis in pa.AXES:
        miss = [e.prefix for e in axis.entries
                if not any(v.startswith(e.prefix) for v in all_variants)]
        if miss:
            print(f"  SKIP {axis.key}: no multiseed rows for {miss}")
            continue
        variant_map = pa._resolve_variants_for_axis(axis, all_variants)
        labels = {e.prefix: e.label for e in axis.entries}
        default_prefix = next(e.prefix for e in axis.entries if e.is_default)
        default_full = variant_map[default_prefix]
        cell_label = pa._label_for_variant(niche_df, default_full, cell_pref)
        niche_label = pa._label_for_variant(niche_df, default_full, niche_pref)

        # Per-metric per-seed array for the default = significance reference.
        ref_arr = {m: _metric_array(long_df, default_full, m, cell_label, niche_label)
                   for m, _ in pa.METRICS}

        per_metric = {m: {} for m, _ in pa.METRICS}
        for entry in axis.entries:
            full = variant_map[entry.prefix]
            for metric_label, _ in pa.METRICS:
                arr = _metric_array(long_df, full, metric_label, cell_label, niche_label)
                val = float(arr.mean()) if arr.size else None
                per_metric[metric_label][entry.label] = val

                n = int(arr.size)
                std = float(arr.std(ddof=1)) if n > 1 else 0.0
                sem = float(std / np.sqrt(n)) if n > 1 else 0.0
                is_def = entry.is_default
                pval = (float("nan") if is_def
                        else sm._pvalue(ref_arr[metric_label], arr, "ttest"))
                combined_records.append({
                    "axis": axis.key, "prefix": entry.prefix, "variant": full,
                    "label": entry.label, "is_default": is_def,
                    "metric": metric_label,
                    "value": val,
                    "n": n,
                    "mean": (float(arr.mean()) if n else float("nan")),
                    "std": std, "sem": sem, "ci95": sm._ci95(arr),
                    "p_vs_default": pval,
                    "stars": "" if is_def else sm._stars(pval),
                    "cell_label_key": cell_label, "niche_label_key": niche_label,
                })

        # Per-axis point-value CSV (parity with the standalone plotter).
        per_axis_rows = [{
            "axis": axis.key, "prefix": e.prefix, "variant": variant_map[e.prefix],
            "label": e.label, "is_default": e.is_default, "metric": m,
            "value": per_metric[m][e.label],
            "cell_label_key": cell_label, "niche_label_key": niche_label,
            "cell_code_key": pa.CELL_CODE_KEY, "niche_code_key": pa.NICHE_CODE_KEY,
            "cell_emb_key": pa.CELL_EMB_KEY, "niche_emb_key": pa.NICHE_EMB_KEY,
        } for e in axis.entries for m, _ in pa.METRICS]
        pd.DataFrame(per_axis_rows).to_csv(out_dir / f"{axis.key}.csv", index=False)

        pa.render_axis(axis=axis, per_metric=per_metric, labels=labels,
                       out_path_base=out_dir / axis.key)
        n_axes_rendered += 1

    # --- combined long CSV (+ discretization pseudo-axis) -------------------
    long_out = pd.DataFrame(combined_records)
    if not args.no_discretization:
        disc = _resolve_discretization(out_dir, args.discretization_summary)
        if disc is not None:
            try:
                long_out = pd.concat([long_out, _load_discretization(disc)],
                                     ignore_index=True, sort=False)
                print(f"  + discretization: {disc.name}  ({disc})")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! discretization skipped ({disc}): {exc}", file=sys.stderr)
        else:
            print("  ! no discretization_per_seed.csv / discretization_summary.csv "
                  "found (pass --discretization-summary or --no-discretization).",
                  file=sys.stderr)

    long_out["_axis_num"] = long_out["axis"].map(agg._axis_num)
    long_out = (long_out.sort_values(["_axis_num", "axis"], kind="stable")
                .drop(columns="_axis_num"))
    long_path = out_dir / "ablations_all_metrics_long.csv"
    long_out.to_csv(long_path, index=False)

    # --- wide / table-ready -------------------------------------------------
    idx_cols = [c for c in ("axis", "variant", "label", "is_default")
                if c in long_out.columns]
    wide = long_out.pivot_table(index=idx_cols, columns="metric",
                                values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    metric_cols = ([m for m in agg.METRIC_ORDER if m in wide.columns]
                   + [c for c in wide.columns
                      if c not in agg.METRIC_ORDER and c not in idx_cols])
    wide = wide[idx_cols + metric_cols]
    wide["_axis_num"] = wide["axis"].map(agg._axis_num)
    sort_cols, asc = ["_axis_num"], [True]
    if "is_default" in wide.columns:
        sort_cols.append("is_default"); asc.append(False)   # default first
    wide = (wide.sort_values(sort_cols, ascending=asc, kind="stable")
            .drop(columns="_axis_num").reset_index(drop=True))
    wide_path = out_dir / "ablations_all_metrics_wide.csv"
    wide.to_csv(wide_path, index=False)

    print("\n" + "=" * 70)
    print("ABLATION REPORT DONE")
    print(f"  axes rendered : {n_axes_rendered}/{len(pa.AXES)}")
    print(f"  figures       : {out_dir}/axis_*.{{svg,png}}  (+ per-axis axis_*.csv)")
    print(f"  metrics long  : {long_path}  ({len(long_out)} rows)")
    print(f"  metrics wide  : {wide_path}  <- paper table")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
