#!/usr/bin/env python3
"""
Significance marks for the reconstruction/imputation benchmark (Table 2 of
the paper). Companion to compute_table1_significance.py, but for the
gene-expression panel CSVs, which are LONG-format (one row per
metric x panel x split x seed) rather than wide.

For each of the two blocks, every non-reference method is tested against the
shaded reference with a two-sample Welch t-test across the five seeds, and
the significance stars printed as superscripts on the non-reference cells:
    *  p<0.05   **  p<0.01   ***  p<0.001
    Reconstruction block: reference = SQUINT.
    Imputation block:     reference = SQUINT (MC).
The reference row carries no marks. Per-comparison (uncorrected) Welch tests,
matching the ablation table's "vs. reference" convention; pass --holm for a
Holm-Bonferroni correction across all tests instead.

The seven displayed columns map to the following panel codes (verified by
matching each column's reference-row mean to the number printed in the
table): rho_cell -> pe_cw_log, rho_gene -> pe_gw_log, rho_HVG -> pe_hv_log,
rho_s -> sp_gw (gene-wise Spearman), RMSE -> rm_cw_raw (== rm_gw_raw here),
AUROC -> zn_au_all, AP -> zn_ap_all. Each column is scored at the cell and
neighbourhood levels (the two CSV files).

Input: the per-seed panel CSVs written by the imputation/reconstruction
runners (via _holdout_utils.build_pearson_dataframe), one per (block, branch):
    {reconstruction,imputation}_benchmark_panel_{cell,niche}.csv
with columns metric,panel,axis,transform,gene_subset,branch,method,split,seed,value.
Default location is the paper's mlcb2026_paper/tables/ dir; override with
--tables-dir. No third-party dependencies (pure-Python Welch t-test).

Usage:
    python compute_table2_significance.py
    python compute_table2_significance.py --tables-dir /path/to/tables --holm
    python compute_table2_significance.py --out table2_significance.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from statistics import mean, variance

# block -> (csv filename prefix, reference method label)
BLOCKS = [("Reconstruction", "reconstruction_benchmark_panel", "SQUINT"),
          ("Imputation", "imputation_benchmark_panel", "SQUINT (MC)")]
BRANCHES = [("cell", "Cell"), ("niche", "Neighbourhood")]
# displayed column -> panel code (see module docstring)
COLUMNS = [("rho_cell", "pe_cw_log"), ("rho_gene", "pe_gw_log"),
           ("rho_HVG", "pe_hv_log"), ("rho_s", "sp_gw"),
           ("RMSE", "rm_cw_raw"), ("AUROC", "zn_au_all"), ("AP", "zn_ap_all")]


# --- pure-Python two-sample Welch t-test (shared with the Table 1 script) ---
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lb + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def welch_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = variance(a), variance(b)
    se2 = va / len(a) + vb / len(b)
    if se2 == 0:
        return 1.0
    t = (mean(a) - mean(b)) / math.sqrt(se2)
    df = se2 * se2 / ((va / len(a)) ** 2 / (len(a) - 1)
                      + (vb / len(b)) ** 2 / (len(b) - 1))
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def stars(p):
    if p is None:
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_tables = os.path.normpath(
        os.path.join(here, "..", "..", "..", "mlcb2026_paper", "tables"))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", default=default_tables,
                    help="Dir with the panel CSVs (default: paper tables/).")
    ap.add_argument("--holm", action="store_true",
                    help="Holm-Bonferroni correction across all tests.")
    ap.add_argument("--out", default=None, help="Optional CSV output path.")
    args = ap.parse_args(argv)

    records = []   # (block, branch_label, column, method, p)
    for block, pref, ref in BLOCKS:
        for branch, blabel in BRANCHES:
            f = os.path.join(args.tables_dir, f"{pref}_{branch}.csv")
            if not os.path.isfile(f):
                print(f"[skip missing] {f}")
                continue
            vals = {}   # panel -> method -> [seed values]
            for r in csv.DictReader(open(f)):
                try:
                    vals.setdefault(r["panel"], {}).setdefault(
                        r["method"], []).append(float(r["value"]))
                except (TypeError, ValueError):
                    pass
            for col, panel in COLUMNS:
                if panel not in vals or ref not in vals[panel]:
                    print(f"[warn] {block}/{branch}: panel {panel} or ref "
                          f"{ref!r} missing", flush=True)
                    continue
                for meth in vals[panel]:
                    if meth == ref:
                        continue
                    records.append([block, blabel, col, meth,
                                    welch_p(vals[panel][meth], vals[panel][ref])])

    if args.holm:
        valid = [r for r in records if r[4] is not None]
        order = sorted(range(len(valid)), key=lambda i: valid[i][4])
        n = len(valid)
        prev = 0.0
        for rank, i in enumerate(order):
            prev = max(prev, min(1.0, valid[i][4] * (n - rank)))
            valid[i][4] = prev

    hdr = f"{'Block':<16}{'Level':<14}{'Column':<10}{'Method':<16}{'p':>10}{'sig':>5}"
    print(hdr)
    print("-" * len(hdr))
    last = None
    for block, blabel, col, meth, p in records:
        key = (block, blabel)
        if key != last:
            print(f"--- {block} / {blabel} (ref = "
                  f"{dict((b, r) for b, _, r in BLOCKS)[block]}) ---")
            last = key
        ps = f"{p:.4f}" if p is not None else "n/a"
        print(f"{block:<16}{blabel:<14}{col:<10}{meth:<16}{ps:>10}{stars(p):>5}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["block", "level", "column", "method", "p_value", "sig"])
            for block, blabel, col, meth, p in records:
                w.writerow([block, blabel, col, meth,
                            "" if p is None else f"{p:.6g}", stars(p)])
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
