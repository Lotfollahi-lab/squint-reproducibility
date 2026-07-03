#!/usr/bin/env python3
"""
Significance marks for the main niche-/cell-type-identification benchmark
(Table 1 of the paper). For every method and every quality metric
(NMI, ARI, MMD) in each dataset, test that method's five per-seed values
against SQUINT's with a two-sample Welch t-test (unpaired: SQUINT and each
baseline are DIFFERENT models with independent seed draws), and emit the
significance stars printed as superscripts on the non-SQUINT cells:
    *  p<0.05   **  p<0.01   ***  p<0.001
SQUINT is the reference row, so it carries no marks (like the ablation
table's shaded default). Direction (who is ahead) is read from the
bold/underline best/second markup already in the table. iLISI and runtime
are NOT marked (iLISI has no unambiguous better/worse direction; runtime is
a resource metric).

Per-comparison (uncorrected) Welch tests, matching the ablation table's
"vs. default" convention. Pass --holm for Holm-Bonferroni correction across
the quality tests instead.

Input: the per-seed benchmark CSVs written by
plot_{niche,cell_type}_identification_benchmark.py, one per dataset, with
columns `method, seed, <metrics...>`. Default location is the paper's
`mlcb2026_paper/tables/` dir; override with --tables-dir.

No third-party dependencies (pure-Python Welch t-test via the regularised
incomplete beta function), so it runs anywhere the CSVs are present.

Usage:
    python compute_table1_significance.py
    python compute_table1_significance.py --tables-dir /path/to/tables --holm
    python compute_table1_significance.py --out table1_significance.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from statistics import mean, variance

# Dataset dir-tag -> column label used in the paper table.
DATASETS = [("mmb0-1b_smb1", "Mouse Brain"),
            ("chl59", "CosMx NSCLC"),
            ("xhs1000", "Xenium Eczema")]
BLOCKS = [("Niche", "niche_identification_benchmark"),
          ("Cell-type", "cell_type_identification_benchmark")]
QUALITY = ("NMI", "ARI", "MMD")     # metrics that carry stars


# --- pure-Python two-sample Welch t-test -----------------------------------
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
    """Two-sided Welch t-test p-value for samples a, b (n>=2 each)."""
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


def _metric_of(col):
    c = col.lower()
    for key in ("nmi", "ari", "mmd"):
        if key in c:
            return key.upper()
    return None


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_tables = os.path.normpath(
        os.path.join(here, "..", "..", "..", "mlcb2026_paper", "tables"))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", default=default_tables,
                    help="Dir with per-seed benchmark CSVs (default: paper tables/).")
    ap.add_argument("--holm", action="store_true",
                    help="Holm-Bonferroni correction across all quality tests.")
    ap.add_argument("--out", default=None, help="Optional CSV output path.")
    args = ap.parse_args(argv)

    records = []   # (block, dataset_label, method, metric, p)
    for block, pref in BLOCKS:
        for ds, dslabel in DATASETS:
            f = os.path.join(args.tables_dir, f"{pref}_{ds}.csv")
            if not os.path.isfile(f):
                print(f"[skip missing] {f}")
                continue
            rows = list(csv.DictReader(open(f)))
            cols = [c for c in rows[0] if c not in ("method", "seed")]
            data = {}
            for r in rows:
                for c in cols:
                    try:
                        data.setdefault(r["method"], {}).setdefault(c, []).append(float(r[c]))
                    except (TypeError, ValueError):
                        pass
            sq = next((m for m in data if m.upper() == "SQUINT"), None)
            if sq is None:
                print(f"[skip: no SQUINT] {f}")
                continue
            metcol = {m: next((c for c in cols if _metric_of(c) == m), None)
                      for m in QUALITY}
            for meth in data:
                if meth.upper() == "SQUINT":
                    continue
                for m in QUALITY:
                    col = metcol[m]
                    if col and data[meth].get(col) and data[sq].get(col):
                        records.append([block, dslabel, meth, m,
                                        welch_p(data[meth][col], data[sq][col])])

    if args.holm:
        valid = [r for r in records if r[4] is not None]
        order = sorted(range(len(valid)), key=lambda i: valid[i][4])
        n = len(valid)
        prev = 0.0
        for rank, i in enumerate(order):
            prev = max(prev, min(1.0, valid[i][4] * (n - rank)))
            valid[i][4] = prev   # overwrite p with corrected value

    # print
    hdr = f"{'Block':<10}{'Dataset':<14}{'Method':<15}{'Metric':<6}{'p':>10}{'sig':>5}"
    print(hdr)
    print("-" * len(hdr))
    for block, dslabel, meth, m, p in records:
        ps = f"{p:.4f}" if p is not None else "n/a"
        print(f"{block:<10}{dslabel:<14}{meth:<15}{m:<6}{ps:>10}{stars(p):>5}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["block", "dataset", "method", "metric", "p_value", "sig"])
            for block, dslabel, meth, m, p in records:
                w.writerow([block, dslabel, meth, m,
                            "" if p is None else f"{p:.6g}", stars(p)])
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
