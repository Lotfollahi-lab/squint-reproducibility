#!/usr/bin/env python
"""
ONE entry-point for the SQUINT ablation report.

multiseed runs ──▶ summary CSVs ──▶ all axis figures + per-axis CSVs ──▶ one
combined metrics CSV (long + wide, incl. the discretization comparison).

It orchestrates the three existing scripts in sequence, threading each step's
output into the next with CONSISTENT paths — so you never again hit the
`summary_long.csv` (plot default) vs `ablation_summary_long.csv` (summarizer
output) filename mismatch or a stale default `--timestamp`:

  1. summarize_ablation_multiseed.py  reads <artifacts>/<dataset>/<set>_v*__multiseed
                                      -> <summary-dir>/ablation_summary_{long,wide}.csv
  2. plots/plot_ablations.py          --summary-csv <that long.csv>
                                      -> <out-dir>/axis_*.{svg,png,csv}
  3. aggregate_ablation_csvs.py       --ablations-dir <out-dir>
                                      -> <out-dir>/ablations_all_metrics_{long,wide}.csv
                                         (folds in the discretization comparison)

Usage:
  python analysis/ablations/run_ablation_report.py --set s57
  python analysis/ablations/run_ablation_report.py --set s57 --skip-summarize   # reuse existing summary
  python analysis/ablations/run_ablation_report.py --set s57 --dry-run          # print the 3 commands
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="preset", default="s57",
                   help="Named ablation set for summarize_ablation_multiseed.py (default s57).")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--summary-dir", default=None,
                   help="Where the summary CSVs live/are written "
                        "(default <artifacts>/<dataset>/_ablation_summary).")
    p.add_argument("--out-dir", default=None,
                   help="Where figures + per-axis CSVs + combined metrics go "
                        "(default <artifacts>/<dataset>/ablations).")
    p.add_argument("--skip-summarize", action="store_true",
                   help="Reuse the existing ablation_summary_long.csv (skip step 1; "
                        "use when the multiseed run dirs are gone or unchanged).")
    p.add_argument("--no-discretization", action="store_true",
                   help="Skip folding the discrete-vs-continuous comparison into the combined CSV.")
    p.add_argument("--discretization-summary", default=None,
                   help="Explicit path to discretization_summary.csv (else auto-discovered).")
    p.add_argument("--dry-run", action="store_true", help="Print the 3 commands; don't run them.")
    args = p.parse_args(argv)

    art = args.artifacts_root
    summary_dir = Path(args.summary_dir or f"{art}/{args.dataset}/_ablation_summary")
    summary_csv = summary_dir / "ablation_summary_long.csv"
    out_dir = Path(args.out_dir or f"{art}/{args.dataset}/ablations")
    py = sys.executable

    summarize = [py, str(HERE / "summarize_ablation_multiseed.py"),
                 "--set", args.preset, "--dataset", args.dataset,
                 "--artifacts-root", art, "--out", str(summary_dir)]
    plot = [py, str(HERE / "plots" / "plot_ablations.py"),
            "--summary-csv", str(summary_csv), "--dataset", args.dataset,
            "--out-dir", str(out_dir)]
    aggregate = [py, str(HERE / "aggregate_ablation_csvs.py"),
                 "--ablations-dir", str(out_dir)]
    if args.no_discretization:
        aggregate.append("--no-discretization")
    if args.discretization_summary:
        aggregate += ["--discretization-summary", args.discretization_summary]

    steps = []
    if not args.skip_summarize:
        steps.append(("summarize", summarize))
    steps += [("plot", plot), ("aggregate", aggregate)]

    if args.dry_run:
        for name, cmd in steps:
            print(f"# {name}\n{' '.join(cmd)}\n")
        print(f"# outputs:\n#   summary : {summary_dir}/ablation_summary_{{long,wide}}.csv")
        print(f"#   figures : {out_dir}/axis_*.{{svg,png,csv}}")
        print(f"#   metrics : {out_dir}/ablations_all_metrics_{{long,wide}}.csv")
        return 0

    for name, cmd in steps:
        print(f"\n===== [{name}] {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\nERROR: step '{name}' failed (exit {exc.returncode}). "
                  f"Fix it and re-run (use --skip-summarize to reuse the summary).",
                  file=sys.stderr)
            return exc.returncode
        if name == "summarize" and not summary_csv.is_file():
            print(f"ERROR: summarize did not produce {summary_csv}. Check the "
                  f"--set / run dirs.", file=sys.stderr)
            return 1

    print("\n" + "=" * 70)
    print("ABLATION REPORT DONE")
    print(f"  summary CSVs : {summary_dir}/ablation_summary_{{long,wide}}.csv")
    print(f"  axis figures : {out_dir}/axis_*.{{svg,png}}  (+ per-axis axis_*.csv)")
    print(f"  combined     : {out_dir}/ablations_all_metrics_{{long,wide}}.csv  <- paper table")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
