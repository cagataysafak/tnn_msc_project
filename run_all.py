#!/usr/bin/env python3
"""
Butun pipeline'i tek komutla calistirir.

Ornekler
--------
# 1) Once sentetik veriyle 2 dakikalik duman testi (veri indirmeden):
python run_all.py --smoke

# 2) Gercek veriyle tam calisma:
python run_all.py --bids-dir data/ds000030 --size 64 --n-jobs 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("\n" + "$ " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    r = subprocess.run([sys.executable] + [str(c) for c in cmd], cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"Adim basarisiz: {cmd}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="sentetik veriyle hizli uctan uca test")
    ap.add_argument("--bids-dir", default="data/ds000030")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--ranks", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tasks", nargs="+",
                    default=["schz_vs_control", "4class"])
    ap.add_argument("--rank-sweep", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        run(["step0_make_synthetic_bids.py", "--out", "data/synthetic", "--n", "80"])
        run(["step1_build_tensor_dataset.py", "--bids-dir", "data/synthetic",
             "--size", 32, "--n-jobs", args.n_jobs])
        for t in args.tasks:
            run(["step2_run_experiments.py", "--size", 32, "--task", t,
                 "--ranks", 8, 8, 8, "--repeats", 1])
        print("\nDUMAN TESTI TAMAM. Pipeline calisiyor.")
        return 0

    if not (HERE / args.bids_dir / "participants.tsv").exists() and \
       not Path(args.bids_dir).joinpath("participants.tsv").exists():
        print(f"UYARI: {args.bids_dir}/participants.tsv bulunamadi.")
        print("Once veriyi indir:  python step0_download_t1w.py "
              f"--target-dir {args.bids_dir}")
        return 1

    run(["step1_build_tensor_dataset.py", "--bids-dir", args.bids_dir,
         "--size", args.size, "--n-jobs", args.n_jobs])
    for t in args.tasks:
        cmd = ["step2_run_experiments.py", "--size", args.size, "--task", t,
               "--ranks", *args.ranks, "--repeats", args.repeats]
        if args.rank_sweep:
            cmd.append("--rank-sweep")
        run(cmd)

    print("\nHEPSI TAMAM. results/ klasorune bak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
