#!/usr/bin/env python3
"""
ADIM 1 -- BIDS klasorundeki T1w goruntulerinden 4-yollu tensor veri seti kur.

Cikti:
    derivatives/X_<S>.npy      (N, S, S, S) float32   -- tensor veri seti
    derivatives/meta_<S>.csv                          -- denek/etiket/QC tablosu
    results/figures/fig01_*.png, fig02_*.png          -- QC gorselleri

Kullanim:
    python step1_build_tensor_dataset.py --bids-dir data/ds000030
    python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 --n-jobs 4
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tnn import config
from tnn.data import build_subject_table
from tnn.preprocessing import preprocess_t1w


# --------------------------------------------------------------------------
def _worker(args):
    idx, path, size, mask_scale, pct = args
    try:
        vol, info = preprocess_t1w(path, out_size=size,
                                   mask_scale=mask_scale,
                                   intensity_percentile=pct)
        return idx, vol, info, None
    except Exception as exc:                       # tek denek patlasa da devam
        return idx, None, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description="T1w -> tensor veri seti")
    ap.add_argument("--bids-dir", default=str(config.BIDS_DIR))
    ap.add_argument("--out-dir", default=str(config.DERIV_DIR))
    ap.add_argument("--size", type=int, default=config.TARGET_SIZE)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="paralel islem sayisi (RAM ~0.5 GB/is)")
    ap.add_argument("--limit", type=int, default=0,
                    help="sadece ilk N denegi isle (hizli deneme icin)")
    ap.add_argument("--keep-ghost", action="store_true",
                    help="hayalet artefaktli T1w'leri de dahil et")
    ap.add_argument("--mask-scale", type=float, default=config.MASK_THRESHOLD_SCALE)
    args = ap.parse_args()

    bids_dir = Path(args.bids_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ADIM 1: T1w -> 4-yollu tensor veri seti")
    print("=" * 70)
    print(f"BIDS klasoru : {bids_dir}")
    print(f"Hedef boyut  : {args.size}^3")

    tbl = build_subject_table(bids_dir, exclude_ghost=not args.keep_ghost)

    print(f"\nparticipants.tsv         : {len(tbl)} denek")
    print(f"  diskte T1w dosyasi olan : {int(tbl['has_t1w_file'].sum())}")
    if not args.keep_ghost:
        print(f"  hayalet artefakti atilan: {int((~tbl['ghost_ok']).sum())}")
    print(f"  ISLENECEK               : {int(tbl['included'].sum())}")

    print("\nTani dagilimi (islenecekler):")
    print(tbl.loc[tbl["included"], "diagnosis"].value_counts().to_string())

    work = tbl.loc[tbl["included"]].reset_index(drop=True)
    if args.limit:
        work = work.iloc[: args.limit].reset_index(drop=True)
    n = len(work)
    if n == 0:
        print("\nHATA: islenecek denek yok. --bids-dir dogru mu?")
        return 1

    X = np.zeros((n, args.size, args.size, args.size), dtype=np.float32)
    infos: list[dict | None] = [None] * n
    errors: list[tuple[str, str]] = []

    jobs = [(i, work.loc[i, "t1w_path"], args.size, args.mask_scale,
             config.INTENSITY_PERCENTILE) for i in range(n)]

    print(f"\n{n} hacim isleniyor (n_jobs={args.n_jobs}) ...")
    t0 = time.time()
    done = 0

    def _record(idx, vol, info, err):
        nonlocal done
        done += 1
        if err is None:
            X[idx] = vol
            infos[idx] = info
        else:
            errors.append((work.loc[idx, "participant_id"], err))
        if done % 10 == 0 or done == n:
            el = time.time() - t0
            eta = el / done * (n - done)
            print(f"  {done}/{n}  gecen {el:6.1f}s  kalan ~{eta:6.1f}s", flush=True)

    if args.n_jobs and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for f in as_completed(futs):
                _record(*f.result())
    else:
        for j in jobs:
            _record(*_worker(j))

    ok = np.array([i is not None for i in infos])
    print(f"\nBasarili: {int(ok.sum())} / {n}")
    for pid, err in errors:
        print(f"  ! {pid}: {err}")

    X = X[ok]
    meta = work.loc[ok].reset_index(drop=True)
    qc = pd.DataFrame([i for i in infos if i is not None]).reset_index(drop=True)
    meta = pd.concat([meta, qc], axis=1)

    x_path = out_dir / f"X_{args.size}.npy"
    m_path = out_dir / f"meta_{args.size}.csv"
    np.save(x_path, X)
    meta.to_csv(m_path, index=False)

    mb = X.nbytes / 1024 ** 2
    print(f"\nKaydedildi:")
    print(f"  {x_path}   sekil={X.shape}  {mb:.1f} MB")
    print(f"  {m_path}")
    print(f"\nToplam sure: {time.time() - t0:.1f} s")

    # ---- QC gorselleri --------------------------------------------------
    try:
        from tnn.viz_utils import plot_demographics, plot_preprocessing_qc
        plot_demographics(meta, config.FIG_DIR / "fig01_demographics.png")
        plot_preprocessing_qc(X, meta, config.FIG_DIR / "fig02_preprocessing_qc.png")
        print(f"QC figurleri: {config.FIG_DIR}")
    except Exception as exc:
        print(f"(figur uretilemedi: {exc})")

    print("\nSonraki adim:  python step2_run_experiments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
