#!/usr/bin/env python3
"""
ADIM 0 (opsiyonel) -- Sahte (sentetik) bir BIDS klasoru uret.

Amac: 3 GB'lik gercek veriyi indirmeden once butun pipeline'in
laptopunda hatasiz calistigini 2-3 dakikada dogrulamak.

Uretilen yapi ds000030 ile ayni:
    <out>/participants.tsv
    <out>/sub-XXXXX/anat/sub-XXXXX_T1w.nii.gz

Hacimler kaba bir "beyin" (elipsoid + ventrikul + kortikal doku) icerir;
tani gruplarina gore ventrikul buyuklugu ve kortikal kalinlik sistematik
olarak degistirilir, boylece siniflandirma gorevi anlamli ama kolay olmaz.

Kullanim:
    python step0_make_synthetic_bids.py --out data/synthetic --n 80
    python step1_build_tensor_dataset.py --bids-dir data/synthetic --size 32
    python step2_run_experiments.py --size 32 --task schz_vs_control --repeats 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


GROUPS = ["CONTROL", "SCHZ", "BIPOLAR", "ADHD"]
GROUP_WEIGHTS = np.array([138, 58, 49, 45], dtype=float)
GROUP_WEIGHTS /= GROUP_WEIGHTS.sum()

# grup -> (ventrikul olcegi, kortikal yogunluk carpani)
GROUP_EFFECT = {
    "CONTROL": (1.00, 1.00),
    "SCHZ":    (1.35, 0.92),
    "BIPOLAR": (1.18, 0.96),
    "ADHD":    (1.05, 0.99),
}


def make_brain(shape, rng, vent_scale=1.0, cortex_gain=1.0):
    nx, ny, nz = shape
    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, nx),
                             np.linspace(-1, 1, ny),
                             np.linspace(-1, 1, nz), indexing="ij")

    # kafa konumu/boyutu deneke gore rastgele degissin (kayit sorununu taklit eder)
    cx, cy, cz = rng.normal(0, 0.07, 3)
    scale = rng.normal(1.0, 0.06)
    ax, ay, az = np.array([0.72, 0.88, 0.66]) * scale

    r = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 + ((zz - cz) / az) ** 2)

    vol = np.zeros(shape, dtype=np.float32)
    brain = r < 1.0
    vol[brain] = 0.55                                   # beyaz cevher

    shell = (r > 0.72) & (r < 1.0)                      # kortikal gri cevher
    vol[shell] = 0.85 * cortex_gain

    skull = (r > 1.0) & (r < 1.16)                      # kafatasi/skalp
    vol[skull] = 0.35

    vr = 0.20 * vent_scale
    vent = (((xx - cx) / (vr * 0.9)) ** 2 +
            ((yy - cy) / (vr * 2.1)) ** 2 +
            ((zz - cz) / (vr * 0.8)) ** 2) < 1.0
    vol[vent] = 0.08                                    # ventrikul (BOS)

    vol *= rng.normal(1.0, 0.05)                        # tarayici kazanci
    vol += rng.normal(0, 0.03, shape).astype(np.float32)
    return np.clip(vol, 0, None).astype(np.float32) * 800.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--shape", type=int, nargs=3, default=[48, 56, 48])
    ap.add_argument("--voxel-mm", type=float, nargs=3, default=[3.0, 3.0, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import nibabel as nib

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    groups = rng.choice(GROUPS, size=args.n, p=GROUP_WEIGHTS)
    rows = []
    for i, g in enumerate(groups):
        pid = f"sub-{10000 + i}"
        anat = out / pid / "anat"
        anat.mkdir(parents=True, exist_ok=True)

        vs, cg = GROUP_EFFECT[g]
        vol = make_brain(tuple(args.shape), rng,
                         vent_scale=vs * rng.normal(1.0, 0.08),
                         cortex_gain=cg * rng.normal(1.0, 0.03))

        aff = np.diag(list(args.voxel_mm) + [1.0])
        aff[:3, 3] = -0.5 * np.array(args.shape) * np.array(args.voxel_mm)
        nib.save(nib.Nifti1Image(vol, aff), str(anat / f"{pid}_T1w.nii.gz"))

        rows.append({
            "participant_id": pid,
            "diagnosis": g,
            "age": int(rng.integers(21, 51)),
            "gender": rng.choice(["M", "F"]),
            "ScannerSerialNumber": rng.choice(["A", "B"]),
            "ghost_NoGhost": rng.choice(["No_ghost", "ghost"], p=[0.85, 0.15]),
            "T1w": 1,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.n}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "participants.tsv", sep="\t", index=False)
    (out / "dataset_description.json").write_text(
        '{"Name": "SYNTHETIC smoke-test dataset", "BIDSVersion": "1.0.2"}',
        encoding="utf-8")

    print(f"\n{args.n} sahte denek yazildi -> {out.resolve()}")
    print("Grup dagilimi:", dict(zip(*np.unique(groups, return_counts=True))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
