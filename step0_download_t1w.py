#!/usr/bin/env python3
"""
ADIM 0 -- ds000030'dan SADECE T1w anatomik goruntuleri indir.

Tum dataset ~70 GB. Bize denek basina tek bir T1w dosyasi yetiyor: ~3 GB.

Neden bu script gerekli?
------------------------
participants.tsv 272 denek listeler ama HEPSINDE T1w taramasi yok. Eksik bir
dosya yolu istendiginde openneuro-py butun indirmeyi RuntimeError ile durdurur.
Bu script iki katmanli koruma uygular:

  1. participants.tsv'deki `T1w` sutununu (1/0) kullanarak sadece T1w'si olan
     denekleri ister.
  2. Buna ragmen eksik cikan bir yol olursa, hata mesajindan o yolu ayikla,
     listeden cikar ve devam et -- indirme durmaz.

Ayrica indirme kucuk gruplar halinde yapilir, boylece tek bir problemli
denek butun isi bozmaz ve kesilen indirme kaldigi yerden devam eder.

Kullanim:
    pip install openneuro-py
    python step0_download_t1w.py --target-dir data/ds000030

    # once kucuk bir parcayla dene (~450 MB):
    python step0_download_t1w.py --target-dir data/ds000030 --limit 40

Alternatif (bu script sorun cikarirsa -- AWS CLI hicbir filtre problemi yasamaz):
    aws s3 sync --no-sign-request s3://openneuro.org/ds000030 data/ds000030 ^
        --exclude "*" --include "participants.tsv" ^
        --include "dataset_description.json" ^
        --include "sub-*/anat/*T1w.nii.gz"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# --------------------------------------------------------------------------
def parse_missing_paths(msg: str) -> list[str]:
    """
    openneuro-py'nin hata mesajindan EKSIK yollari ayikla.

    Mesaj su formatta gelir:
        Could not find path in the dataset:
        - sub-10299/anat/sub-10299_T1w.nii.gz
        Perhaps you mean one of these paths:
        - sub-10998/anat/sub-10998_T1w.nii.gz      <-- bunlar oneri, eksik degil
    "Perhaps you mean" sonrasindaki satirlari almamak kritik.
    """
    missing, collecting = [], False
    for raw in msg.splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("could not find path"):
            collecting = True
            continue
        if low.startswith("perhaps you mean"):
            collecting = False
            continue
        if collecting and line.startswith("-"):
            p = line.lstrip("-").strip()
            if p:
                missing.append(p)
    return missing


def download_with_skip(on, dataset, tag, target_dir, include, max_retries=25):
    """
    include listesini indir; eksik yollar cikarsa onlari atlayip devam et.

    Returns
    -------
    (indirilen_yol_sayisi, atlanan_yollar)
    """
    remaining = list(include)
    dropped: list[str] = []

    for _ in range(max_retries):
        if not remaining:
            return 0, dropped
        try:
            on.download(dataset=dataset, tag=tag,
                        target_dir=str(target_dir), include=remaining)
            return len(remaining), dropped
        except RuntimeError as exc:
            missing = parse_missing_paths(str(exc))
            if not missing:
                raise                                   # baska bir hata
            before = len(remaining)
            missing_set = set(missing)
            remaining = [p for p in remaining if p not in missing_set]
            dropped.extend(missing)
            if len(remaining) == before:                # ilerleme yok -> sonsuz dongu
                raise
            print(f"    (atlandi: {', '.join(missing)})", flush=True)

    raise RuntimeError("Cok fazla eksik yol; indirme tamamlanamadi.")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default="data/ds000030")
    ap.add_argument("--dataset", default="ds000030")
    ap.add_argument("--tag", default="1.0.0", help="snapshot surumu")
    ap.add_argument("--limit", type=int, default=0,
                    help="sadece ilk N denegi indir (deneme icin)")
    ap.add_argument("--chunk", type=int, default=20,
                    help="tek seferde kac denek istenecek")
    ap.add_argument("--with-json", action="store_true",
                    help="T1w .json yan dosyalarini da indir (proje icin gerekmez)")
    ap.add_argument("--no-t1w-filter", action="store_true",
                    help="participants.tsv'deki T1w sutununu yoksay")
    args = ap.parse_args()

    try:
        import openneuro as on
    except ImportError:
        print("openneuro-py kurulu degil:  pip install openneuro-py\n")
        print(__doc__)
        return 1

    import pandas as pd

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # --- 1) participants.tsv ---------------------------------------------
    ptsv = target / "participants.tsv"
    if ptsv.exists():
        print(f"participants.tsv zaten var: {ptsv}")
    else:
        print("participants.tsv indiriliyor ...")
        on.download(dataset=args.dataset, tag=args.tag,
                    target_dir=str(target),
                    include=["participants.tsv", "dataset_description.json"])
    if not ptsv.exists():
        print(f"HATA: {ptsv} olusmadi. Docstring'deki AWS CLI yontemini dene.")
        return 1

    df = pd.read_csv(ptsv, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    pid_col = lower.get("participant_id")
    if pid_col is None:
        print(f"HATA: participant_id sutunu yok. Sutunlar: {list(df.columns)}")
        return 1
    df[pid_col] = df[pid_col].astype(str).str.strip()

    print(f"\nparticipants.tsv: {len(df)} denek")

    # --- 2) SADECE T1w'si olan denekler ----------------------------------
    t1_col = lower.get("t1w")
    if t1_col and not args.no_t1w_filter:
        flag = pd.to_numeric(df[t1_col], errors="coerce").fillna(0).astype(int)
        n_missing = int((flag != 1).sum())
        df = df.loc[flag == 1]
        print(f"  T1w sutunu bulundu -> T1w'si olmayan {n_missing} denek atlandi")
    else:
        print("  (T1w sutunu yok/yoksayildi; butun denekler denenecek)")

    diag_col = lower.get("diagnosis")
    if diag_col:
        counts = df[diag_col].astype(str).str.strip().str.upper().value_counts()
        print("  tani dagilimi:", dict(counts))

    subs = df[pid_col].tolist()
    if args.limit:
        subs = subs[: args.limit]
    print(f"  INDIRILECEK: {len(subs)} denek")

    # --- 3) gruplar halinde indir ----------------------------------------
    est_gb = len(subs) * 11 / 1024
    print(f"\nTahmini boyut: ~{est_gb:.1f} GB")
    print("Kesilirse ayni komutu tekrar calistir; kaldigi yerden devam eder.\n")

    all_dropped: list[str] = []
    n_chunks = (len(subs) + args.chunk - 1) // args.chunk

    for ci in range(n_chunks):
        batch = subs[ci * args.chunk:(ci + 1) * args.chunk]
        include = [f"{s}/anat/{s}_T1w.nii.gz" for s in batch]
        if args.with_json:
            include += [f"{s}/anat/{s}_T1w.json" for s in batch]

        print(f"[{ci + 1}/{n_chunks}] {batch[0]} ... {batch[-1]} "
              f"({len(batch)} denek)", flush=True)
        try:
            _, dropped = download_with_skip(on, args.dataset, args.tag,
                                            target, include)
            all_dropped.extend(dropped)
        except Exception as exc:
            print(f"    ! grup basarisiz ({type(exc).__name__}: {exc})")
            print("    devam ediliyor; bu grubu sonra tekrar deneyebilirsin.")

        found = len(list(target.glob("sub-*/anat/*_T1w.nii.gz")))
        print(f"    diskte toplam {found} T1w dosyasi", flush=True)

    # --- 4) ozet ----------------------------------------------------------
    found = sorted(target.glob("sub-*/anat/*_T1w.nii.gz"))
    total_gb = sum(p.stat().st_size for p in found) / 1024 ** 3
    print("\n" + "=" * 62)
    print(f"Bitti: {len(found)} T1w dosyasi, toplam {total_gb:.2f} GB")
    print(f"Klasor: {target.resolve()}")
    if all_dropped:
        print(f"\nDatasette bulunmadigi icin atlanan {len(all_dropped)} yol:")
        for p in all_dropped[:20]:
            print(f"  - {p}")
        if len(all_dropped) > 20:
            print(f"  ... ve {len(all_dropped) - 20} tane daha")
        print("Bu normaldir: bazi deneklerin T1w taramasi yok.")

    missing_subs = [s for s in subs
                    if not (target / s / "anat" / f"{s}_T1w.nii.gz").exists()]
    if missing_subs:
        print(f"\nHala eksik olan {len(missing_subs)} denek "
              f"(scripti tekrar calistirmayi dene): {missing_subs[:10]}")

    print("\nSonraki adim:")
    print(f"  python step1_build_tensor_dataset.py --bids-dir {args.target_dir} "
          f"--size 64 --n-jobs 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())