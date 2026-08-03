#!/usr/bin/env python3
"""
Butun pipeline'i tek komutla calistirir (final surum).

Calistirilan zincir
-------------------
  ADIM 1  T1w goruntuleri -> (N, S, S, S) tensor veri seti
          - ana kume   : hayalet artefaktli denekler ELENIR   -> derivatives/
          - duyarlilik : hayaletliler DAHIL edilir            -> derivatives_ghost/
  ADIM 2  Tucker(MPCA)+MLP + baseline'lar + istatistik + figurler
  ADIM 3  Karistirici ve secim yanliligi kontrolleri
  ADIM 4  Kovaryant regresyonu + tarayici-disi genelleme
  ADIM 5  Butun sonuclari tek rapora topla (markdown + LaTeX)

Varsayilan kosular
------------------
  schz_vs_control      ana kume     sizofreni vs kontrol
  patient_vs_control   ana kume     tum hastalar vs kontrol
  4class               ana kume     dort tani
  schz_keepghost       duyarlilik   hayaletliler dahil, sizofreni vs kontrol

Ornekler
--------
# 0) Veri indirmeden, sentetik veriyle uctan uca duman testi (~4 dk):
python run_all.py --smoke

# 1) Tam calisma (varsayilan):
python run_all.py --bids-dir data/ds000030

# 2) Hizli surum: tek gorev, az tekrar, rank taramasi ve duyarlilik yok
python run_all.py --bids-dir data/ds000030 --tasks schz_vs_control \
    --repeats 2 --no-rank-sweep --no-ghost-run

# 3) On isleme zaten yapildiysa onu atla
python run_all.py --bids-dir data/ds000030 --skip-step1

# 4) Sadece raporu yeniden uret (analiz yapmaz, saniyeler surer)
python run_all.py --only-report

Notlar
------
* Bir kosu hata verirse pipeline DURMAZ; sonundaki ozet tabloda gorunur.
* --skip-step1 ile on isleme atlanip geri kalani tekrar calistirilabilir.
* RAM sikintisinda: --size 48 ve/veya --skip-pca kullan.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ana kumede calistirilacak gorevler
DEFAULT_TASKS = ["schz_vs_control", "patient_vs_control", "4class"]

# duyarlilik analizi (hayaletliler dahil)
GHOST_TASK = "schz_vs_control"
GHOST_TAG = "schz_keepghost"
GHOST_DERIV = "derivatives_ghost"


# ==========================================================================
class Runner:
    """Adimlari sirayla calistirir; sureleri ve hatalari toplar."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.log: list[dict] = []

    def run(self, script: str, *cli_args, label: str = "") -> bool:
        cmd = [script] + [str(a) for a in cli_args]
        title = label or script

        print("\n" + "-" * 74)
        print(f">> {title}")
        print("-" * 74)
        print("$ python " + " ".join(cmd) + "\n", flush=True)

        if self.dry_run:
            self.log.append({"adim": title, "durum": "dry-run", "sure": 0.0})
            return True

        t0 = time.time()
        try:
            r = subprocess.run([sys.executable] + cmd, cwd=HERE)
            ok = (r.returncode == 0)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"\n!! {type(exc).__name__}: {exc}")
            ok = False
        dt = time.time() - t0

        self.log.append({"adim": title, "durum": "OK" if ok else "HATA",
                         "sure": dt})
        if not ok:
            print(f"\n!! ADIM BASARISIZ: {title}")
            print("   Pipeline devam ediyor; ozette gorunecek.")
        return ok

    # ------------------------------------------------------------------
    def summary(self) -> int:
        print("\n" + "=" * 74)
        print("OZET")
        print("=" * 74)
        if not self.log:
            print("  (hicbir adim calistirilmadi)")
            return 0
        w = max(len(e["adim"]) for e in self.log)
        total = 0.0
        for e in self.log:
            mark = {"OK": "[+]", "HATA": "[!]"}.get(e["durum"], "[.]")
            print(f"  {mark} {e['adim']:<{w}s}  {e['durum']:<8s} "
                  f"{e['sure']:8.1f} s")
            total += e["sure"]
        fails = [e["adim"] for e in self.log if e["durum"] == "HATA"]
        print(f"\n  Toplam sure: {total / 60:.1f} dakika")
        if fails:
            print(f"  BASARISIZ ADIMLAR ({len(fails)}):")
            for f in fails:
                print(f"    - {f}")
            return 1
        print("  Butun adimlar tamamlandi.")
        return 0


# ==========================================================================
def check_bids(bids_dir: str) -> bool:
    for cand in (HERE / bids_dir, Path(bids_dir)):
        if (cand / "participants.tsv").exists():
            return True
    return False


def smoke(rn: Runner, args) -> int:
    """Sentetik veriyle uctan uca hizli test (gercek veri gerekmez)."""
    syn = "data/synthetic"
    size = 32
    ranks = [6, 6, 6]

    rn.run("step0_make_synthetic_bids.py", "--out", syn, "--n", 120,
           label="ADIM 0: sentetik veri")
    rn.run("step1_build_tensor_dataset.py", "--bids-dir", syn,
           "--size", size, "--n-jobs", args.n_jobs,
           label="ADIM 1: tensor veri seti")

    for task in ("schz_vs_control", "4class"):
        rn.run("step2_run_experiments.py", "--size", size, "--task", task,
               "--ranks", *ranks, "--repeats", 1,
               label=f"ADIM 2: {task}")
        rn.run("step3_confound_checks.py", "--bids-dir", syn, "--size", size,
               "--task", task, "--ranks", *ranks, "--repeats", 1,
               label=f"ADIM 3: {task}")
        rn.run("step4_deconfound.py", "--size", size, "--task", task,
               "--ranks", *ranks, "--repeats", 1,
               label=f"ADIM 4: {task}")

    rn.run("step5_make_report.py", "--size", size, label="ADIM 5: rapor")

    rc = rn.summary()
    if rc == 0:
        print("\n  DUMAN TESTI TAMAM -- pipeline makinende calisiyor.")
        print("  Sonuclar anlamsizdir (veri sentetik); amac yalnizca koddu.")
        print("\n  Simdi gercek veriyle:")
        print("    python run_all.py --bids-dir data/ds000030")
    return rc


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tensors and Neural Networks -- tam pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--bids-dir", default="data/ds000030",
                    help="BIDS klasoru (participants.tsv burada olmali)")
    ap.add_argument("--size", type=int, default=64,
                    help="tensor kenar uzunlugu; RAM azsa 48")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="on islemede paralel islem sayisi")
    ap.add_argument("--ranks", type=int, nargs=3, default=[8, 8, 8],
                    help="Tucker cekirdek boyutlari")
    ap.add_argument("--repeats", type=int, default=3,
                    help="capraz dogrulama tekrar sayisi")
    ap.add_argument("--splits", type=int, default=5,
                    help="capraz dogrulama kat sayisi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    help="ana kumede calistirilacak gorevler")

    ap.add_argument("--smoke", action="store_true",
                    help="sentetik veriyle hizli uctan uca test")
    ap.add_argument("--only-report", action="store_true",
                    help="sadece ADIM 5 (mevcut CSV'lerden rapor uret)")

    ap.add_argument("--skip-step1", action="store_true",
                    help="on islemeyi atla (tensor dosyalari zaten varsa)")
    ap.add_argument("--no-ghost-run", action="store_true",
                    help="hayalet duyarlilik analizini calistirma")
    ap.add_argument("--no-confounds", action="store_true",
                    help="ADIM 3 ve 4'u calistirma")
    ap.add_argument("--no-rank-sweep", action="store_true",
                    help="rank taramasini calistirma (daha hizli)")
    ap.add_argument("--skip-pca", action="store_true",
                    help="PCA baseline'ini atla (RAM azsa)")
    ap.add_argument("--dry-run", action="store_true",
                    help="komutlari sadece yazdir, calistirma")
    args = ap.parse_args()

    print("=" * 74)
    print("TENSORS AND NEURAL NETWORKS -- tam pipeline")
    print("=" * 74)

    rn = Runner(dry_run=args.dry_run)

    # ---------------- duman testi -------------------------------------
    if args.smoke:
        return smoke(rn, args)

    # ---------------- sadece rapor ------------------------------------
    if args.only_report:
        rn.run("step5_make_report.py", "--size", args.size,
               label="ADIM 5: rapor")
        return rn.summary()

    # ---------------- on kontrol --------------------------------------
    if not args.skip_step1 and not check_bids(args.bids_dir):
        print(f"\nHATA: {args.bids_dir}/participants.tsv bulunamadi.")
        print("Once veriyi indir:")
        print(f"  python step0_download_t1w.py --target-dir {args.bids_dir}")
        print("\nVeri indirmeden pipeline'i test etmek istersen:")
        print("  python run_all.py --smoke")
        return 1

    ghost_run = not args.no_ghost_run
    print(f"\nBIDS klasoru : {args.bids_dir}")
    print(f"Tensor boyutu: {args.size}^3")
    print(f"Tucker rank  : {tuple(args.ranks)}")
    print(f"CV           : {args.splits}-kat x {args.repeats} tekrar")
    print(f"Gorevler     : {', '.join(args.tasks)}")
    print(f"Hayalet duyarlilik kosusu : {'evet' if ghost_run else 'hayir'}")
    print(f"Karistirici kontrolleri   : "
          f"{'hayir' if args.no_confounds else 'evet'}")

    common = ["--size", args.size, "--ranks", *args.ranks,
              "--splits", args.splits, "--repeats", args.repeats,
              "--seed", args.seed]

    # ---------------- ADIM 1 -------------------------------------------
    if args.skip_step1:
        print("\n(ADIM 1 atlandi -- mevcut tensor dosyalari kullanilacak)")
    else:
        rn.run("step1_build_tensor_dataset.py",
               "--bids-dir", args.bids_dir, "--size", args.size,
               "--n-jobs", args.n_jobs,
               label="ADIM 1: tensor (ana kume)")

        if ghost_run:
            rn.run("step1_build_tensor_dataset.py",
                   "--bids-dir", args.bids_dir, "--size", args.size,
                   "--n-jobs", args.n_jobs, "--keep-ghost",
                   "--out-dir", GHOST_DERIV,
                   label="ADIM 1: tensor (hayaletliler dahil)")

    # ---------------- ADIM 2-4: ana kume -------------------------------
    for task in args.tasks:
        cmd = ["step2_run_experiments.py", "--task", task] + common
        if not args.no_rank_sweep:
            cmd.append("--rank-sweep")
        if args.skip_pca:
            cmd.append("--skip-pca")
        rn.run(*cmd, label=f"ADIM 2: {task}")

        if args.no_confounds:
            continue

        rn.run("step3_confound_checks.py", "--bids-dir", args.bids_dir,
               "--task", task, *common, label=f"ADIM 3: {task}")
        rn.run("step4_deconfound.py", "--task", task, *common,
               label=f"ADIM 4: {task}")

    # ---------------- ADIM 2-4: duyarlilik kosusu ----------------------
    if ghost_run:
        cmd = ["step2_run_experiments.py", "--deriv-dir", GHOST_DERIV,
               "--task", GHOST_TASK, "--tag", GHOST_TAG] + common
        if args.skip_pca:
            cmd.append("--skip-pca")
        rn.run(*cmd, label=f"ADIM 2: {GHOST_TAG}")

        if not args.no_confounds:
            # ADIM 3 duyarlilik kumesi icin de gerekli: bu kume tezin BIRINCIL
            # analizi oldugundan ornek karakteristikleri (Tablo 1) ve
            # oznitelik-icerik analizi bu n=175 kume icin de raporlanmali.
            rn.run("step3_confound_checks.py", "--bids-dir", args.bids_dir,
                   "--deriv-dir", GHOST_DERIV, "--task", GHOST_TASK,
                   "--tag", GHOST_TAG, "--skip-ghost-check", *common,
                   label=f"ADIM 3: {GHOST_TAG}")
            rn.run("step4_deconfound.py", "--deriv-dir", GHOST_DERIV,
                   "--task", GHOST_TASK, "--tag", GHOST_TAG, *common,
                   label=f"ADIM 4: {GHOST_TAG}")

    # ---------------- ADIM 5 -------------------------------------------
    rn.run("step5_make_report.py", "--size", args.size,
           label="ADIM 5: birlesik rapor")

    rc = rn.summary()

    print("\n" + "=" * 74)
    print("CIKTILAR")
    print("=" * 74)
    print("  results/THESIS_REPORT.md       <- ANA RAPOR (once buna bak)")
    print("  results/latex/*.tex            <- teze yapistirilacak tablolar")
    print("  results/figures/*.png          <- butun figurler")
    print("  results/tables/*.csv           <- ham sayilar")
    print("  results/report_*.md            <- kosu bazli ozetler")
    print("  results/confound_report_*.md   <- karistirici kontrolleri")
    print("  results/deconfound_report_*.md <- arindirilmis analiz")
    if rc != 0:
        print("\n  Bazi adimlar basarisiz oldu; yukaridaki ozete bak.")
        print("  Tek bir adimi tekrar denemek icin ilgili komutu dogrudan")
        print("  calistirabilirsin (komut satirlari ciktida gorunuyor).")
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nKullanici tarafindan durduruldu.")
        raise SystemExit(130)