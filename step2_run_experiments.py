#!/usr/bin/env python3
"""
ADIM 2 -- Tucker (MPCA) + MLP pipeline'i, baseline'lar, istatistik ve figurler.

Karsilastirilan yontemler
-------------------------
  Tucker+MLP     : kismi Tucker (MPCA) cekirdek ozellikleri -> MLP        [ANA YONTEM]
  Tucker+LogReg  : ayni ozellikler -> lojistik regresyon    (NN olmayan referans)
  PCA+MLP        : hacim duzlestirilip PCA -> MLP           (vektor tabanli indirgeme)
  Voxel-MLP      : ayristirmasiz, havuzlanmis voksellerle MLP  (ayristirmasiz baseline)
  Dummy          : sinif oranlarina gore rastgele            (sans seviyesi)

Kritik nokta: MPCA/PCA/olcekleyici HER FOLD'DA SADECE EGITIM VERISIYLE
fit edilir, test verisi yalnizca transform edilir. Aksi halde veri sizintisi
(data leakage) olur ve sonuclar sisirilir.

Kullanim:
    python step2_run_experiments.py --task schz_vs_control
    python step2_run_experiments.py --task 4class --ranks 12 12 12 --repeats 5
    python step2_run_experiments.py --task schz_vs_control --rank-sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from tnn import config
from tnn.data import apply_task
from tnn.evaluation import (METRIC_NAMES, aggregate_confusion, cohens_d_paired,
                            compute_metrics, corrected_paired_ttest,
                            holm_bonferroni, mean_ci, per_class_f1,
                            wilcoxon_test)
from tnn.nn_utils import backend_name, predict_proba, set_threads, train_mlp
from tnn.tensor_utils import MPCA
from tnn import viz_utils as viz


REFERENCE = "Tucker+MLP"


# --------------------------------------------------------------------------
def avg_pool(X: np.ndarray, factor: int) -> np.ndarray:
    """Hacimleri `factor` katsayisiyla ortalama havuzlama ile kucult."""
    if factor <= 1:
        return X
    n, a, b, c = X.shape
    a2, b2, c2 = (a // factor) * factor, (b // factor) * factor, (c // factor) * factor
    Xc = X[:, :a2, :b2, :c2]
    return Xc.reshape(n, a2 // factor, factor, b2 // factor, factor,
                      c2 // factor, factor).mean(axis=(2, 4, 6))


def fit_scaler_pair(F_tr, F_te):
    sc = StandardScaler().fit(F_tr)
    return sc.transform(F_tr).astype(np.float32), sc.transform(F_te).astype(np.float32)


# --------------------------------------------------------------------------
def run_fold(X_tr, X_te, y_tr, y_te, n_classes, ranks, mlp_params, seed,
             pca_components, pool_factor, skip_pca=False, record_history=False):
    """Tek bir CV fold'unda butun yontemleri egit ve degerlendir."""
    out, extras = {}, {}

    # ---------------- Tucker / MPCA (SADECE egitimle fit) ----------------
    t0 = time.time()
    mpca = MPCA(ranks=ranks, n_iter=config.MPCA_N_ITER).fit(X_tr)
    F_tr, F_te = mpca.features(X_tr), mpca.features(X_te)
    F_tr, F_te = fit_scaler_pair(F_tr, F_te)
    extras["mpca_time"] = time.time() - t0
    extras["mpca_explained"] = mpca.explained_ratio_
    extras["mpca_recon_err_test"] = mpca.reconstruction_error(X_te)
    extras["mpca"] = mpca

    model, hist = train_mlp(F_tr, y_tr, n_classes, mlp_params, seed=seed,
                            record_history=record_history)
    proba = predict_proba(model, F_te)
    out["Tucker+MLP"] = (proba.argmax(1), proba)
    if record_history:
        extras["history"] = hist

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(F_tr, y_tr)
    p = lr.predict_proba(F_te)
    out["Tucker+LogReg"] = (p.argmax(1), p)

    # ---------------- PCA + MLP ------------------------------------------
    if not skip_pca:
        V_tr = X_tr.reshape(len(X_tr), -1)
        V_te = X_te.reshape(len(X_te), -1)
        k = int(min(pca_components, len(X_tr) - 1, V_tr.shape[1]))
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
        P_tr = pca.fit_transform(V_tr)
        P_te = pca.transform(V_te)
        P_tr, P_te = fit_scaler_pair(P_tr, P_te)
        m2, _ = train_mlp(P_tr, y_tr, n_classes, mlp_params, seed=seed)
        p = predict_proba(m2, P_te)
        out["PCA+MLP"] = (p.argmax(1), p)
        extras["pca_explained"] = float(pca.explained_variance_ratio_.sum())
        del V_tr, V_te

    # ---------------- Ayristirmasiz voksel MLP ---------------------------
    W_tr = avg_pool(X_tr, pool_factor).reshape(len(X_tr), -1)
    W_te = avg_pool(X_te, pool_factor).reshape(len(X_te), -1)
    W_tr, W_te = fit_scaler_pair(W_tr, W_te)
    m3, _ = train_mlp(W_tr, y_tr, n_classes, mlp_params, seed=seed)
    p = predict_proba(m3, W_te)
    out["Voxel-MLP"] = (p.argmax(1), p)
    extras["voxel_dim"] = W_tr.shape[1]

    # ---------------- Sans seviyesi --------------------------------------
    dm = DummyClassifier(strategy="stratified", random_state=seed).fit(
        np.zeros((len(y_tr), 1)), y_tr)
    p = dm.predict_proba(np.zeros((len(y_te), 1)))
    out["Dummy"] = (p.argmax(1), p)

    return out, extras


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Tucker + MLP deneyleri")
    ap.add_argument("--deriv-dir", default=str(config.DERIV_DIR))
    ap.add_argument("--size", type=int, default=config.TARGET_SIZE)
    ap.add_argument("--task", default=config.DEFAULT_TASK, choices=list(config.TASKS))
    ap.add_argument("--ranks", type=int, nargs=3, default=list(config.DEFAULT_RANKS))
    ap.add_argument("--splits", type=int, default=config.N_SPLITS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--pca-components", type=int, default=100)
    ap.add_argument("--pool-target", type=int, default=16,
                    help="ayristirmasiz baseline icin havuzlama sonrasi kenar")
    ap.add_argument("--skip-pca", action="store_true",
                    help="RAM azsa PCA baseline'ini atla")
    ap.add_argument("--rank-sweep", action="store_true",
                    help="ek olarak rank taramasi yap (yavas)")
    ap.add_argument("--threads", type=int, default=config.N_THREADS or 0)
    ap.add_argument("--tag", default="", help="cikti dosyalarina eklenecek etiket")
    args = ap.parse_args()

    if args.threads:
        set_threads(args.threads)

    deriv = Path(args.deriv_dir)
    x_path = deriv / f"X_{args.size}.npy"
    m_path = deriv / f"meta_{args.size}.csv"
    if not x_path.exists():
        print(f"HATA: {x_path} yok. Once step1_build_tensor_dataset.py calistir.")
        return 1

    tag = args.tag or args.task
    print("=" * 74)
    print("ADIM 2: Tucker (MPCA) + MLP -- capraz dogrulamali degerlendirme")
    print("=" * 74)
    print(f"NN backend  : {backend_name()}")

    X_all = np.load(x_path)
    meta_all = pd.read_csv(m_path)
    print(f"Tensor      : {X_all.shape}  ({X_all.nbytes / 1024**2:.0f} MB)")

    meta, class_names = apply_task(meta_all, config.TASKS[args.task])
    sel = meta_all["participant_id"].isin(meta["participant_id"]).values
    X = X_all[sel]
    del X_all
    y = meta["label"].values.astype(int)
    n_classes = len(class_names)

    print(f"Gorev       : {args.task}  -> siniflar {class_names}")
    for i, c in enumerate(class_names):
        print(f"   {c:<10s} n = {int((y == i).sum())}")
    print(f"Tucker rank : {tuple(args.ranks)}  -> {int(np.prod(args.ranks))} ozellik")
    print(f"CV          : {args.splits}-fold x {args.repeats} tekrar")

    min_count = int(np.bincount(y).min())
    if min_count < args.splits:
        print(f"HATA: en kucuk sinifta {min_count} ornek var, "
              f"{args.splits}-fold yapilamaz.")
        return 1

    pool_factor = max(1, args.size // args.pool_target)

    # ---------------- betimleyici figurler --------------------------------
    print("\nBetimleyici figurler uretiliyor ...")
    viz.plot_class_means(X, y, class_names,
                         config.FIG_DIR / f"fig03_class_means_{tag}.png")

    cv = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats,
                                 random_state=args.seed)
    folds = list(cv.split(X, y))
    tr0, te0 = folds[0]

    vis_mpca = MPCA(ranks=tuple(args.ranks), n_iter=config.MPCA_N_ITER).fit(X[tr0])
    viz.plot_mpca_spectra(vis_mpca, config.FIG_DIR / f"fig04_mpca_spectra_{tag}.png")
    viz.plot_eigen_volumes(vis_mpca, config.FIG_DIR / f"fig07_eigen_volumes_{tag}.png")
    viz.plot_embedding(vis_mpca.features(X), y, class_names,
                       config.FIG_DIR / f"fig08_embedding_{tag}.png")

    sweep_ranks = [r for r in (2, 4, 6, 8, 10, 12, 16) if r <= args.size]
    errs, energies, mpca_by_rank = [], [], {}
    for r in sweep_ranks:
        m = MPCA(ranks=(r, r, r), n_iter=config.MPCA_N_ITER).fit(X[tr0])
        errs.append(m.reconstruction_error(X[te0]))
        energies.append(m.explained_ratio_)
        if r in (2, 6, 10, 16):
            mpca_by_rank[r] = m
    viz.plot_rank_vs_error(sweep_ranks, errs, energies,
                           config.FIG_DIR / f"fig05_rank_vs_error_{tag}.png")
    viz.plot_reconstructions(X, mpca_by_rank, sample_idx=0,
                             path=config.FIG_DIR / f"fig06_reconstruction_{tag}.png")
    pd.DataFrame({"rank": sweep_ranks, "test_recon_error": errs,
                  "train_explained_energy": energies}).to_csv(
        config.TAB_DIR / f"compression_{tag}.csv", index=False)

    # ---------------- ana CV dongusu -------------------------------------
    print(f"\n{len(folds)} fold calistiriliyor ...")
    rows, cms, roc_data = [], {}, {}
    y_true_acc, y_pred_acc = {}, {}
    pcf = {}
    history_example = None
    t_start = time.time()

    for fi, (tr, te) in enumerate(folds):
        seed = args.seed + fi
        res, extras = run_fold(
            X[tr], X[te], y[tr], y[te], n_classes, tuple(args.ranks),
            config.MLP_PARAMS, seed, args.pca_components, pool_factor,
            skip_pca=args.skip_pca, record_history=(fi == 0))
        if fi == 0:
            history_example = extras.get("history")

        for name, (pred, proba) in res.items():
            m = compute_metrics(y[te], pred, proba, n_classes)
            m.update(method=name, fold=fi,
                     repeat=fi // args.splits, split=fi % args.splits,
                     n_train=len(tr), n_test=len(te))
            if name == REFERENCE:
                m["mpca_explained"] = extras["mpca_explained"]
                m["mpca_recon_err_test"] = extras["mpca_recon_err_test"]
                m["mpca_fit_sec"] = extras["mpca_time"]
            rows.append(m)

            y_true_acc.setdefault(name, []).append(y[te])
            y_pred_acc.setdefault(name, []).append(pred)
            pcf.setdefault(name, []).append(per_class_f1(y[te], pred, n_classes))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y[te], proba[:, 1])
                grid = np.linspace(0, 1, 101)
                roc_data.setdefault(name, {"tprs": [], "aucs": []})
                roc_data[name]["tprs"].append(np.interp(grid, fpr, tpr))
                roc_data[name]["aucs"].append(m["auc"])

        el = time.time() - t_start
        ref_f1 = next(r["f1_macro"] for r in reversed(rows)
                      if r["method"] == REFERENCE)
        print(f"  fold {fi + 1:2d}/{len(folds)}  "
              f"{REFERENCE} F1={ref_f1:.3f}  "
              f"gecen {el:6.1f}s  kalan ~{el / (fi + 1) * (len(folds) - fi - 1):6.1f}s",
              flush=True)

    df = pd.DataFrame(rows)
    methods = [m for m in ["Tucker+MLP", "Tucker+LogReg", "PCA+MLP",
                           "Voxel-MLP", "Dummy"] if m in set(df["method"])]
    df["method"] = pd.Categorical(df["method"], categories=methods, ordered=True)
    df = df.sort_values(["method", "fold"]).reset_index(drop=True)
    df.to_csv(config.TAB_DIR / f"fold_metrics_{tag}.csv", index=False)

    # ---------------- ozet tablo -----------------------------------------
    summary = []
    for m in methods:
        sub = df[df["method"] == m]
        row = {"method": m}
        for met in METRIC_NAMES:
            mean, lo, hi = mean_ci(sub[met].values)
            row[met] = mean
            row[f"{met}_sd"] = float(np.nanstd(sub[met].values, ddof=1))
            row[f"{met}_ci95"] = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "-"
        summary.append(row)
    sm = pd.DataFrame(summary)
    sm.to_csv(config.TAB_DIR / f"summary_{tag}.csv", index=False)

    print("\n" + "=" * 74)
    print("OZET (fold ortalamasi ± sd)")
    print("=" * 74)
    print(f"{'yontem':<16s} {'accuracy':>16s} {'bal.acc':>16s} "
          f"{'F1 macro':>16s} {'AUC':>16s}")
    for _, r in sm.iterrows():
        print(f"{r['method']:<16s} "
              f"{r['accuracy']:>10.3f}±{r['accuracy_sd']:.3f} "
              f"{r['balanced_accuracy']:>10.3f}±{r['balanced_accuracy_sd']:.3f} "
              f"{r['f1_macro']:>10.3f}±{r['f1_macro_sd']:.3f} "
              f"{r['auc']:>10.3f}±{r['auc_sd']:.3f}")

    # ---------------- sinif bazli F1 --------------------------------------
    pc_rows = []
    for m in methods:
        arr = np.vstack(pcf[m])
        for i, c in enumerate(class_names):
            pc_rows.append({"method": m, "class": c,
                            "f1_mean": arr[:, i].mean(),
                            "f1_sd": arr[:, i].std(ddof=1)})
    pd.DataFrame(pc_rows).to_csv(config.TAB_DIR / f"per_class_f1_{tag}.csv",
                                 index=False)

    # ---------------- istatistiksel testler --------------------------------
    n_train = int(df["n_train"].iloc[0])
    n_test = int(df["n_test"].iloc[0])
    stat_rows = []
    for met in ["f1_macro", "balanced_accuracy", "accuracy"]:
        ref = df[df["method"] == REFERENCE].sort_values("fold")[met].values
        for m in methods:
            if m == REFERENCE:
                continue
            oth = df[df["method"] == m].sort_values("fold")[met].values
            t, p_t = corrected_paired_ttest(ref - oth, n_train, n_test)
            w, p_w = wilcoxon_test(ref, oth)
            stat_rows.append({
                "metric": met, "A": REFERENCE, "B": m,
                "mean_A": ref.mean(), "mean_B": oth.mean(),
                "mean_diff": ref.mean() - oth.mean(),
                "cohens_d": cohens_d_paired(ref, oth),
                "t_corrected": t, "p_corrected_ttest": p_t,
                "wilcoxon_W": w, "p_wilcoxon": p_w,
            })
    st = pd.DataFrame(stat_rows)
    for met in st["metric"].unique():
        mask = st["metric"] == met
        st.loc[mask, "p_ttest_holm"] = holm_bonferroni(
            st.loc[mask, "p_corrected_ttest"].values)
        st.loc[mask, "p_wilcoxon_holm"] = holm_bonferroni(
            st.loc[mask, "p_wilcoxon"].values)
    st.to_csv(config.TAB_DIR / f"statistical_tests_{tag}.csv", index=False)

    print("\n" + "=" * 74)
    print(f"ISTATISTIKSEL KARSILASTIRMA -- {REFERENCE} vs digerleri (f1_macro)")
    print("=" * 74)
    sub = st[st["metric"] == "f1_macro"]
    print(f"{'karsilastirma':<28s} {'fark':>8s} {'d':>7s} "
          f"{'p(t,duz.)':>11s} {'p(Wilcoxon)':>12s}")
    for _, r in sub.iterrows():
        print(f"{REFERENCE + ' vs ' + r['B']:<28s} "
              f"{r['mean_diff']:>8.3f} {r['cohens_d']:>7.2f} "
              f"{r['p_ttest_holm']:>11.4f} {r['p_wilcoxon_holm']:>12.4f}")
    print("\n(p(t,duz.) = Nadeau-Bengio duzeltilmis eslesmis t-testi + Holm)")

    # ---------------- figurler --------------------------------------------
    print("\nSonuc figurleri uretiliyor ...")
    if history_example:
        viz.plot_training_curve(history_example,
                                config.FIG_DIR / f"fig09_training_curve_{tag}.png")
    cm_list = [aggregate_confusion(y_true_acc[m], y_pred_acc[m], n_classes)
               for m in methods]
    viz.plot_confusion_matrices(cm_list, methods, class_names,
                                config.FIG_DIR / f"fig10_confusion_{tag}.png")
    viz.plot_metric_boxplots(df, ["f1_macro", "balanced_accuracy", "auc"],
                             config.FIG_DIR / f"fig11_boxplots_{tag}.png")
    if n_classes == 2 and roc_data:
        grid = np.linspace(0, 1, 101)
        viz.plot_roc(grid,
                     {m: roc_data[m]["tprs"] for m in methods if m in roc_data},
                     {m: roc_data[m]["aucs"] for m in methods if m in roc_data},
                     config.FIG_DIR / f"fig12_roc_{tag}.png")

    # ---------------- opsiyonel rank taramasi -----------------------------
    if args.rank_sweep:
        print("\nRank taramasi (kisa CV) ...")
        sweep_res = []
        cv2 = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=1,
                                      random_state=args.seed)
        for r in config.RANK_SWEEP:
            if r > args.size:
                continue
            scores = []
            for fi, (tr, te) in enumerate(cv2.split(X, y)):
                mp = MPCA(ranks=(r, r, r), n_iter=config.MPCA_N_ITER).fit(X[tr])
                A, B = fit_scaler_pair(mp.features(X[tr]), mp.features(X[te]))
                mdl, _ = train_mlp(A, y[tr], n_classes, config.MLP_PARAMS,
                                   seed=args.seed + fi)
                pr = predict_proba(mdl, B)
                scores.append(compute_metrics(y[te], pr.argmax(1), pr,
                                              n_classes)["f1_macro"])
            sweep_res.append({"rank": r, "n_features": r ** 3,
                              "mean": float(np.mean(scores)),
                              "std": float(np.std(scores, ddof=1))})
            print(f"  rank {r:2d} ({r**3:5d} ozellik): "
                  f"F1={sweep_res[-1]['mean']:.3f}±{sweep_res[-1]['std']:.3f}")
        pd.DataFrame(sweep_res).to_csv(
            config.TAB_DIR / f"rank_sweep_{tag}.csv", index=False)
        viz.plot_rank_sweep(sweep_res,
                            config.FIG_DIR / f"fig13_rank_sweep_{tag}.png")

    # ---------------- rapor ------------------------------------------------
    meta_info = {
        "task": args.task, "classes": class_names,
        "n_subjects": int(len(y)),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(class_names)},
        "tensor_shape": list(X.shape), "tucker_ranks": list(args.ranks),
        "n_tucker_features": int(np.prod(args.ranks)),
        "cv": f"{args.splits}-fold x {args.repeats}",
        "nn_backend": backend_name(), "mlp_params": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in config.MLP_PARAMS.items()},
        "seed": args.seed,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    (config.RESULTS_DIR / f"run_info_{tag}.json").write_text(
        json.dumps(meta_info, indent=2), encoding="utf-8")

    def _table(frame, fmt=".3f"):
        try:                       # `tabulate` kuruluysa markdown tablo
            return frame.to_markdown(index=False, floatfmt=fmt)
        except Exception:          # degilse duz metin
            return "```\n" + frame.to_string(index=False) + "\n```"

    lines = [f"# Sonuclar -- gorev: {args.task}", "",
             f"- Denek sayisi: {len(y)}  {meta_info['class_counts']}",
             f"- Tensor: {tuple(X.shape)}, Tucker rank {tuple(args.ranks)} "
             f"({int(np.prod(args.ranks))} ozellik)",
             f"- CV: {args.splits}-fold x {args.repeats} tekrar, seed {args.seed}",
             f"- NN backend: {backend_name()}", "", "## Ozet", "",
             _table(sm[["method"] + METRIC_NAMES]),
             "", "## Istatistiksel testler (f1_macro)", "",
             _table(sub[["B", "mean_diff", "cohens_d", "p_corrected_ttest",
                         "p_ttest_holm", "p_wilcoxon", "p_wilcoxon_holm"]], ".4f")]
    (config.RESULTS_DIR / f"report_{tag}.md").write_text("\n".join(lines),
                                                         encoding="utf-8")

    print(f"\nToplam sure: {time.time() - t_start:.1f} s")
    print(f"Tablolar : {config.TAB_DIR}")
    print(f"Figurler : {config.FIG_DIR}")
    print(f"Rapor    : {config.RESULTS_DIR / f'report_{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
