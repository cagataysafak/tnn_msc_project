#!/usr/bin/env python3
"""
ADIM 3 -- Karistirici (confound) ve secim yanliligi kontrolleri.

step2 bir performans sayisi verir; bu script o sayinin NEREDEN geldigini sorar.
Tez savunmasinda sorulacak dort soruyu yanitlar:

  A) Hayalet artefakti elemesi tani gruplarina esit mi dagilmis?
     (esit degilse SECIM YANLILIGI var -> --keep-ghost ile duyarlilik analizi sart)

  B) Gruplar yas / cinsiyet / tarayici bakimindan dengeli mi?
     (degilse model tanidan cok demografiyi ogreniyor olabilir)

  C) SADECE demografiyle (yas + cinsiyet + tarayici, hic goruntu yok) ayni
     CV fold'larinda ne kadar basari alinir? Tucker+MLP bunu geciyor mu?
     -> Goruntunun demografinin USTUNE ne kattigini olcen tek durust test.

  D) Tucker cekirdek ozellikleri neyi kodluyor? Onlardan yas / cinsiyet /
     tarayici ne kadar iyi tahmin edilebiliyor?
     (yas cok iyi tahmin ediliyorsa ozellikler buyuk olcude yas tasiyor)

Kullanim (step1 ve step2 calistirildiktan SONRA):
    python step3_confound_checks.py --bids-dir data/ds000030 --task schz_vs_control
    python step3_confound_checks.py --bids-dir data/ds000030 --task 4class
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tnn import config
from tnn.data import apply_task, build_subject_table
from tnn.evaluation import (compute_metrics, corrected_paired_ttest,
                            wilcoxon_test)
from tnn.nn_utils import predict_proba, train_mlp
from tnn.tensor_utils import MPCA


def _hdr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _fmt_p(p) -> str:
    if not np.isfinite(p):
        return "  n/a"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ==========================================================================
# A) Hayalet artefakti x tani ilişkisi
# ==========================================================================
def check_ghost_bias(bids_dir, out_lines, tag="run"):
    _hdr("A) HAYALET ARTEFAKTI ELEMESI TANIYA GORE YANLI MI?")

    tbl = build_subject_table(bids_dir, exclude_ghost=False)
    tbl = tbl.loc[tbl["has_t1w_file"]].copy()
    if tbl.empty:
        print("T1w dosyasi bulunamadi, atlaniyor.")
        return None

    # DIKKAT: exclude_ghost=False cagrildiginda `ghost_ok` hepsi True olur.
    # Bayragi ham `ghost` sutunundan, step1'deki mantikla ayni sekilde uret.
    is_ghost = (tbl["ghost"].astype(str).str.strip().str.lower() == "ghost")
    if not is_ghost.any():
        print("participants.tsv'de hayalet isaretli denek yok "
              "(ghost sutunu bos veya farkli kodlanmis) -- kontrol atlaniyor.")
        return None
    tbl["ghost_flag"] = np.where(is_ghost, "hayalet", "temiz")
    ct = pd.crosstab(tbl["diagnosis"], tbl["ghost_flag"])
    for c in ("temiz", "hayalet"):
        if c not in ct:
            ct[c] = 0
    ct = ct[["temiz", "hayalet"]]
    ct["toplam"] = ct.sum(axis=1)
    ct["hayalet_%"] = (100 * ct["hayalet"] / ct["toplam"]).round(1)

    print(f"T1w dosyasi olan {len(tbl)} denek:\n")
    print(ct.to_string())

    obs = ct[["temiz", "hayalet"]].values
    keep = obs.sum(axis=1) > 0
    obs = obs[keep]
    chi2 = p = np.nan
    if obs.shape[0] >= 2 and obs.sum() > 0 and (obs.sum(axis=0) > 0).all():
        chi2, p, dof, _ = stats.chi2_contingency(obs)
        print(f"\nki-kare bagimsizlik testi: chi2={chi2:.3f}, "
              f"df={dof}, p={_fmt_p(p)}")
        if p < 0.05:
            print("  -> YANLI. Eleme tanidan bagimsiz DEGIL. Bu bir secim")
            print("     yanliligidir ve raporda sinirlilik olarak yazilmali.")
            print("     `--keep-ghost` ile duyarlilik analizi yapmak SART.")
        else:
            print("  -> Anlamli yanlilik saptanmadi (ama guc dusuk olabilir).")

    print("\nOrneklem paylarindaki degisim (eleme oncesi -> sonrasi):")
    inc = tbl.loc[~is_ghost, "diagnosis"].value_counts()
    allc = tbl["diagnosis"].value_counts()
    for g in allc.index:
        before = 100 * allc[g] / allc.sum()
        after = 100 * inc.get(g, 0) / max(inc.sum(), 1)
        print(f"  {g:<10s} {allc[g]:>4d} -> {inc.get(g, 0):>4d}   "
              f"pay %{before:.1f} -> %{after:.1f}")

    ct.to_csv(config.TAB_DIR / f"confound_ghost_{tag}.csv")
    out_lines += ["## A) Hayalet artefakti elemesi", "",
                  "```", ct.to_string(), "```", "",
                  f"ki-kare: chi2={chi2:.3f}, p={_fmt_p(p)}", ""]
    return ct


# ==========================================================================
# B) Demografik denge
# ==========================================================================
def _sample_table(meta, class_names):
    """Tablo 1: ornek karakteristikleri (n, yas, cinsiyet, tarayici)."""
    rows = []
    for i, name in enumerate(class_names):
        g = meta.loc[meta["label"] == i]
        age = pd.to_numeric(g["age"], errors="coerce").dropna()
        r = {"grup": name, "n": int(len(g)),
             "yas_ort": round(float(age.mean()), 1) if len(age) else np.nan,
             "yas_sd": round(float(age.std(ddof=1)), 1) if len(age) > 1 else np.nan}
        for col, label in (("gender", "cinsiyet"), ("scanner", "tarayici")):
            if col in g:
                # kategorileri alfabetik sirala -> satirlar arasi tutarli gorunum
                vc = g[col].astype(str).value_counts().sort_index()
                r[label] = " / ".join(f"{k}:{v}" for k, v in vc.items())
        rows.append(r)
    return pd.DataFrame(rows)


def check_balance(meta, class_names, out_lines, tag="run"):
    _hdr("B) GRUPLAR YAS / CINSIYET / TARAYICI BAKIMINDAN DENGELI MI?")

    rows = []
    groups = [meta.loc[meta["label"] == i] for i in range(len(class_names))]

    ages = [g["age"].dropna().values for g in groups]
    print("Yas:")
    for name, a in zip(class_names, ages):
        if len(a):
            print(f"  {name:<10s} n={len(a):>3d}  ortalama={a.mean():5.1f}  "
                  f"sd={a.std(ddof=1):4.1f}  aralik=[{a.min():.0f}, {a.max():.0f}]")
    usable = [a for a in ages if len(a) > 1]
    if len(usable) >= 2:
        H, p = stats.kruskal(*usable)
        print(f"  Kruskal-Wallis: H={H:.3f}, p={_fmt_p(p)}"
              + ("   <- GRUPLAR YAS BAKIMINDAN FARKLI" if p < 0.05 else ""))
        rows.append({"degisken": "yas", "test": "Kruskal-Wallis",
                     "istatistik": H, "p": p})

    for var, label in (("gender", "Cinsiyet"), ("scanner", "Tarayici")):
        if var not in meta or meta[var].nunique() < 2:
            continue
        ct = pd.crosstab(meta["label_name"], meta[var])
        print(f"\n{label}:")
        print("  " + ct.to_string().replace("\n", "\n  "))
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, _ = stats.chi2_contingency(ct.values)
            print(f"  ki-kare: chi2={chi2:.3f}, df={dof}, p={_fmt_p(p)}"
                  + (f"   <- GRUPLAR {label.upper()} BAKIMINDAN FARKLI"
                     if p < 0.05 else ""))
            rows.append({"degisken": var, "test": "ki-kare",
                         "istatistik": chi2, "p": p})

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(config.TAB_DIR / f"confound_balance_{tag}.csv", index=False)
    _sample_table(meta, class_names).to_csv(
        config.TAB_DIR / f"sample_characteristics_{tag}.csv", index=False)
    out_lines += ["## B) Demografik denge", "",
                  "```", df.to_string(index=False) if len(df) else "-", "```", ""]
    return df


# ==========================================================================
# C) Sadece demografiyle baseline
# ==========================================================================
def build_demo_features(meta):
    """Yas + cinsiyet + tarayici -> sayisal ozellik matrisi."""
    age = meta[["age"]].astype(float).values
    age = SimpleImputer(strategy="median").fit_transform(age)
    parts = [age]
    names = ["age"]
    for var in ("gender", "scanner"):
        if var in meta and meta[var].nunique() > 1:
            try:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:                      # eski sklearn
                enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
            arr = enc.fit_transform(meta[[var]].astype(str).values)
            parts.append(arr)
            names += [f"{var}={c}" for c in enc.categories_[0]]
    return np.hstack(parts).astype(np.float32), names


def check_demographic_baseline(meta, y, class_names, folds, tag, out_lines):
    _hdr("C) SADECE DEMOGRAFI (GORUNTU YOK) NE KADAR BASARILI?")

    D, names = build_demo_features(meta)
    print(f"Ozellikler ({D.shape[1]}): {', '.join(names)}")
    n_classes = len(class_names)

    scores = {"Demografi+LogReg": [], "Demografi+MLP": [], "Dummy(cogunluk)": []}
    for fi, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(D[tr])
        A, B = sc.transform(D[tr]), sc.transform(D[te])

        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A, y[tr])
        p = lr.predict_proba(B)
        scores["Demografi+LogReg"].append(
            compute_metrics(y[te], p.argmax(1), p, n_classes))

        mdl, _ = train_mlp(A.astype(np.float32), y[tr], n_classes,
                           config.MLP_PARAMS, seed=config.RANDOM_STATE + fi)
        p = predict_proba(mdl, B.astype(np.float32))
        scores["Demografi+MLP"].append(
            compute_metrics(y[te], p.argmax(1), p, n_classes))

        dm = DummyClassifier(strategy="most_frequent").fit(A, y[tr])
        p = dm.predict_proba(B)
        scores["Dummy(cogunluk)"].append(
            compute_metrics(y[te], p.argmax(1), p, n_classes))

    print(f"\n{'yontem':<20s} {'accuracy':>16s} {'bal.acc':>16s} {'F1 macro':>16s}")
    summary = {}
    for name, lst in scores.items():
        d = pd.DataFrame(lst)
        summary[name] = d
        print(f"{name:<20s} "
              f"{d['accuracy'].mean():>10.3f}±{d['accuracy'].std(ddof=1):.3f} "
              f"{d['balanced_accuracy'].mean():>10.3f}"
              f"±{d['balanced_accuracy'].std(ddof=1):.3f} "
              f"{d['f1_macro'].mean():>10.3f}±{d['f1_macro'].std(ddof=1):.3f}")

    # ---- step2 sonuclariyla eslesmis karsilastirma --------------------
    fm_path = config.TAB_DIR / f"fold_metrics_{tag}.csv"
    if not fm_path.exists():
        print(f"\n({fm_path} yok; step2'yi bu gorev icin once calistir.)")
        out_lines += ["## C) Demografi baseline", "", "(step2 sonucu yok)", ""]
        return summary

    fm = pd.read_csv(fm_path)
    n_train = int(fm["n_train"].iloc[0])
    n_test = int(fm["n_test"].iloc[0])

    print("\nEslesmis karsilastirma (macro F1, ayni fold'lar):")
    print(f"{'karsilastirma':<40s} {'fark':>8s} {'p(t,duz.)':>11s} "
          f"{'p(Wilc.)':>10s}")
    cmp_rows = []
    for img_method in ["Tucker+MLP", "Tucker+LogReg", "Voxel-MLP"]:
        sub = fm[fm["method"] == img_method].sort_values("fold")
        if sub.empty:
            continue
        img = sub["f1_macro"].values
        for demo_name in ["Demografi+LogReg", "Demografi+MLP"]:
            dem = summary[demo_name]["f1_macro"].values
            if len(dem) != len(img):
                continue
            t, p_t = corrected_paired_ttest(img - dem, n_train, n_test)
            _, p_w = wilcoxon_test(img, dem)
            diff = img.mean() - dem.mean()
            print(f"{img_method + ' - ' + demo_name:<40s} {diff:>8.3f} "
                  f"{_fmt_p(p_t):>11s} {_fmt_p(p_w):>10s}")
            cmp_rows.append({"goruntu_yontemi": img_method,
                             "demografi_yontemi": demo_name,
                             "F1_goruntu": img.mean(), "F1_demografi": dem.mean(),
                             "fark": diff, "p_corrected_ttest": p_t,
                             "p_wilcoxon": p_w})

    cdf = pd.DataFrame(cmp_rows)
    if len(cdf):
        cdf.to_csv(config.TAB_DIR / f"confound_image_vs_demo_{tag}.csv", index=False)

        # DIKKAT: karar EN GUCLU demografi baseline'ina gore verilmeli.
        # En buyuk farki secmek, en zayif baseline'i secmek anlamina gelir
        # ve yaniltici bir "goruntu kazandi" sonucu uretir.
        demo_means = cdf.groupby("demografi_yontemi")["F1_demografi"].mean()
        best_demo = demo_means.idxmax()
        print(f"\nEn guclu demografi baseline: {best_demo} "
              f"(macro F1 = {demo_means[best_demo]:.3f})")
        sub_best = cdf[cdf["demografi_yontemi"] == best_demo]
        best_img = sub_best.loc[sub_best["fark"].idxmax()]
        print(f"En iyi goruntu yontemi       : {best_img['goruntu_yontemi']} "
              f"(macro F1 = {best_img['F1_goruntu']:.3f})")
        print(f"Fark                         : {best_img['fark']:+.3f}  "
              f"(p_duz={_fmt_p(best_img['p_corrected_ttest'])}, "
              f"p_Wilc={_fmt_p(best_img['p_wilcoxon'])})")
        print()
        if best_img["fark"] <= 0:
            print("  -> UYARI: hicbir goruntu yontemi en guclu demografi")
            print("     baseline'ini GECEMIYOR. Bildirilen basari buyuk olcude")
            print("     demografik/teknik farklardan kaynaklaniyor. MR'in ek")
            print("     katkisi gosterilemedi -- bu bir sinirlilik olarak")
            print("     raporlanmali. step4_deconfound.py ile devam et.")
        elif best_img["p_corrected_ttest"] > 0.05:
            print("  -> Goruntu daha iyi ama fark anlamli degil. Durust ifade:")
            print("     'MR'in demografinin uzerine olculebilir bir katki")
            print("     sagladigi gosterilemedi.' step4_deconfound.py ile devam et.")
        else:
            print("  -> Goruntu, en guclu demografi baseline'ini anlamli olarak")
            print("     geciyor. Bulgu demografik karistiricilarla aciklanamiyor.")

    out_lines += ["## C) Demografi baseline vs goruntu", "",
                  "```", cdf.to_string(index=False) if len(cdf) else "-",
                  "```", ""]
    return summary


# ==========================================================================
# D) Tucker ozellikleri neyi kodluyor?
# ==========================================================================
def check_what_features_encode(X, meta, y, class_names, ranks, folds, tag,
                               out_lines):
    _hdr("D) TUCKER OZELLIKLERI NEYI KODLUYOR?")

    tr0 = folds[0][0]
    mpca = MPCA(ranks=ranks, n_iter=config.MPCA_N_ITER).fit(X[tr0])
    F = mpca.features(X)
    print(f"Ozellik matrisi: {F.shape}  "
          f"(aciklanan enerji {mpca.explained_ratio_:.3f})")
    Fs = StandardScaler().fit_transform(F)

    rows = []

    # --- yas: ridge regresyon, CV R^2 ---------------------------------
    age = pd.to_numeric(meta["age"], errors="coerce").values
    ok = np.isfinite(age)
    if ok.sum() > 30:
        ridge = RidgeCV(alphas=np.logspace(-1, 5, 25))
        pred = cross_val_predict(ridge, Fs[ok], age[ok], cv=5)
        ss_res = float(((age[ok] - pred) ** 2).sum())
        ss_tot = float(((age[ok] - age[ok].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        mae = float(np.abs(age[ok] - pred).mean())
        r = float(np.corrcoef(age[ok], pred)[0, 1])
        print(f"\nYas tahmini (ridge, 5-fold CV): R^2={r2:.3f}  "
              f"MAE={mae:.1f} yil  r={r:.3f}")
        if r2 > 0.25:
            print("  -> Ozellikler YAS bilgisini belirgin sekilde tasiyor.")
            print("     Gruplar yasca farkliysa bu bir karistiricidir;")
            print("     yasi regresyonla cikarmayi (deconfounding) dusun.")
        else:
            print("  -> Yas bilgisi zayif; yas karistiricisi riski dusuk.")
        rows.append({"hedef": "yas", "olcut": "CV R^2", "deger": r2})

    # --- cinsiyet / tarayici: siniflandirma --------------------------
    for var, label in (("gender", "cinsiyet"), ("scanner", "tarayici")):
        if var not in meta or meta[var].nunique() < 2:
            continue
        vals = meta[var].astype(str).values
        cats, codes = np.unique(vals, return_inverse=True)
        if len(cats) < 2 or np.bincount(codes).min() < 10:
            continue
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        pred = cross_val_predict(lr, Fs, codes, cv=5)
        from sklearn.metrics import balanced_accuracy_score
        ba = balanced_accuracy_score(codes, pred)
        chance = 1.0 / len(cats)
        print(f"\n{label.capitalize()} tahmini ({len(cats)} sinif): "
              f"dengeli dogruluk={ba:.3f}  (sans={chance:.3f})")
        if ba > 0.70:
            print(f"  -> Ozellikler {label} bilgisini guclu sekilde tasiyor.")
        rows.append({"hedef": label, "olcut": "dengeli dogruluk", "deger": ba})

    # --- kiyas: taniyi ne kadar tasiyor? ------------------------------
    from sklearn.metrics import balanced_accuracy_score
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    pred = cross_val_predict(lr, Fs, y, cv=5)
    ba_dx = balanced_accuracy_score(y, pred)
    print(f"\nKIYAS -- tani tahmini: dengeli dogruluk={ba_dx:.3f}  "
          f"(sans={1 / len(class_names):.3f})")
    rows.append({"hedef": "tani", "olcut": "dengeli dogruluk", "deger": ba_dx})

    df = pd.DataFrame(rows)
    df.to_csv(config.TAB_DIR / f"confound_feature_content_{tag}.csv", index=False)
    print("\nYorum: tani icin elde edilen deger, yas/cinsiyet/tarayici icin")
    print("elde edilenlerden belirgin dusukse, ozellikler taniyi degil")
    print("agirlikla bu nuisance degiskenleri kodluyor demektir.")

    out_lines += ["## D) Ozellikler neyi kodluyor?", "",
                  "```", df.to_string(index=False), "```", ""]

    # --- figur ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        labels = df["hedef"].tolist()
        vals = df["deger"].tolist()
        colors = ["#C44E52" if l != "tani" else "#4C72B0" for l in labels]
        ax.barh(labels, vals, color=colors, edgecolor="black", lw=0.6)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("CV R^2 (yas) / dengeli dogruluk (digerleri)")
        ax.set_title("Sekil 14 -- Tucker ozellikleri neyi kodluyor?\n"
                     "(mavi = hedef degisken, kirmizi = karistiricilar)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        out = config.FIG_DIR / f"fig14_feature_content_{tag}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"\nFigur: {out}")
    except Exception as exc:
        print(f"(figur uretilemedi: {exc})")

    return df


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Karistirici kontrolleri")
    ap.add_argument("--bids-dir", default=str(config.BIDS_DIR))
    ap.add_argument("--deriv-dir", default=str(config.DERIV_DIR))
    ap.add_argument("--size", type=int, default=config.TARGET_SIZE)
    ap.add_argument("--task", default=config.DEFAULT_TASK, choices=list(config.TASKS))
    ap.add_argument("--ranks", type=int, nargs=3, default=list(config.DEFAULT_RANKS))
    ap.add_argument("--splits", type=int, default=config.N_SPLITS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--skip-ghost-check", action="store_true")
    ap.add_argument("--tag", default="",
                    help="cikti dosyalarina eklenecek etiket "
                         "(ornek: --keep-ghost kosusu icin schz_keepghost)")
    args = ap.parse_args()

    deriv = Path(args.deriv_dir)
    x_path = deriv / f"X_{args.size}.npy"
    m_path = deriv / f"meta_{args.size}.csv"
    if not x_path.exists():
        print(f"HATA: {x_path} yok. Once step1_build_tensor_dataset.py calistir.")
        return 1

    tag = args.tag or args.task
    print("=" * 74)
    print(f"ADIM 3: Karistirici ve secim yanliligi kontrolleri -- {args.task}")
    print("=" * 74)

    out_lines = [f"# Karistirici kontrolleri -- gorev: {args.task}", ""]

    if not args.skip_ghost_check:
        try:
            check_ghost_bias(args.bids_dir, out_lines, tag=tag)
        except Exception as exc:
            print(f"(A adimi atlandi: {type(exc).__name__}: {exc})")

    X_all = np.load(x_path)
    meta_all = pd.read_csv(m_path)
    meta, class_names = apply_task(meta_all, config.TASKS[args.task])
    sel = meta_all["participant_id"].isin(meta["participant_id"]).values
    X = X_all[sel]
    del X_all
    y = meta["label"].values.astype(int)

    print(f"\nGorev: {args.task}, {len(y)} denek, siniflar {class_names}")

    cv = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats,
                                 random_state=args.seed)
    folds = list(cv.split(X, y))

    check_balance(meta, class_names, out_lines, tag=tag)
    check_demographic_baseline(meta, y, class_names, folds, tag, out_lines)
    check_what_features_encode(X, meta, y, class_names, tuple(args.ranks),
                               folds, tag, out_lines)

    rep = config.RESULTS_DIR / f"confound_report_{tag}.md"
    rep.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nRapor: {rep}")
    print(f"Tablolar: {config.TAB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())