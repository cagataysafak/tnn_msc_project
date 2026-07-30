#!/usr/bin/env python3
"""
ADIM 4 -- Karistiricilardan arindirilmis (deconfounded) analiz.

step3 sunu gosterdi: Tucker ozellikleri tarayici ve cinsiyet bilgisini,
taniyi kodladigindan daha iyi kodluyor; ve bazi gorevlerde sadece demografi
kullanan bir model goruntu tabanli modelleri geciyor.

Bu script tek bir soruyu yanitlar:
    "Yas, cinsiyet ve tarayici etkisi ozniteliklerden cikarildiktan sonra
     beyin goruntusunde tani hakkinda sinyal KALIYOR mu?"

Uc analiz yapar:

  1. KOVARYANT REGRESYONU (residualization)
     Her oznitelik icin  f = b0 + b1*yas + b2*cinsiyet + b3*tarayici + e
     modeli SADECE EGITIM verisiyle kurulur; siniflandirmada artik (e) kullanilir.
     Boylece dogrusal karistirici etkisi cikarilir.

  2. KARSILASTIRMALI YONTEM SETI (ayni fold'lar, eslesmis testler)
       Tucker+MLP            ham oznitelikler                    [referans]
       Tucker(arinmis)+MLP   kovaryant regresyonundan sonra
       Demo+LogReg           sadece yas+cinsiyet+tarayici
       Tucker+Demo+MLP       goruntu + demografi birlestirilmis
       Dummy(cogunluk)       en sik sinifi tahmin et

  3. TARAYICI-DISI GENELLEME (leave-one-scanner-out)
     Bir tarayicida egit, DIGER tarayicida test et. Model gercekten anatomi
     ogrendiyse tarayici degisince de calismali; site etkisi ogrendiyse cokecek.

Ayrica her yontem icin dengeli dogrulugun SANS SEVIYESINDEN anlamli olarak
yuksek olup olmadigi tek-orneklemli duzeltilmis t-testiyle sinanir.

ONEMLI KAVRAMSAL UYARI
----------------------
Tarayici ile tani bu veri setinde guclu iliskili (chi2=21.7, p<0.001).
Iliskili bir degiskeni regresyonla cikarmak, tani sinyalinin bir kismini da
birlikte siler (over-correction). Yani:
  * Arinmis modelin performansi dususe -> sinyalin ne kadari anatomik, ne
    kadari site kaynakli olduguna KARAR VERILEMEZ.
  * Arinmis model hala sanstan iyiyse -> site ile aciklanamayan bir sinyal var.
Ilk durum bu veri altkumesi icin gecerli bir SONUCTUR: "bu ornekle
karistiricisiz bir yapisal siniflandirma yapilamaz."

Kullanim (step1 + step2 sonrasi):
    python step4_deconfound.py --size 64 --task schz_vs_control
    python step4_deconfound.py --size 64 --task 4class
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tnn import config
from tnn.data import apply_task
from tnn.evaluation import (aggregate_confusion, compute_metrics,
                            corrected_paired_ttest, holm_bonferroni, mean_ci,
                            wilcoxon_test)
from tnn.nn_utils import backend_name, predict_proba, train_mlp
from tnn.tensor_utils import MPCA

REFERENCE = "Tucker+MLP"
METHODS = ["Tucker+MLP", "Tucker(arinmis)+MLP", "Demo+LogReg",
           "Tucker+Demo+MLP", "Dummy(cogunluk)"]


def _fmt_p(p) -> str:
    if not np.isfinite(p):
        return "  n/a"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ==========================================================================
# Karistirici tasarim matrisi
# ==========================================================================
def build_confound_matrix(meta: pd.DataFrame):
    """
    [1, yas, cinsiyet kuklalari, tarayici kuklalari] tasarim matrisi.

    One-hot kategorileri TUM veriden alinir; bu etiket (y) bilgisi
    icermedigi icin veri sizintisi olusturmaz, ama fold'lar arasi sutun
    tutarliligini garanti eder.
    """
    n = len(meta)
    cols = [np.ones((n, 1), dtype=np.float64)]
    names = ["intercept"]

    age = pd.to_numeric(meta["age"], errors="coerce").values.reshape(-1, 1)
    age = SimpleImputer(strategy="median").fit_transform(age)
    cols.append(age.astype(np.float64))
    names.append("age")

    for var in ("gender", "scanner"):
        if var not in meta or meta[var].nunique() < 2:
            continue
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                drop="first")
        except TypeError:                                  # eski sklearn
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore",
                                drop="first")
        arr = enc.fit_transform(meta[[var]].astype(str).values)
        cols.append(arr.astype(np.float64))
        cats = enc.categories_[0][1:]
        names += [f"{var}={c}" for c in cats]

    return np.hstack(cols), names


class ConfoundRegressor:
    """
    Kovaryant regresyonu ile arindirma.

    fit()  : SADECE egitim verisiyle  F_tr ~ C_tr  en kucuk kareler cozumu
    transform() : F - C @ beta   (egitim ve teste ayni beta uygulanir)
    """

    def __init__(self):
        self.beta_ = None

    def fit(self, F: np.ndarray, C: np.ndarray) -> "ConfoundRegressor":
        self.beta_, *_ = np.linalg.lstsq(C.astype(np.float64),
                                         F.astype(np.float64), rcond=None)
        return self

    def transform(self, F: np.ndarray, C: np.ndarray) -> np.ndarray:
        return (F.astype(np.float64) - C.astype(np.float64) @ self.beta_
                ).astype(np.float32)


def _scale(A, B):
    sc = StandardScaler().fit(A)
    return sc.transform(A).astype(np.float32), sc.transform(B).astype(np.float32)


# ==========================================================================
# Tek fold
# ==========================================================================
def run_fold(X, y, C, tr, te, n_classes, ranks, seed):
    out = {}

    mpca = MPCA(ranks=ranks, n_iter=config.MPCA_N_ITER).fit(X[tr])
    F_tr, F_te = mpca.features(X[tr]), mpca.features(X[te])

    # --- 1) ham Tucker ozellikleri ------------------------------------
    A, B = _scale(F_tr, F_te)
    mdl, _ = train_mlp(A, y[tr], n_classes, config.MLP_PARAMS, seed=seed)
    p = predict_proba(mdl, B)
    out["Tucker+MLP"] = (p.argmax(1), p)

    # --- 2) karistiricilardan arindirilmis ----------------------------
    cr = ConfoundRegressor().fit(F_tr, C[tr])
    R_tr, R_te = cr.transform(F_tr, C[tr]), cr.transform(F_te, C[te])
    A2, B2 = _scale(R_tr, R_te)
    mdl2, _ = train_mlp(A2, y[tr], n_classes, config.MLP_PARAMS, seed=seed)
    p = predict_proba(mdl2, B2)
    out["Tucker(arinmis)+MLP"] = (p.argmax(1), p)

    # --- 3) sadece demografi ------------------------------------------
    D = C[:, 1:]                                    # intercept'i at
    A3, B3 = _scale(D[tr], D[te])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A3, y[tr])
    p = lr.predict_proba(B3)
    out["Demo+LogReg"] = (p.argmax(1), p)

    # --- 4) goruntu + demografi ---------------------------------------
    A4 = np.hstack([A, A3]).astype(np.float32)
    B4 = np.hstack([B, B3]).astype(np.float32)
    mdl4, _ = train_mlp(A4, y[tr], n_classes, config.MLP_PARAMS, seed=seed)
    p = predict_proba(mdl4, B4)
    out["Tucker+Demo+MLP"] = (p.argmax(1), p)

    # --- 5) cogunluk sinifi -------------------------------------------
    dm = DummyClassifier(strategy="most_frequent").fit(A3, y[tr])
    p = dm.predict_proba(B3)
    out["Dummy(cogunluk)"] = (p.argmax(1), p)

    return out


# ==========================================================================
# Duzeltilmis guven araligi
# ==========================================================================
def corrected_ci(values, n_train, n_test, alpha: float = 0.05):
    """
    Nadeau-Bengio duzeltmesiyle %95 guven araligi.

    Duz t-araligi se = sd/sqrt(k) kullanir ve fold'larin bagimsiz oldugunu
    varsayar. Tekrarli CV'de egitim kumeleri ortustugu icin bu aralik ANTI-
    KONSERVATIFTIR (gercekte olduğundan dar cikar) ve ayni verideki
    duzeltilmis p-degeriyle CELISEBILIR. Burada varyansa ayni duzeltme
    terimi (1/k + n_test/n_train) uygulanir; boylece aralik ile p-degeri
    tutarli olur.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return (float(v.mean()) if k else np.nan), np.nan, np.nan
    m = float(v.mean())
    corr = 1.0 / k + float(n_test) / float(max(n_train, 1))
    se = float(np.sqrt(v.var(ddof=1) * corr))
    h = se * stats.t.ppf(1 - alpha / 2, k - 1)
    return m, m - h, m + h


# ==========================================================================
# Sans seviyesine karsi tek-orneklemli test
# ==========================================================================
def vs_chance(values, chance, n_train, n_test):
    """Nadeau-Bengio duzeltmesiyle: ortalama > sans mi?"""
    d = np.asarray(values, float) - chance
    d = d[np.isfinite(d)]
    k = len(d)
    if k < 2:
        return np.nan, np.nan
    var = d.var(ddof=1)
    if var <= 0:
        return (np.inf if d.mean() > 0 else 0.0), (0.0 if d.mean() > 0 else 1.0)
    corr = 1.0 / k + float(n_test) / float(max(n_train, 1))
    t = d.mean() / np.sqrt(var * corr)
    p = 1.0 - stats.t.cdf(t, df=k - 1)              # tek yonlu
    return float(t), float(p)


# ==========================================================================
def leave_one_scanner_out(X, y, meta, class_names, ranks, seed, out_lines,
                          tag="run"):
    _hdr("3) TARAYICI-DISI GENELLEME (leave-one-scanner-out)")

    if "scanner" not in meta or meta["scanner"].nunique() < 2:
        print("Tek tarayici var; bu analiz atlaniyor.")
        return None

    n_classes = len(class_names)
    scanners = sorted(meta["scanner"].astype(str).unique())
    rows = []
    print("Her satir: BIR tarayicida test, digerlerinde egitim.\n")

    for s in scanners:
        te = np.where(meta["scanner"].astype(str).values == s)[0]
        tr = np.where(meta["scanner"].astype(str).values != s)[0]
        if len(te) < 10 or len(tr) < 20:
            print(f"  tarayici {s}: yetersiz ornek (test {len(te)}, "
                  f"egitim {len(tr)}) -- atlandi")
            continue
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            print(f"  tarayici {s}: bir tarafta tek sinif var -- atlandi")
            continue

        mpca = MPCA(ranks=ranks, n_iter=config.MPCA_N_ITER).fit(X[tr])
        A, B = _scale(mpca.features(X[tr]), mpca.features(X[te]))
        mdl, _ = train_mlp(A, y[tr], n_classes, config.MLP_PARAMS, seed=seed)
        p = predict_proba(mdl, B)
        m = compute_metrics(y[te], p.argmax(1), p, n_classes)

        # ---- ONSEL KAYMASI (prior shift) duzeltmesi ------------------
        # Tarayicilarin sinif dagilimlari cok farkli. argmax karari egitim
        # onselini tasidigi icin dengeli dogruluk cokebilirken AUC saglam
        # kalir. Posterioru EGITIM onseline bolerek onselden arindiriyoruz;
        # bu islem test etiketlerini/dagilimini KULLANMAZ, yani mesru.
        prior_tr = np.bincount(y[tr], minlength=n_classes).astype(float)
        prior_tr /= prior_tr.sum()
        prior_tr[prior_tr <= 0] = 1e-9
        p_adj = p / prior_tr[None, :]
        p_adj /= np.maximum(p_adj.sum(axis=1, keepdims=True), 1e-12)
        m_adj = compute_metrics(y[te], p_adj.argmax(1), p_adj, n_classes)

        dist_tr = np.bincount(y[tr], minlength=n_classes)
        dist_te = np.bincount(y[te], minlength=n_classes)
        print(f"  tarayici {s}: egitim n={len(tr)} {dist_tr.tolist()}, "
              f"test n={len(te)} {dist_te.tolist()}")
        print(f"      bal.acc={m['balanced_accuracy']:.3f}  "
              f"(onsel-duzeltilmis {m_adj['balanced_accuracy']:.3f})  "
              f"F1={m['f1_macro']:.3f}  AUC={m['auc']:.3f}   "
              f"(sans: bal.acc {1 / n_classes:.3f}, AUC 0.500)")
        rows.append({"test_scanner": s, "n_train": len(tr), "n_test": len(te),
                     "balanced_accuracy": m["balanced_accuracy"],
                     "balanced_accuracy_prior_adj": m_adj["balanced_accuracy"],
                     "f1_macro": m["f1_macro"], "auc": m["auc"]})

    df = pd.DataFrame(rows)
    if len(df):
        mean_ba = df["balanced_accuracy"].mean()
        mean_adj = df["balanced_accuracy_prior_adj"].mean()
        mean_auc = df["auc"].mean()
        print(f"\n  Ortalama: bal.acc {mean_ba:.3f} | onsel-duzeltilmis "
              f"{mean_adj:.3f} | AUC {mean_auc:.3f}")
        print(f"  (sans: bal.acc {1 / n_classes:.3f}, AUC 0.500)")

        # AUC birincil olcut: esikten bagimsiz oldugu icin onsel kaymasindan
        # etkilenmez. bal.acc ikincil, cunku karar kurali onsele duyarli.
        print()
        if mean_auc >= 0.60:
            print("  -> AUC sanstan belirgin yuksek: model tarayici disinda da")
            print("     siniflari SIRALAYABILIYOR. Ogrenilen sinyal salt")
            print("     site imzasi DEGIL.")
            if mean_adj < 1 / n_classes + 0.05:
                print("     Ancak karar kurali genellemiyor (onsel kaymasi):")
                print("     tarayicilar arasi sinif dagilimlari cok farkli.")
                print("     Raporda birincil olcut olarak AUC'yi kullan.")
        elif mean_auc >= 0.55:
            print("  -> AUC sansin hafif uzerinde: zayif ama sifir olmayan bir")
            print("     tarayici-disi sinyal. Temkinli yorumla.")
        else:
            print("  -> AUC de sans seviyesinde: model tarayici disina")
            print("     GENELLEMIYOR. Bu, ogrenilen sinyalin buyuk olcude")
            print("     site/tarayici kaynakli oldugunun guclu kanitidir.")

        print("\n  NOT: yalnizca 2 tarayici var, yani bu analiz 2 gozleme")
        print("  dayaniyor; istatistiksel test yapilamaz, betimleyicidir.")

    if len(df):
        df.to_csv(config.TAB_DIR / f"deconf_loso_{tag}.csv", index=False)
    out_lines += ["## 3) Tarayici-disi genelleme", "",
                  "```", df.to_string(index=False) if len(df) else "-", "```", ""]
    return df


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Karistiricilardan arindirilmis analiz")
    ap.add_argument("--deriv-dir", default=str(config.DERIV_DIR))
    ap.add_argument("--size", type=int, default=config.TARGET_SIZE)
    ap.add_argument("--task", default=config.DEFAULT_TASK, choices=list(config.TASKS))
    ap.add_argument("--ranks", type=int, nargs=3, default=list(config.DEFAULT_RANKS))
    ap.add_argument("--splits", type=int, default=config.N_SPLITS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--skip-loso", action="store_true")
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
    print("=" * 78)
    print(f"ADIM 4: Arindirilmis analiz -- gorev: {args.task}")
    print("=" * 78)
    print(f"NN backend: {backend_name()}")

    X_all = np.load(x_path)
    meta_all = pd.read_csv(m_path)
    meta, class_names = apply_task(meta_all, config.TASKS[args.task])
    sel = meta_all["participant_id"].isin(meta["participant_id"]).values
    X = X_all[sel]
    del X_all
    y = meta["label"].values.astype(int)
    n_classes = len(class_names)
    chance = 1.0 / n_classes

    C, cnames = build_confound_matrix(meta)
    print(f"Denek sayisi   : {len(y)}  siniflar {class_names} "
          f"{np.bincount(y).tolist()}")
    print(f"Tucker rank    : {tuple(args.ranks)} -> {int(np.prod(args.ranks))} ozellik")
    print(f"Karistiricilar : {', '.join(cnames[1:])}")
    print(f"CV             : {args.splits}-fold x {args.repeats} tekrar")

    out_lines = [f"# Arindirilmis analiz -- gorev: {args.task}", "",
                 f"- Denek: {len(y)}, siniflar {class_names} "
                 f"{np.bincount(y).tolist()}",
                 f"- Karistiricilar: {', '.join(cnames[1:])}", ""]

    cv = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats,
                                 random_state=args.seed)
    folds = list(cv.split(X, y))

    _hdr("1-2) KOVARYANT REGRESYONU VE YONTEM KARSILASTIRMASI")
    rows, ytrue, ypred = [], {}, {}
    for fi, (tr, te) in enumerate(folds):
        res = run_fold(X, y, C, tr, te, n_classes, tuple(args.ranks),
                       args.seed + fi)
        for name, (pred, proba) in res.items():
            m = compute_metrics(y[te], pred, proba, n_classes)
            m.update(method=name, fold=fi, n_train=len(tr), n_test=len(te))
            rows.append(m)
            ytrue.setdefault(name, []).append(y[te])
            ypred.setdefault(name, []).append(pred)
        print(f"  fold {fi + 1:2d}/{len(folds)} tamam", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(config.TAB_DIR / f"deconf_fold_metrics_{tag}.csv", index=False)
    n_train = int(df["n_train"].iloc[0])
    n_test = int(df["n_test"].iloc[0])
    methods = [m for m in METHODS if m in set(df["method"])]

    print(f"\n{'yontem':<22s} {'bal.acc (%95 GA, duz.)':>28s} {'F1 macro':>16s} "
          f"{'p(sans)':>9s}")
    summary = []
    for m in methods:
        sub = df[df["method"] == m]
        # DIKKAT: duz t-araligi degil, Nadeau-Bengio duzeltilmis aralik.
        # Boylece aralik ile p(sans) ayni varsayimlari paylasir.
        ba, lo, hi = corrected_ci(sub["balanced_accuracy"].values,
                                  n_train, n_test)
        # Dengeli dogruluk tanimi geregi [0, 1] araligindadir; aralik bu
        # araligin disina tasarsa parametre uzayina kirpiyoruz (standart pratik).
        lo, hi = float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
        ba_u, lo_u, hi_u = mean_ci(sub["balanced_accuracy"].values)
        f1 = sub["f1_macro"].mean()
        f1sd = sub["f1_macro"].std(ddof=1)
        _, p_ch = vs_chance(sub["balanced_accuracy"].values, chance,
                            n_train, n_test)
        print(f"{m:<22s} {ba:>8.3f} [{lo:.3f}, {hi:.3f}] "
              f"{f1:>10.3f}±{f1sd:.3f} {_fmt_p(p_ch):>9s}")
        summary.append({"method": m, "balanced_accuracy": ba,
                        "ba_ci_low": lo, "ba_ci_high": hi,
                        "ba_ci_low_uncorrected": lo_u,
                        "ba_ci_high_uncorrected": hi_u,
                        "f1_macro": f1, "f1_macro_sd": f1sd,
                        "auc": sub["auc"].mean(),
                        "p_vs_chance": p_ch})
    sm = pd.DataFrame(summary)
    sm.to_csv(config.TAB_DIR / f"deconf_summary_{tag}.csv", index=False)
    print(f"\nSans seviyesi (dengeli dogruluk) = {chance:.3f}")
    print("p(sans): tek yonlu, Nadeau-Bengio duzeltilmis tek-orneklemli t-testi")
    print("GA    : ayni duzeltmeyle hesaplandi (fold'lar bagimsiz degil).")
    print("        Duzeltilmemis dar aralik CSV'de *_uncorrected sutunlarinda.")

    # ---- eslesmis karsilastirmalar --------------------------------------
    print(f"\n{'karsilastirma':<46s} {'fark':>8s} {'p(t,duz.)':>11s} "
          f"{'p(Wilc.)':>10s}")
    pairs = [(REFERENCE, m) for m in methods if m != REFERENCE]
    pairs += [("Tucker(arinmis)+MLP", "Dummy(cogunluk)"),
              ("Tucker+Demo+MLP", "Demo+LogReg")]
    seen, cmp_rows = set(), []
    for a, b in pairs:
        if (a, b) in seen or a not in methods or b not in methods:
            continue
        seen.add((a, b))
        va = df[df["method"] == a].sort_values("fold")["f1_macro"].values
        vb = df[df["method"] == b].sort_values("fold")["f1_macro"].values
        t, p_t = corrected_paired_ttest(va - vb, n_train, n_test)
        _, p_w = wilcoxon_test(va, vb)
        cmp_rows.append({"A": a, "B": b, "mean_A": va.mean(), "mean_B": vb.mean(),
                         "fark": va.mean() - vb.mean(),
                         "p_corrected_ttest": p_t, "p_wilcoxon": p_w})
    cdf = pd.DataFrame(cmp_rows)
    if len(cdf):
        cdf["p_ttest_holm"] = holm_bonferroni(cdf["p_corrected_ttest"].values)
        cdf["p_wilcoxon_holm"] = holm_bonferroni(cdf["p_wilcoxon"].values)
        for _, r in cdf.iterrows():
            print(f"{r['A'] + ' - ' + r['B']:<46s} {r['fark']:>8.3f} "
                  f"{_fmt_p(r['p_ttest_holm']):>11s} "
                  f"{_fmt_p(r['p_wilcoxon_holm']):>10s}")
        cdf.to_csv(config.TAB_DIR / f"deconf_tests_{tag}.csv", index=False)
        print("\n(p degerleri Holm-Bonferroni duzeltilmis)")

    # ---- otomatik yorum --------------------------------------------------
    _hdr("YORUM")
    def _get(m, col="balanced_accuracy"):
        s = sm.loc[sm["method"] == m, col]
        return float(s.iloc[0]) if len(s) else np.nan

    ba_raw = _get("Tucker+MLP")
    ba_dec = _get("Tucker(arinmis)+MLP")
    ba_demo = _get("Demo+LogReg")
    ba_both = _get("Tucker+Demo+MLP")
    p_dec = float(sm.loc[sm["method"] == "Tucker(arinmis)+MLP",
                         "p_vs_chance"].iloc[0])

    print(f"Ham Tucker            : bal.acc {ba_raw:.3f}")
    print(f"Arindirilmis Tucker   : bal.acc {ba_dec:.3f}  "
          f"(degisim {ba_dec - ba_raw:+.3f})")
    print(f"Sadece demografi      : bal.acc {ba_demo:.3f}")
    print(f"Goruntu + demografi   : bal.acc {ba_both:.3f}\n")

    if p_dec < 0.05 and ba_dec > chance:
        print("* Arindirmadan sonra dengeli dogruluk hala sanstan ANLAMLI")
        print("  olarak yuksek -> yas/cinsiyet/tarayici ile aciklanamayan bir")
        print("  yapisal sinyal var. Ana bulgun bu olabilir.")
    else:
        print("* Arindirmadan sonra dengeli dogruluk sanstan ayirt edilemiyor")
        print("  -> bu ornekte, dogrusal karistirici etkileri cikarildiginda")
        print("  olculebilir bir yapisal tani sinyali kalmiyor.")
        print("  Bu GECERLI bir bulgudur; 'yontem kotu' demek degildir.")

    if ba_both > max(ba_demo, ba_raw) + 0.02:
        print("* Goruntu + demografi, ikisinden de iyi -> goruntu demografiye")
        print("  EK bilgi tasiyor (tamamlayici).")
    elif ba_demo >= ba_both - 0.02:
        print("* Goruntu eklemek demografinin uzerine katki yapmiyor.")

    print("\nHatirlatma: tarayici ile tani iliskili oldugu icin bu arindirma")
    print("asiri-duzeltme (over-correction) yapar; asagi yonlu bir sinirdir.")

    out_lines += ["## 1-2) Yontem karsilastirmasi", "",
                  "```", sm.to_string(index=False), "```", "",
                  "```", cdf.to_string(index=False) if len(cdf) else "-",
                  "```", ""]

    # ---- LOSO ------------------------------------------------------------
    if not args.skip_loso:
        leave_one_scanner_out(X, y, meta, class_names, tuple(args.ranks),
                              args.seed, out_lines, tag=tag)

    # ---- figur -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        names = sm["method"].tolist()
        vals = sm["balanced_accuracy"].values
        err = np.vstack([vals - sm["ba_ci_low"].values,
                         sm["ba_ci_high"].values - vals])
        colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B3", "#999999"][:len(names)]
        ax.bar(range(len(names)), vals, yerr=err, capsize=5,
               color=colors, edgecolor="black", lw=0.7)
        ax.axhline(chance, color="red", ls="--", lw=1.4,
                   label=f"sans seviyesi = {chance:.3f}")
        for i, (v, p) in enumerate(zip(vals, sm["p_vs_chance"].values)):
            star = "*" if (np.isfinite(p) and p < 0.05) else ""
            ax.text(i, v + 0.02, f"{v:.3f}{star}", ha="center", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
        ax.set_ylabel("dengeli dogruluk (%95 GA, duzeltilmis)")
        ax.set_ylim(0, max(1.0, float(sm["ba_ci_high"].max()) + 0.08))
        ax.set_title("Sekil 15 -- Karistiricilardan arindirma etkisi\n"
                     "(* = sanstan anlamli yuksek, p<0.05; GA fold bagimliligina gore duzeltilmis)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = config.FIG_DIR / f"fig15_deconfound_{tag}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"\nFigur: {out}")
    except Exception as exc:
        print(f"(figur uretilemedi: {exc})")

    rep = config.RESULTS_DIR / f"deconfound_report_{tag}.md"
    rep.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Rapor: {rep}")
    print(f"Tablolar: {config.TAB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())