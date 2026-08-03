#!/usr/bin/env python3
"""
ADIM 5 -- Butun kosularin sonuclarini TEK bir teze hazir rapora topla.

step1-4 sonuclari 15+ CSV dosyasina dagilir. Bu script hepsini tarar,
birlestirir ve iki formatta yazar:

  results/THESIS_REPORT.md      -- markdown tablolar + bulgu listesi
  results/latex/*.tex           -- booktabs tablolari (\\input ile teze eklenir)

Etiketleri (tag) otomatik bulur, yani `--tag schz_keepghost` gibi ek kosular
yaptiysan onlar da rapora girer.

Onemli: bu script YENI bir analiz yapmaz, yorum URETMEZ. Yalnizca mevcut
sayilari toplar ve p-degerlerine gore kosullu, olgusal cumleler kurar.
Yorumu sen yazacaksin -- burada ciktilar hazir malzeme.

Kullanim:
    python step5_make_report.py
    python step5_make_report.py --size 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tnn import config


# ==========================================================================
# LaTeX yazici (pandas surum bagimliligi olmadan)
# ==========================================================================
def _esc(x) -> str:
    """LaTeX ozel karakterlerini kacir. pdfLaTeX uyumlu (Unicode birakmaz)."""
    s = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"),
                 # matematik modu gerektirenler
                 ("\u00b1", r"$\pm$"), ("\u2264", r"$\leq$"),
                 ("\u2265", r"$\geq$"), ("<", r"$<$"), (">", r"$>$")):
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame, caption: str, label: str,
             floatfmt: str = "{:.3f}") -> str:
    """booktabs formatinda tablo. \\usepackage{booktabs} gerekir."""
    cols = list(df.columns)
    align = "l" + "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols[1:])
    lines = [r"\begin{table}[htbp]", r"  \centering",
             f"  \\caption{{{_esc(caption)}}}", f"  \\label{{{label}}}",
             r"  \small",
             f"  \\begin{{tabular}}{{{align}}}", r"    \toprule",
             "    " + " & ".join(_esc(c) for c in cols) + r" \\",
             r"    \midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(_esc(v))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _read(path: Path):
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


def _sig(p, alpha: float = 0.05) -> str:
    if p is None or not np.isfinite(p):
        return "belirsiz"
    return "ANLAMLI" if p < alpha else "anlamli degil"


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Teze hazir rapor uret")
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    ap.add_argument("--deriv-dir", default=str(config.DERIV_DIR))
    ap.add_argument("--size", type=int, default=config.TARGET_SIZE)
    args = ap.parse_args()

    res = Path(args.results_dir)
    tab = res / "tables"
    tex = res / "latex"
    tex.mkdir(parents=True, exist_ok=True)

    # ESKI .tex DOSYALARINI SIL.
    # Tablo numaralari kosu sayisina gore degisir; onceki calistirmadan kalan
    # dosyalar silinmezse teze YANLIS/ESKI tablo \input edilebilir.
    stale = sorted(tex.glob("*.tex"))
    for f in stale:
        f.unlink()

    if not tab.is_dir():
        print(f"HATA: {tab} yok. Once step2 (ve varsa step3/step4) calistir.")
        return 1

    tags = sorted(p.stem.replace("summary_", "")
                  for p in tab.glob("summary_*.csv"))
    deconf_tags = sorted(p.stem.replace("deconf_summary_", "")
                         for p in tab.glob("deconf_summary_*.csv"))
    all_tags = sorted(set(tags) | set(deconf_tags))

    print("=" * 70)
    print("ADIM 5: Teze hazir rapor")
    print("=" * 70)
    if not all_tags:
        print("Hicbir sonuc bulunamadi.")
        return 1
    print(f"Bulunan kosular ({len(all_tags)}): {', '.join(all_tags)}")

    md = ["# Sonuclar -- birlesik rapor", "",
          "Bu dosya `step5_make_report.py` tarafindan mevcut CSV ciktilarindan",
          "otomatik uretildi. Tablolar `results/latex/` altinda LaTeX olarak da",
          "mevcuttur (`\\usepackage{booktabs}` gerekir).", "",
          f"Kosular: {', '.join('`' + t + '`' for t in all_tags)}", ""]
    findings = []
    tno = 0

    # ---------------- ANA TABLO: kosular arasi ozet -----------------------
    master = []
    for t in all_tags:
        sm = _read(tab / f"summary_{t}.csv")
        dc = _read(tab / f"deconf_summary_{t}.csv")
        lo = _read(tab / f"deconf_loso_{t}.csv")
        fm = _read(tab / f"fold_metrics_{t}.csv")
        row = {"kosu": t}

        if fm is not None and {"n_train", "n_test"} <= set(fm.columns):
            row["n"] = int(fm["n_train"].iloc[0] + fm["n_test"].iloc[0])

        def _pick(frame, method, col):
            if frame is None or "method" not in frame.columns:
                return np.nan
            s = frame.loc[frame["method"] == method, col]
            return float(s.iloc[0]) if len(s) and col in frame.columns else np.nan

        row["Tucker+MLP"] = _pick(sm, "Tucker+MLP", "balanced_accuracy")
        row["Voxel-MLP"] = _pick(sm, "Voxel-MLP", "balanced_accuracy")
        row["arinmis"] = _pick(dc, "Tucker(arinmis)+MLP", "balanced_accuracy")
        row["arinmis_p"] = _pick(dc, "Tucker(arinmis)+MLP", "p_vs_chance")
        row["demografi"] = _pick(dc, "Demo+LogReg", "balanced_accuracy")
        if np.isfinite(row["Tucker+MLP"]) and np.isfinite(row["demografi"]):
            row["goruntu-demografi"] = row["Tucker+MLP"] - row["demografi"]
        row["LOSO AUC"] = (float(lo["auc"].mean())
                           if lo is not None and "auc" in lo.columns and len(lo)
                           else np.nan)
        master.append(row)

    if master:
        mdf = pd.DataFrame(master)
        cols = [c for c in ["kosu", "n", "Tucker+MLP", "Voxel-MLP", "arinmis",
                            "arinmis_p", "demografi", "goruntu-demografi",
                            "LOSO AUC"] if c in mdf.columns]
        mdf = mdf[cols]
        tno += 1
        cap = ("Butun kosularin ozeti. Degerler dengeli dogruluktur; "
               "`arinmis_p` arindirilmis modelin sans seviyesine karsi "
               "tek yonlu duzeltilmis p-degeridir.")
        md += [f"## Tablo {tno}. {cap}", "", md_table(mdf), "",
               "*Sans seviyesi: ikili gorevlerde 0.500, 4 sinifta 0.250.*", ""]
        (tex / f"table{tno}_master.tex").write_text(
            to_latex(mdf, "Butun kosularin ozeti (dengeli dogruluk)",
                     "tab:master"), encoding="utf-8")

        for _, r in mdf.iterrows():
            if "goruntu-demografi" in r and np.isfinite(r["goruntu-demografi"]):
                sign = "gecti" if r["goruntu-demografi"] > 0 else "GECEMEDI"
                findings.append(
                    f"[{r['kosu']}] Goruntu, demografi baseline'ini {sign} "
                    f"({r['goruntu-demografi']:+.3f} dengeli dogruluk).")

    # ---------------- Tablo 1: ornek karakteristikleri -------------------
    for t in all_tags:
        st = _read(tab / f"sample_characteristics_{t}.csv")
        if st is None:
            continue
        tno += 1
        cap = f"Ornek karakteristikleri -- {t}"
        md += [f"## Tablo {tno}. {cap}", "", md_table(st, ".1f"), ""]
        (tex / f"table{tno}_sample_{t}.tex").write_text(
            to_latex(st, cap, f"tab:sample_{t}", "{:.1f}"), encoding="utf-8")

    # ---------------- Tablo: ana performans karsilastirmasi --------------
    METRICS = ["balanced_accuracy", "f1_macro", "auc", "accuracy", "kappa"]
    for t in tags:
        sm = _read(tab / f"summary_{t}.csv")
        if sm is None:
            continue
        keep = ["method"] + [m for m in METRICS if m in sm.columns]
        out = sm[keep].copy()
        for m in METRICS:
            sd = f"{m}_sd"
            if m in out.columns and sd in sm.columns:
                out[m] = [f"{v:.3f} ± {s:.3f}"
                          for v, s in zip(sm[m], sm[sd])]
        tno += 1
        cap = (f"Capraz dogrulama performansi (fold ortalamasi ± sd) -- {t}")
        md += [f"## Tablo {tno}. {cap}", "", md_table(out), ""]
        (tex / f"table{tno}_main_{t}.tex").write_text(
            to_latex(out, cap, f"tab:main_{t}"), encoding="utf-8")

        # bulgu: en iyi yontem ve sans karsilastirmasi
        if "balanced_accuracy" in sm.columns:
            best = sm.loc[sm["balanced_accuracy"].idxmax()]
            findings.append(
                f"[{t}] En yuksek dengeli dogruluk: **{best['method']}** "
                f"({best['balanced_accuracy']:.3f}).")

    # ---------------- Tablo: istatistiksel testler -----------------------
    for t in tags:
        st = _read(tab / f"statistical_tests_{t}.csv")
        if st is None:
            continue
        sub = st[st["metric"] == "f1_macro"] if "metric" in st.columns else st
        cols = [c for c in ["A", "B", "mean_diff", "cohens_d",
                            "p_corrected_ttest", "p_ttest_holm",
                            "p_wilcoxon", "p_wilcoxon_holm"] if c in sub.columns]
        tno += 1
        cap = f"Eslesmis karsilastirmalar, macro F1 -- {t}"
        md += [f"## Tablo {tno}. {cap}", "", md_table(sub[cols], ".4f"), "",
               "*p(t): Nadeau-Bengio duzeltilmis eslesmis t-testi; "
               "Holm = coklu karsilastirma duzeltmesi.*", ""]
        (tex / f"table{tno}_tests_{t}.tex").write_text(
            to_latex(sub[cols], cap, f"tab:tests_{t}", "{:.4f}"),
            encoding="utf-8")

        for _, r in sub.iterrows():
            p = r.get("p_ttest_holm", np.nan)
            if np.isfinite(p) and p < 0.05:
                findings.append(
                    f"[{t}] {r['A']} vs {r['B']}: fark "
                    f"{r['mean_diff']:+.3f} macro F1, p={p:.4f} (Holm) -- "
                    f"ANLAMLI.")

    # ---------------- Tablo: arindirma ------------------------------------
    for t in deconf_tags:
        sm = _read(tab / f"deconf_summary_{t}.csv")
        if sm is None:
            continue
        out = sm[["method"]].copy()
        out["dengeli dogruluk [%95 GA]"] = [
            f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
            for v, lo, hi in zip(sm["balanced_accuracy"],
                                 sm["ba_ci_low"], sm["ba_ci_high"])]
        if "f1_macro" in sm.columns:
            out["macro F1"] = [f"{v:.3f} ± {s:.3f}" for v, s in
                               zip(sm["f1_macro"], sm.get("f1_macro_sd", sm["f1_macro"] * 0))]
        if "auc" in sm.columns:
            out["AUC"] = sm["auc"].round(3)
        out["p (sans)"] = ["<0.001" if (np.isfinite(p) and p < 0.001)
                           else (f"{p:.4f}" if np.isfinite(p) else "--")
                           for p in sm["p_vs_chance"]]
        tno += 1
        cap = (f"Karistiricilardan arindirma etkisi -- {t}. "
               f"Guven araliklari fold bagimliligina gore duzeltilmistir.")
        md += [f"## Tablo {tno}. {cap}", "", md_table(out), ""]
        (tex / f"table{tno}_deconf_{t}.tex").write_text(
            to_latex(out, cap, f"tab:deconf_{t}"), encoding="utf-8")

        def _g(name, col="balanced_accuracy"):
            s = sm.loc[sm["method"] == name, col]
            return float(s.iloc[0]) if len(s) else np.nan

        raw, dec = _g("Tucker+MLP"), _g("Tucker(arinmis)+MLP")
        p_dec = _g("Tucker(arinmis)+MLP", "p_vs_chance")
        demo = _g("Demo+LogReg")
        if np.isfinite(dec):
            findings.append(
                f"[{t}] Arindirma sonrasi dengeli dogruluk {dec:.3f} "
                f"(ham {raw:.3f}, degisim {dec - raw:+.3f}); sansa karsi "
                f"p={p_dec:.4f} -> {_sig(p_dec)}.")
        if np.isfinite(demo) and np.isfinite(raw):
            findings.append(
                f"[{t}] Sadece demografi {demo:.3f} vs ham goruntu {raw:.3f} "
                f"(fark {raw - demo:+.3f}).")

    # ---------------- Tablo: LOSO -----------------------------------------
    for t in deconf_tags:
        lo = _read(tab / f"deconf_loso_{t}.csv")
        if lo is None:
            continue
        tno += 1
        cap = (f"Tarayici-disi genelleme (leave-one-scanner-out) -- {t}. "
               f"Birincil olcut AUC; dengeli dogruluk onsel kaymasina duyarlidir.")
        md += [f"## Tablo {tno}. {cap}", "", md_table(lo), ""]
        (tex / f"table{tno}_loso_{t}.tex").write_text(
            to_latex(lo, cap, f"tab:loso_{t}"), encoding="utf-8")
        if "auc" in lo.columns and len(lo):
            findings.append(
                f"[{t}] LOSO ortalama AUC {lo['auc'].mean():.3f} "
                f"(sans 0.500); dengeli dogruluk "
                f"{lo['balanced_accuracy'].mean():.3f}"
                + (f", onsel-duzeltilmis "
                   f"{lo['balanced_accuracy_prior_adj'].mean():.3f}"
                   if "balanced_accuracy_prior_adj" in lo.columns else "") + ".")

    # ---------------- Tablo: karistirici kontrolleri ----------------------
    for t in all_tags:
        gh = _read(tab / f"confound_ghost_{t}.csv")
        if gh is not None:
            tno += 1
            cap = f"Hayalet artefakti x tani caprazlamasi -- {t}"
            md += [f"## Tablo {tno}. {cap}", "", md_table(gh, ".1f"), ""]
            (tex / f"table{tno}_ghost_{t}.tex").write_text(
                to_latex(gh, cap, f"tab:ghost_{t}", "{:.1f}"), encoding="utf-8")

        bal = _read(tab / f"confound_balance_{t}.csv")
        if bal is not None:
            tno += 1
            cap = f"Gruplar arasi demografik denge testleri -- {t}"
            md += [f"## Tablo {tno}. {cap}", "", md_table(bal, ".4f"), ""]
            (tex / f"table{tno}_balance_{t}.tex").write_text(
                to_latex(bal, cap, f"tab:balance_{t}", "{:.4f}"), encoding="utf-8")
            for _, r in bal.iterrows():
                if np.isfinite(r.get("p", np.nan)) and r["p"] < 0.05:
                    findings.append(
                        f"[{t}] Gruplar **{r['degisken']}** bakimindan farkli "
                        f"({r['test']}, p={r['p']:.4f}) -- karistirici riski.")

        fc = _read(tab / f"confound_feature_content_{t}.csv")
        if fc is not None:
            tno += 1
            cap = (f"Tucker ozniteliklerinin kodladigi bilgi -- {t}. "
                   f"Yas icin CV $R^2$, digerleri icin dengeli dogruluk.")
            md += [f"## Tablo {tno}. {cap}", "", md_table(fc), ""]
            (tex / f"table{tno}_content_{t}.tex").write_text(
                to_latex(fc, cap, f"tab:content_{t}"), encoding="utf-8")
            dx = fc.loc[fc["hedef"] == "tani", "deger"]
            nuis = fc.loc[fc["hedef"].isin(["cinsiyet", "tarayici"]), "deger"]
            if len(dx) and len(nuis) and float(nuis.max()) > float(dx.iloc[0]):
                worst = fc.loc[fc["deger"].idxmax()]
                findings.append(
                    f"[{t}] Oznitelikler **{worst['hedef']}** bilgisini "
                    f"({worst['deger']:.3f}) taniyi ({float(dx.iloc[0]):.3f}) "
                    f"kodladigindan daha iyi kodluyor.")

        cmpd = _read(tab / f"confound_image_vs_demo_{t}.csv")
        if cmpd is not None and len(cmpd):
            best_demo = cmpd.groupby("demografi_yontemi")["F1_demografi"].mean().idxmax()
            sub = cmpd[cmpd["demografi_yontemi"] == best_demo]
            b = sub.loc[sub["fark"].idxmax()]
            findings.append(
                f"[{t}] En guclu demografi baseline ({best_demo}, "
                f"F1={b['F1_demografi']:.3f}) vs en iyi goruntu yontemi "
                f"({b['goruntu_yontemi']}, F1={b['F1_goruntu']:.3f}): "
                f"fark {b['fark']:+.3f}, p={b['p_corrected_ttest']:.4f} "
                f"-> {_sig(b['p_corrected_ttest'])}.")

    # ---------------- Sikistirma -----------------------------------------
    for t in tags:
        cp = _read(tab / f"compression_{t}.csv")
        if cp is None:
            continue
        tno += 1
        cap = f"Tucker rank vs sikistirma kalitesi -- {t}"
        md += [f"## Tablo {tno}. {cap}", "", md_table(cp, ".4f"), ""]
        (tex / f"table{tno}_compression_{t}.tex").write_text(
            to_latex(cp, cap, f"tab:compression_{t}", "{:.4f}"), encoding="utf-8")

        rs = _read(tab / f"rank_sweep_{t}.csv")
        if rs is not None and len(rs):
            b = rs.loc[rs["mean"].idxmax()]
            findings.append(
                f"[{t}] Rank taramasinda en iyi: rank {int(b['rank'])} "
                f"({int(b['n_features'])} oznitelik), macro F1 "
                f"{b['mean']:.3f} ± {b['std']:.3f}.")

    # ---------------- Bulgular -------------------------------------------
    md += ["---", "", "## Olgusal bulgu listesi", "",
           "Asagidaki maddeler CSV ciktilarindan dogrudan uretildi. "
           "Yorum eklenmemistir -- tezdeki tartisma bolumunu bunlarin "
           "uzerine sen yazacaksin.", ""]
    md += [f"{i + 1}. {f}" for i, f in enumerate(findings)] or ["(bulgu yok)"]
    md += ["", "---", "", "## Raporlama hatirlatmalari", "",
           "- Sinif dengesizligi nedeniyle birincil olcut **dengeli dogruluk** "
           "ve **macro F1** olmali; accuracy yaniltici "
           "(cogunluk sinifi tahmini yuksek accuracy verir).",
           "- Guven araliklari ve p-degerleri Nadeau-Bengio duzeltmelidir; "
           "tekrarlı CV'de fold'lar bagimsiz degildir.",
           "- LOSO'da birincil olcut AUC'dir (esikten bagimsiz, onsel "
           "kaymasindan etkilenmez).",
           "- Tarayici ile tani iliskili oldugu icin kovaryant regresyonu "
           "asiri duzeltme yapar; arindirilmis sonuclar **alt sinir**dir.",
           "- Tensor ayrisimi her fold'da yalnizca egitim verisiyle fit "
           "edilmistir (veri sizintisi yok).", ""]

    out_md = res / "THESIS_REPORT.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    if stale:
        print(f"\n({len(stale)} eski .tex dosyasi silindi)")
    print(f"\n{tno} tablo uretildi.")
    print(f"{len(findings)} olgusal bulgu cikarildi.")
    print(f"\nMarkdown : {out_md}")
    n_tex = len(list(tex.glob("*.tex")))
    print(f"LaTeX    : {tex}  ({n_tex} dosya)")
    if n_tex != tno:
        print(f"  UYARI: tablo sayisi ({tno}) ile .tex dosya sayisi "
              f"({n_tex}) uyusmuyor.")
    print("\nOnizleme -- bulgular:")
    for f in findings[:12]:
        print("  * " + f.replace("**", ""))
    if len(findings) > 12:
        print(f"  ... ve {len(findings) - 12} tane daha (raporda)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())