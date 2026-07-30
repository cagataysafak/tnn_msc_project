"""
Gorsellestirme yardimcilari. Hepsi matplotlib; ekstra bagimlilik yok.
Tum fonksiyonlar dosyaya kaydedip yolu dondurur.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _boxplot(ax, data, labels, **kw):
    """matplotlib >=3.9 'tick_labels', <3.9 'labels' -- ikisini de destekle."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def _mid_slices(vol: np.ndarray):
    s = vol.shape
    return (vol[s[0] // 2, :, :], vol[:, s[1] // 2, :], vol[:, :, s[2] // 2])


def montage(vol: np.ndarray, n: int = 6, axis: int = 2) -> np.ndarray:
    """Bir hacimden n dilimlik yatay serit olustur."""
    idx = np.linspace(0.2, 0.8, n) * vol.shape[axis]
    sl = [np.take(vol, int(i), axis=axis) for i in idx]
    return np.concatenate([np.rot90(s) for s in sl], axis=1)


# ==========================================================================
# ADIM 1 figurleri
# ==========================================================================
def plot_demographics(meta, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    counts = meta["diagnosis"].value_counts()
    axes[0].bar(counts.index, counts.values,
                color=PALETTE[: len(counts)], edgecolor="black", lw=0.6)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=9)
    axes[0].set_title("Tani dagilimi (islenen denekler)")
    axes[0].set_ylabel("denek sayisi")
    axes[0].tick_params(axis="x", rotation=20)

    groups, data = [], []
    for g in counts.index:
        a = meta.loc[meta["diagnosis"] == g, "age"].dropna().values
        if len(a):
            groups.append(g)
            data.append(a)
    if data:
        bp = _boxplot(axes[1], data, groups, patch_artist=True)
        for patch, c in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    axes[1].set_title("Yasa gore dagilim")
    axes[1].set_ylabel("yas")
    axes[1].tick_params(axis="x", rotation=20)

    if "gender" in meta:
        ct = meta.groupby(["diagnosis", "gender"]).size().unstack(fill_value=0)
        bottom = np.zeros(len(ct))
        for j, col in enumerate(ct.columns):
            axes[2].bar(ct.index, ct[col].values, bottom=bottom,
                        label=str(col), color=PALETTE[j % len(PALETTE)],
                        edgecolor="black", lw=0.5)
            bottom += ct[col].values
        axes[2].legend(fontsize=8)
    axes[2].set_title("Cinsiyet dagilimi")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Sekil 1 -- Ornek karakteristikleri", y=1.02, fontsize=12)
    return _save(fig, path)


def plot_preprocessing_qc(X, meta, path, n_subjects: int = 4):
    n = min(n_subjects, len(X))
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n))
    axes = np.atleast_1d(axes)
    for i in range(n):
        axes[i].imshow(montage(X[i], n=8), cmap="gray", vmin=0, vmax=1.2)
        axes[i].set_ylabel(
            f"{meta.loc[i, 'participant_id']}\n{meta.loc[i, 'diagnosis']}",
            fontsize=8, rotation=0, ha="right", va="center")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.suptitle("Sekil 2 -- On isleme sonrasi hacimler (aksiyel dilimler)",
                 fontsize=12)
    return _save(fig, path)


def plot_class_means(X, labels, class_names, path):
    k = len(class_names)
    fig, axes = plt.subplots(2, k, figsize=(3.6 * k, 6))
    axes = np.atleast_2d(axes)
    means = [X[labels == i].mean(axis=0) for i in range(k)]
    ref = means[0]
    for i in range(k):
        axes[0, i].imshow(montage(means[i], n=5), cmap="gray", vmin=0, vmax=1.1)
        axes[0, i].set_title(f"{class_names[i]}  (n={int((labels == i).sum())})",
                             fontsize=10)
        axes[0, i].axis("off")

        d = montage(means[i] - ref, n=5)
        lim = float(np.abs(d).max()) or 1.0
        im = axes[1, i].imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[1, i].set_title(f"{class_names[i]} - {class_names[0]}", fontsize=9)
        axes[1, i].axis("off")
        fig.colorbar(im, ax=axes[1, i], fraction=0.03)
    fig.suptitle("Sekil 3 -- Sinif ortalama hacimleri ve farklari", fontsize=12)
    return _save(fig, path)


# ==========================================================================
# Tensor ayrisimi figurleri
# ==========================================================================
def plot_mpca_spectra(mpca, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["mod-1 (x)", "mod-2 (y)", "mod-3 (z)"]
    for i, w in enumerate(mpca.eigenvalues_):
        w = np.clip(w, 0, None)
        cum = np.cumsum(w) / max(w.sum(), 1e-12)
        axes[0].semilogy(np.arange(1, len(w) + 1), w + 1e-12,
                         color=PALETTE[i], label=names[i])
        axes[1].plot(np.arange(1, len(cum) + 1), cum, color=PALETTE[i],
                     label=names[i])
        axes[1].axvline(mpca.ranks[i], color=PALETTE[i], ls=":", alpha=0.7)
    axes[0].set_title("Mod bazli ozdeger spektrumu")
    axes[0].set_xlabel("bilesen"); axes[0].set_ylabel("ozdeger (log)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_title("Kumulatif aciklanan varyans (mod bazli)")
    axes[1].set_xlabel("bilesen sayisi"); axes[1].set_ylabel("oran")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Sekil 4 -- MPCA / kismi Tucker spektrumu "
        f"(rank={mpca.ranks}, aciklanan enerji={mpca.explained_ratio_:.3f})",
        fontsize=12)
    return _save(fig, path)


def plot_rank_vs_error(ranks, errors, energies, path):
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(ranks, errors, "o-", color=PALETTE[3], label="yeniden kurma hatasi")
    ax1.set_xlabel("her modda rank r")
    ax1.set_ylabel("goreli Frobenius hatasi", color=PALETTE[3])
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ranks, energies, "s--", color=PALETTE[0], label="aciklanan enerji")
    ax2.set_ylabel("aciklanan enerji orani", color=PALETTE[0])
    for r, e in zip(ranks, errors):
        ax1.annotate(f"{r**3}", (r, e), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, alpha=0.7)
    ax1.set_title("Sekil 5 -- Rank vs sikistirma kalitesi\n"
                  "(etiketler: cekirdekteki ozellik sayisi r^3)")
    return _save(fig, path)


def plot_reconstructions(X, mpca_by_rank, sample_idx, path):
    ranks = sorted(mpca_by_rank.keys())
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(3.0 * (len(ranks) + 1), 3.4))
    axes[0].imshow(np.rot90(X[sample_idx][:, :, X.shape[3] // 2]),
                   cmap="gray", vmin=0, vmax=1.2)
    axes[0].set_title("orijinal", fontsize=10); axes[0].axis("off")
    for j, r in enumerate(ranks):
        m = mpca_by_rank[r]
        rec = m.inverse_transform(m.transform(X[sample_idx:sample_idx + 1]))[0]
        axes[j + 1].imshow(np.rot90(rec[:, :, rec.shape[2] // 2]),
                           cmap="gray", vmin=0, vmax=1.2)
        axes[j + 1].set_title(f"rank {r}  ({r**3} oz.)", fontsize=10)
        axes[j + 1].axis("off")
    fig.suptitle("Sekil 6 -- Farkli Tucker ranklarinda yeniden kurma", fontsize=12)
    return _save(fig, path)


def plot_eigen_volumes(mpca, path, n: int = 6):
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8))
    for j in range(n):
        ev = mpca.eigen_volume((j % mpca.ranks[0], j % mpca.ranks[1],
                                j % mpca.ranks[2]))
        sl = np.rot90(ev[:, :, ev.shape[2] // 2])
        lim = float(np.abs(sl).max()) or 1.0
        axes[j].imshow(sl, cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[j].set_title(f"u1_{j} o u2_{j} o u3_{j}", fontsize=8)
        axes[j].axis("off")
    fig.suptitle("Sekil 7 -- Tucker faktorlerinin olusturdugu oz-hacimler "
                 "(rank-1 bilesenler)", fontsize=12)
    return _save(fig, path)


def plot_embedding(F, labels, class_names, path, title_no: int = 8):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    Fs = StandardScaler().fit_transform(F)
    p2 = PCA(n_components=2, random_state=0).fit_transform(Fs)
    perp = max(5.0, min(30.0, (len(Fs) - 1) / 3.0))
    try:
        t2 = TSNE(n_components=2, random_state=0, init="pca",
                  perplexity=perp).fit_transform(Fs)
    except Exception:
        t2 = p2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, name in enumerate(class_names):
        m = labels == i
        axes[0].scatter(p2[m, 0], p2[m, 1], s=22, alpha=0.8,
                        color=PALETTE[i % len(PALETTE)], label=name)
        axes[1].scatter(t2[m, 0], t2[m, 1], s=22, alpha=0.8,
                        color=PALETTE[i % len(PALETTE)], label=name)
    axes[0].set_title("PCA (Tucker cekirdek ozellikleri)")
    axes[1].set_title("t-SNE (Tucker cekirdek ozellikleri)")
    for a in axes:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(f"Sekil {title_no} -- Ozellik uzayinin 2B gorunumu", fontsize=12)
    return _save(fig, path)


# ==========================================================================
# Sonuc figurleri
# ==========================================================================
def plot_training_curve(history, path, title_no: int = 9):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if history.get("train_loss"):
        ax.plot(history["train_loss"], label="egitim kaybi", color=PALETTE[0])
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="dogrulama kaybi", color=PALETTE[3])
    ax.set_xlabel("epok"); ax.set_ylabel("cross-entropy")
    ax.set_title(f"Sekil {title_no} -- MLP egitim egrisi (ornek fold)")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_confusion_matrices(cms, method_names, class_names, path, title_no: int = 10):
    k = len(cms)
    fig, axes = plt.subplots(1, k, figsize=(3.7 * k, 3.6))
    axes = np.atleast_1d(axes)
    for i, (cm, name) in enumerate(zip(cms, method_names)):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
        im = axes[i].imshow(norm, cmap="Blues", vmin=0, vmax=1)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xticks(range(len(class_names)))
        axes[i].set_yticks(range(len(class_names)))
        axes[i].set_xticklabels(class_names, rotation=45, fontsize=8, ha="right")
        axes[i].set_yticklabels(class_names, fontsize=8)
        for a in range(len(class_names)):
            for b in range(len(class_names)):
                axes[i].text(b, a, f"{norm[a, b]:.2f}", ha="center", va="center",
                             fontsize=8,
                             color="white" if norm[a, b] > 0.5 else "black")
        if i == 0:
            axes[i].set_ylabel("gercek")
        axes[i].set_xlabel("tahmin")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
    fig.suptitle(f"Sekil {title_no} -- Karisiklik matrisleri (satira gore normalize)",
                 fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_boxplots(df, metrics, path, title_no: int = 11):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)
    methods = list(df["method"].unique())
    for j, met in enumerate(metrics):
        data = [df.loc[df["method"] == m, met].values for m in methods]
        bp = _boxplot(axes[j], data, methods, patch_artist=True)
        for patch, c in zip(bp["boxes"], PALETTE * 3):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for i, d in enumerate(data):
            axes[j].scatter(np.full(len(d), i + 1) + np.random.uniform(-.06, .06, len(d)),
                            d, s=12, color="black", alpha=0.5, zorder=3)
        axes[j].set_title(met)
        axes[j].tick_params(axis="x", rotation=30)
        axes[j].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Sekil {title_no} -- Fold bazli performans dagilimi",
                 fontsize=12)
    return _save(fig, path)


def plot_roc(fpr_grid, tprs_by_method, aucs_by_method, path, title_no: int = 12):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for i, (name, tprs) in enumerate(tprs_by_method.items()):
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        auc_m = np.mean(aucs_by_method[name])
        auc_s = np.std(aucs_by_method[name])
        ax.plot(fpr_grid, mean_tpr, color=PALETTE[i % len(PALETTE)],
                label=f"{name}  AUC={auc_m:.3f}±{auc_s:.3f}")
        ax.fill_between(fpr_grid, np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=PALETTE[i % len(PALETTE)], alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("yanlis pozitif orani"); ax.set_ylabel("dogru pozitif orani")
    ax.set_title(f"Sekil {title_no} -- ROC egrileri (fold ortalamasi)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_rank_sweep(results, path, title_no: int = 13):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ranks = [r["rank"] for r in results]
    mean = [r["mean"] for r in results]
    std = [r["std"] for r in results]
    ax.errorbar(ranks, mean, yerr=std, fmt="o-", capsize=4, color=PALETTE[0])
    ax.set_xlabel("her modda rank r  (ozellik sayisi = r^3)")
    ax.set_ylabel("macro F1 (CV ortalamasi ± sd)")
    ax.set_title(f"Sekil {title_no} -- Tucker rank taramasi")
    ax.grid(alpha=0.3)
    return _save(fig, path)
