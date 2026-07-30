"""
Tensor cebiri + MPCA (Multilinear PCA = ornek modu sikistirilmamis Tucker).

Neden hazir bir kutuphane degil de kendi implementasyonumuz?
  * tensorly'nin `partial_tucker` API'si surumden surume degisti; burada
    surum bagimliligi olmadan calisan, denetlenebilir bir kod istiyoruz.
  * Egitim/test ayrimini dogru kurmak icin `fit` (sadece train) ve
    `transform` (train + test) adimlarinin ayri olmasi sart. Asagidaki
    sinif tam olarak bunu saglar.

Matematiksel ozet
-----------------
Veri: X in R^{N x d1 x d2 x d3}   (N denek, her biri 3-yollu MR hacmi)

Tucker ayrisimi butun modlari sikistirir. Siniflandirma icin ornek modunu
(N) sikistirmak istemeyiz -- cunku yeni bir denek geldiginde onu ayni uzaya
projekte edebilmemiz gerekir. Bu yuzden "kismi Tucker" / MPCA kullaniyoruz:

    X ~= C  x_1 U1  x_2 U2  x_3 U3            U_k in R^{d_k x r_k},  U_k^T U_k = I

Her denek icin ozellik vektoru cekirdek tensordur:

    C_n = (X_n - Xbar)  x_1 U1^T  x_2 U2^T  x_3 U3^T    in R^{r1 x r2 x r3}

U_k'lar HOOI (higher-order orthogonal iteration) ile bulunur: mod-k
sacilim matrisinin (scatter matrix) en buyuk r_k ozvektoru, diger modlar
projekte edilmis haldeyken, yakinsayana kadar donusumlu olarak guncellenir.
"""
from __future__ import annotations

import numpy as np


# ==========================================================================
# Temel tensor islemleri
# ==========================================================================
def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mod-`mode` matrislestirme (unfolding). Sonuc: (shape[mode], -1)."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """n-mod carpim: tensor x_mode matrix.  matrix shape = (yeni, eski)."""
    if matrix.shape[1] != tensor.shape[mode]:
        raise ValueError(
            f"mode_dot boyut uyusmazligi: matrix {matrix.shape} vs "
            f"tensor mode {mode} boyutu {tensor.shape[mode]}"
        )
    moved = np.moveaxis(tensor, mode, 0)
    new_shape = (matrix.shape[0],) + moved.shape[1:]
    out = matrix @ np.reshape(moved, (moved.shape[0], -1))
    return np.moveaxis(np.reshape(out, new_shape), 0, mode)


def multi_mode_dot(tensor, matrices, modes, transpose: bool = False) -> np.ndarray:
    """Birden fazla modda ardisik n-mod carpim."""
    out = tensor
    for mat, mode in zip(matrices, modes):
        out = mode_dot(out, mat.T if transpose else mat, mode)
    return out


def frob(tensor: np.ndarray) -> float:
    """Frobenius normu (float64 birikimli, tasma/hassasiyet icin)."""
    return float(np.sqrt(np.sum(np.asarray(tensor, dtype=np.float64) ** 2)))


# ==========================================================================
# MPCA / kismi Tucker
# ==========================================================================
class MPCA:
    """
    Multilinear PCA (ornek modu haric butun modlarda Tucker sikistirmasi).

    Parameters
    ----------
    ranks : tuple[int, ...]
        Uzamsal modlar icin cekirdek boyutlari, or. (10, 10, 10).
    n_iter : int
        HOOI iterasyon sayisi. 0 -> yalnizca HOSVD (tek gecis, iteratif degil).
    center : bool
        True ise egitim ortalamasi hacmi cikartilir (onerilir).
    tol : float
        Aciklanan enerjideki goreli degisim bu esigin altina inince durur.
    verbose : bool
    """

    def __init__(self, ranks=(10, 10, 10), n_iter: int = 5,
                 center: bool = True, tol: float = 1e-6, verbose: bool = False):
        self.ranks = tuple(int(r) for r in ranks)
        self.n_iter = int(n_iter)
        self.center = bool(center)
        self.tol = float(tol)
        self.verbose = bool(verbose)

        self.factors_ = None        # list[np.ndarray], her biri (d_k, r_k)
        self.mean_ = None           # (d1, d2, d3)
        self.eigenvalues_ = None    # list[np.ndarray] tam ozdeger spektrumu
        self.explained_ratio_ = None
        self.total_energy_ = None

    # ------------------------------------------------------------------
    @staticmethod
    def _scatter(tensor: np.ndarray, mode: int) -> np.ndarray:
        """Mod-`mode` sacilim matrisi: sum_n X_n(mode) X_n(mode)^T."""
        A = unfold(tensor, mode)
        return np.asarray(A, dtype=np.float64) @ np.asarray(A, dtype=np.float64).T

    @staticmethod
    def _top_eigvecs(S: np.ndarray, r: int):
        """Simetrik S'nin en buyuk r ozvektoru + tam ozdeger spektrumu (azalan)."""
        w, V = np.linalg.eigh(S)          # artan sirali
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]
        return np.ascontiguousarray(V[:, :r]), w

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "MPCA":
        """
        X : (N, d1, d2, d3) float array. SADECE egitim verisi verilmelidir.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4:
            raise ValueError(f"X 4 boyutlu olmali (N,d1,d2,d3), verilen: {X.shape}")
        n_modes = X.ndim - 1
        if len(self.ranks) != n_modes:
            raise ValueError(f"{n_modes} rank bekleniyor, {len(self.ranks)} verildi")
        dims = X.shape[1:]
        for r, d in zip(self.ranks, dims):
            if not (1 <= r <= d):
                raise ValueError(f"gecersiz rank {r}, boyut {d}")

        self.mean_ = X.mean(axis=0) if self.center else np.zeros(dims, dtype=np.float32)
        Xc = X - self.mean_
        self.total_energy_ = float(np.sum(np.asarray(Xc, dtype=np.float64) ** 2))

        modes = list(range(1, X.ndim))

        # --- HOSVD baslangici -------------------------------------------
        factors, eigvals = [], []
        for k, r in zip(modes, self.ranks):
            U, w = self._top_eigvecs(self._scatter(Xc, k), r)
            factors.append(U)
            eigvals.append(w)

        # --- HOOI iterasyonlari -----------------------------------------
        prev = -np.inf
        for it in range(self.n_iter):
            for j, (k, r) in enumerate(zip(modes, self.ranks)):
                others = [factors[i] for i in range(len(modes)) if i != j]
                other_modes = [modes[i] for i in range(len(modes)) if i != j]
                proj = multi_mode_dot(Xc, others, other_modes, transpose=True)
                U, w = self._top_eigvecs(self._scatter(proj, k), r)
                factors[j] = U
                eigvals[j] = w

            core = multi_mode_dot(Xc, factors, modes, transpose=True)
            energy = float(np.sum(np.asarray(core, dtype=np.float64) ** 2))
            ratio = energy / max(self.total_energy_, 1e-12)
            if self.verbose:
                print(f"    [MPCA] iter {it + 1}: aciklanan enerji = {ratio:.4f}")
            if prev > -np.inf and abs(ratio - prev) < self.tol:
                break
            prev = ratio

        self.factors_ = factors
        self.eigenvalues_ = eigvals

        core = multi_mode_dot(Xc, factors, modes, transpose=True)
        self.explained_ratio_ = float(
            np.sum(np.asarray(core, dtype=np.float64) ** 2)
            / max(self.total_energy_, 1e-12)
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray, chunk: int = 64) -> np.ndarray:
        """(N, d1, d2, d3) -> (N, r1, r2, r3) cekirdek tensorler."""
        if self.factors_ is None:
            raise RuntimeError("Once fit() cagirin.")
        X = np.asarray(X, dtype=np.float32)
        modes = list(range(1, X.ndim))
        outs = []
        for s in range(0, X.shape[0], chunk):
            block = X[s:s + chunk] - self.mean_
            outs.append(multi_mode_dot(block, self.factors_, modes, transpose=True))
        return np.concatenate(outs, axis=0).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, cores: np.ndarray) -> np.ndarray:
        """Cekirdeklerden hacimleri geri kur (gorsellestirme / hata analizi)."""
        modes = list(range(1, cores.ndim))
        rec = multi_mode_dot(cores, self.factors_, modes, transpose=False)
        return (rec + self.mean_).astype(np.float32)

    # ------------------------------------------------------------------
    def features(self, X: np.ndarray) -> np.ndarray:
        """Duzlestirilmis ozellik matrisi: (N, r1*r2*r3)."""
        C = self.transform(X)
        return C.reshape(C.shape[0], -1)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Goreli Frobenius yeniden kurma hatasi ||X - X_hat||_F / ||X||_F."""
        Xr = self.inverse_transform(self.transform(X))
        return frob(X - Xr) / max(frob(X), 1e-12)

    @property
    def n_features_(self) -> int:
        return int(np.prod(self.ranks))

    def eigen_volume(self, idx=(0, 0, 0)) -> np.ndarray:
        """
        Tek bir cekirdek girdisine karsilik gelen "oz-hacim":
        u1_{i} o u2_{j} o u3_{k}  (dis carpim). Gorsellestirme icin.
        """
        u1 = self.factors_[0][:, idx[0]]
        u2 = self.factors_[1][:, idx[1]]
        u3 = self.factors_[2][:, idx[2]]
        return np.einsum("i,j,k->ijk", u1, u2, u3)


# ==========================================================================
# Basit dogrulama testleri (python -m tnn.tensor_utils ile calistirilir)
# ==========================================================================
def _self_test(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    # 1) unfold / mode_dot tutarliligi
    T = rng.standard_normal((4, 5, 6))
    M = rng.standard_normal((3, 5))
    lhs = unfold(mode_dot(T, M, 1), 1)
    rhs = M @ unfold(T, 1)
    assert np.allclose(lhs, rhs), "mode_dot/unfold tutarsiz"

    # 2) ortonormal faktorlerle donusum tersinir olmali
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    T2 = mode_dot(T, Q.T, 2)
    T3 = mode_dot(T2, Q, 2)
    assert np.allclose(T, T3, atol=1e-10), "n-mod carpim tersinirligi bozuk"

    # 3) tam rank MPCA -> hata ~ 0
    X = rng.standard_normal((12, 5, 6, 7)).astype(np.float32)
    m = MPCA(ranks=(5, 6, 7), n_iter=2).fit(X)
    err = m.reconstruction_error(X)
    assert err < 1e-5, f"tam rankta hata sifir olmali, bulunan {err}"

    # 4) dusuk rank -> hata artmali ama makul kalmali, aciklanan enerji [0,1]
    m2 = MPCA(ranks=(2, 3, 3), n_iter=3).fit(X)
    assert 0.0 <= m2.explained_ratio_ <= 1.0
    assert m2.reconstruction_error(X) > err

    # 5) dusuk-ranki verideki HOOI, HOSVD'den daha iyi olmali
    core = rng.standard_normal((20, 2, 2, 2))
    U = [np.linalg.qr(rng.standard_normal((d, 2)))[0] for d in (8, 9, 10)]
    Xlow = multi_mode_dot(core, U, [1, 2, 3]).astype(np.float32)
    Xlow += 0.01 * rng.standard_normal(Xlow.shape).astype(np.float32)
    e_hosvd = MPCA(ranks=(2, 2, 2), n_iter=0).fit(Xlow).reconstruction_error(Xlow)
    e_hooi = MPCA(ranks=(2, 2, 2), n_iter=8).fit(Xlow).reconstruction_error(Xlow)
    assert e_hooi <= e_hosvd + 1e-6, (e_hooi, e_hosvd)

    # 6) transform tutarli (chunk'li ve chunk'siz ayni)
    a = m2.transform(X, chunk=3)
    b = m2.transform(X, chunk=1000)
    assert np.allclose(a, b, atol=1e-5)

    print("tensor_utils self-test: OK")
    print(f"  tam rank hata      = {err:.2e}")
    print(f"  (2,3,3) hata       = {m2.reconstruction_error(X):.4f}")
    print(f"  HOSVD vs HOOI hata = {e_hosvd:.5f} vs {e_hooi:.5f}")


if __name__ == "__main__":
    _self_test()
