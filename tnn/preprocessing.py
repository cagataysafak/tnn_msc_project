"""
T1w MR hacimlerini karsilastirilabilir bir tensor gridine tasiyan on isleme.

ds000030'daki T1w goruntuleri denek uzayindadir (native space): her deneginin
kafa pozisyonu, kafa buyuklugu ve voksel boyutu farklidir. Voksel-bazli
karsilastirma yapabilmek icin en azindan kaba bir uzamsal normalizasyon
gerekir. Tam bir kayit (registration) icin ANTs/FSL gerekirdi; bunlar
Windows'ta kurulumu zahmetli ve CPU'da saatler surer.

Burada kayit yazilimi gerektirmeyen, tamamen numpy/scipy ile calisan
"bounding-box tabanli uzamsal normalizasyon" uyguluyoruz:

  1. Kanonik RAS yonelimine cevir (nibabel)         -> eksen sirasi ayni
  2. Otsu esigi ile kafa maskesi + en buyuk bilesen  -> arka plan atilir
  3. Maskenin sinir kutusuna (bounding box) kirp     -> konum normalizasyonu
  4. mm cinsinden kupe tamamla (aspect korunur)      -> anatomik oran korunur
  5. TARGET_SIZE^3 gridine yeniden orneklenir        -> kafa boyutu normalize
  6. Maske ici %99 persentil ile yogunluk olcekleme  -> tarayici farki azalir

Sonuc: her denek icin (S, S, S) float32 hacim; hepsi ayni gridde.

Bu, tam bir MNI kaydinin yerini tutmaz -- rapora sinirlilik olarak yazilmali.
Ama tensor ayrisimi + siniflandirma zincirini calistirmak icin yeterli ve
tamamen tekrar uretilebilir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi


# --------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Klasik Otsu esigi (skimage bagimliligi olmadan)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    vmin, vmax = float(values.min()), float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return vmin
    hist, edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centers) / w0
        m1 = (np.cumsum((hist * centers)[::-1])[::-1] - hist * centers) / w1
    var_between = w0 * w1 * (m0 - m1) ** 2
    var_between = np.nan_to_num(var_between, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centers[int(np.argmax(var_between))])


def head_mask(vol: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Otsu + en buyuk baglantili bilesen + delik doldurma -> kafa maskesi."""
    thr = otsu_threshold(vol) * scale
    mask = vol > thr
    if mask.sum() == 0:                       # guvenlik agi
        mask = vol > np.percentile(vol, 60)

    # kucuk gurultuyu temizle, sonra en buyuk bileseni al
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3)))
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == int(np.argmax(sizes))
    elif n == 0:
        mask = vol > np.percentile(vol, 60)

    mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def bounding_box(mask: np.ndarray, pad: int = 2):
    """Maskenin sinir kutusu (her eksen icin slice), kenarda `pad` voksel payla."""
    idx = np.where(mask)
    slices = []
    for ax in range(mask.ndim):
        lo = max(int(idx[ax].min()) - pad, 0)
        hi = min(int(idx[ax].max()) + pad + 1, mask.shape[ax])
        slices.append(slice(lo, hi))
    return tuple(slices)


def pad_to_cube_mm(vol: np.ndarray, zooms) -> np.ndarray:
    """
    Hacmi, fiziksel (mm) boyutlari esit olacak sekilde sifirla doldur.
    Boylece sonraki yeniden orneklemede anatomik en-boy orani bozulmaz.
    """
    zooms = np.asarray(zooms, dtype=np.float64)
    extent_mm = np.asarray(vol.shape, dtype=np.float64) * zooms
    target_mm = float(extent_mm.max())

    pads = []
    for ax in range(3):
        need_vox = int(np.ceil((target_mm - extent_mm[ax]) / zooms[ax]))
        need_vox = max(need_vox, 0)
        before = need_vox // 2
        pads.append((before, need_vox - before))
    return np.pad(vol, pads, mode="constant", constant_values=0.0)


def resize_volume(vol: np.ndarray, out_size: int, order: int = 1) -> np.ndarray:
    """Hacmi (out_size, out_size, out_size) boyutuna yeniden ornekle."""
    factors = [out_size / s for s in vol.shape]
    out = ndi.zoom(vol, factors, order=order, mode="nearest", prefilter=(order > 1))
    # zoom yuvarlama nedeniyle +-1 voksel sapabilir; kirp/doldur
    fixed = np.zeros((out_size,) * 3, dtype=np.float32)
    sl = tuple(slice(0, min(out.shape[a], out_size)) for a in range(3))
    fixed[sl] = out[sl]
    return fixed


# --------------------------------------------------------------------------
def preprocess_t1w(path, out_size: int = 64,
                   mask_scale: float = 1.0,
                   intensity_percentile: float = 99.0):
    """
    Tek bir T1w NIfTI dosyasini isle.

    Returns
    -------
    vol : (out_size, out_size, out_size) float32
    info : dict  -- QC metrikleri
    """
    import nibabel as nib

    path = Path(path)
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)          # RAS+ yonelimi
    data = np.asarray(img.dataobj, dtype=np.float32)

    if data.ndim == 4:                           # bazi dosyalar tek hacimli 4B olabilir
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"3B hacim bekleniyordu, bulunan sekil: {data.shape}")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)              # negatif degerleri kirp

    zooms = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
    zooms = np.where(zooms > 0, zooms, 1.0)

    mask = head_mask(data, scale=mask_scale)
    frac = float(mask.mean())

    bb = bounding_box(mask, pad=2)
    vol = data[bb] * mask[bb]                    # maske disini sifirla (kaba beyin ayirma)

    vol = pad_to_cube_mm(vol, zooms)
    vol = resize_volume(vol, out_size, order=1)

    # yogunluk normalizasyonu: maske ici persentil ile olcekle
    nz = vol[vol > 0]
    scale = float(np.percentile(nz, intensity_percentile)) if nz.size else 1.0
    if scale <= 0:
        scale = 1.0
    vol = np.clip(vol / scale, 0.0, 2.0).astype(np.float32)

    info = dict(
        source=path.name,
        orig_shape="x".join(str(s) for s in data.shape),
        voxel_mm="x".join(f"{z:.2f}" for z in zooms),
        mask_fraction=round(frac, 4),
        bbox_mm="x".join(
            f"{(bb[a].stop - bb[a].start) * zooms[a]:.0f}" for a in range(3)
        ),
        intensity_scale=round(scale, 3),
        nonzero_fraction=round(float((vol > 0).mean()), 4),
    )
    return vol, info
