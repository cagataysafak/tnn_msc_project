"""
Merkezi konfigurasyon.

Butun scriptler bu dosyadaki varsayilanlari kullanir; komut satirindan
override edilebilir. Yol ayarlarini burada bir kere degistirmen yeterli.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Yollar
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Indirdigin BIDS klasoru (icinde participants.tsv ve sub-XXXXX/ klasorleri olmali)
BIDS_DIR = Path(os.environ.get("TNN_BIDS_DIR", PROJECT_ROOT / "data" / "ds000030"))

# Ara ciktilar (tensor dataset, npy dosyalari)
DERIV_DIR = Path(os.environ.get("TNN_DERIV_DIR", PROJECT_ROOT / "derivatives"))

# Sonuclar (figurler, tablolar)
RESULTS_DIR = Path(os.environ.get("TNN_RESULTS_DIR", PROJECT_ROOT / "results"))

FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"

for _d in (DERIV_DIR, RESULTS_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# On isleme
# --------------------------------------------------------------------------
# Her denegin T1w hacmi bu kup boyutuna getirilir -> (N, S, S, S) 4-yollu tensor
TARGET_SIZE = 64

# Otsu esiginin carpani (kafa maskesi icin). 1.0 = duz Otsu.
MASK_THRESHOLD_SCALE = 1.0

# Yogunluk normalizasyonu icin ust persentil
INTENSITY_PERCENTILE = 99.0

# --------------------------------------------------------------------------
# Siniflandirma gorevleri
# --------------------------------------------------------------------------
# raw participants.tsv "diagnosis" degerleri: CONTROL, SCHZ, BIPOLAR, ADHD
TASKS = {
    # 4 sinif, tam problem
    "4class": {
        "CONTROL": "CONTROL",
        "SCHZ": "SCHZ",
        "BIPOLAR": "BIPOLAR",
        "ADHD": "ADHD",
    },
    # ikili, literaturde en cok calisilan kontrast
    "schz_vs_control": {
        "CONTROL": "CONTROL",
        "SCHZ": "SCHZ",
    },
    # ikili, dengeli-ish: hasta vs saglikli
    "patient_vs_control": {
        "CONTROL": "CONTROL",
        "SCHZ": "PATIENT",
        "BIPOLAR": "PATIENT",
        "ADHD": "PATIENT",
    },
}

DEFAULT_TASK = "schz_vs_control"

# --------------------------------------------------------------------------
# Tensor ayristirma
# --------------------------------------------------------------------------
# Tucker/MPCA cekirdek boyutlari (3 uzamsal mod icin)
DEFAULT_RANKS = (8, 8, 8)
MPCA_N_ITER = 5          # HOOI iterasyon sayisi (0 -> sadece HOSVD)
RANK_SWEEP = [2, 4, 6, 8, 10, 12]

# --------------------------------------------------------------------------
# Sinir agi
# --------------------------------------------------------------------------
MLP_PARAMS = dict(
    hidden=(128, 32),
    dropout=0.4,
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=32,
    max_epochs=300,
    patience=30,
    val_fraction=0.15,
)

# --------------------------------------------------------------------------
# Degerlendirme
# --------------------------------------------------------------------------
N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42

# torch icin kullanilacak CPU thread sayisi (None -> otomatik)
N_THREADS = None
