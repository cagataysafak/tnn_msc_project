"""
BIDS klasorunu tarama + participants.tsv'den etiket cikarma.

ds000030 participants.tsv sutunlari (surum 1.0.0):
    participant_id, diagnosis, age, gender, ScannerSerialNumber,
    ghost_NoGhost, T1w, bart, bht, pamenc, pamret, rest, scap,
    stopsignal, taskswitch
diagnosis degerleri: CONTROL, SCHZ, BIPOLAR, ADHD
ghost_NoGhost: 'No_ghost' / 'ghost'  (T1w'de hayalet artefakti olan denekler)

Kod, sutun isimlerini buyuk/kucuk harf duyarsiz arar; sutun yoksa
sessizce atlar. Boylece dataset surumu degisse bile kirilmaz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_LABELS = {
    "CONTROL": "CONTROL", "CONTROLS": "CONTROL", "HC": "CONTROL",
    "HEALTHY": "CONTROL", "NC": "CONTROL",
    "SCHZ": "SCHZ", "SCZ": "SCHZ", "SCHIZ": "SCHZ",
    "SCHIZOPHRENIA": "SCHZ", "SCHZ.": "SCHZ",
    "BIPOLAR": "BIPOLAR", "BIPOLAR_DISORDER": "BIPOLAR", "BD": "BIPOLAR",
    "ADHD": "ADHD",
}


def _find_col(df: pd.DataFrame, *candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_participants(bids_dir) -> pd.DataFrame:
    """participants.tsv'yi oku, etiketleri normalize et."""
    bids_dir = Path(bids_dir)
    path = bids_dir / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"participants.tsv bulunamadi: {path}\n"
            f"BIDS_DIR ayarini kontrol et (su an: {bids_dir})"
        )

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    pid = _find_col(df, "participant_id", "participant", "subject_id")
    if pid is None:
        raise ValueError(f"participant_id sutunu yok. Sutunlar: {list(df.columns)}")
    df = df.rename(columns={pid: "participant_id"})
    df["participant_id"] = df["participant_id"].str.strip()

    diag = _find_col(df, "diagnosis", "group", "dx")
    if diag is None:
        raise ValueError(f"diagnosis sutunu yok. Sutunlar: {list(df.columns)}")
    raw = df[diag].astype(str).str.strip().str.upper()
    df["diagnosis_raw"] = raw
    df["diagnosis"] = raw.map(lambda v: CANONICAL_LABELS.get(v, v))

    age = _find_col(df, "age")
    df["age"] = pd.to_numeric(df[age], errors="coerce") if age else np.nan

    gen = _find_col(df, "gender", "sex")
    df["gender"] = df[gen].astype(str).str.strip().str.upper() if gen else "NA"

    ghost = _find_col(df, "ghost_NoGhost", "ghost")
    df["ghost"] = df[ghost].astype(str).str.strip() if ghost else "unknown"

    t1 = _find_col(df, "T1w")
    if t1:
        df["has_T1w_flag"] = pd.to_numeric(df[t1], errors="coerce").fillna(0).astype(int)
    else:
        df["has_T1w_flag"] = 1

    scanner = _find_col(df, "ScannerSerialNumber", "scanner")
    df["scanner"] = df[scanner].astype(str) if scanner else "NA"

    return df[["participant_id", "diagnosis", "diagnosis_raw", "age",
               "gender", "ghost", "has_T1w_flag", "scanner"]]


def find_t1w(bids_dir, participant_id: str):
    """Bir denegin T1w dosyasini bul (yoksa None)."""
    anat = Path(bids_dir) / participant_id / "anat"
    if not anat.is_dir():
        return None
    cands = sorted(anat.glob("*_T1w.nii.gz")) + sorted(anat.glob("*_T1w.nii"))
    return cands[0] if cands else None


def build_subject_table(bids_dir, exclude_ghost: bool = True) -> pd.DataFrame:
    """participants.tsv + diskteki T1w dosyalarini birlestir."""
    df = load_participants(bids_dir)
    df["t1w_path"] = [
        str(p) if (p := find_t1w(bids_dir, s)) is not None else ""
        for s in df["participant_id"]
    ]
    df["has_t1w_file"] = df["t1w_path"] != ""

    df["ghost_ok"] = True
    if exclude_ghost and "ghost" in df:
        known = df["ghost"].str.lower()
        df["ghost_ok"] = ~known.str.contains("^ghost$", regex=True, na=False)

    df["included"] = df["has_t1w_file"] & df["ghost_ok"]
    return df


def apply_task(df: pd.DataFrame, task_map: dict):
    """
    Gorev haritasina gore etiketleri yeniden atar ve haritada olmayan
    denekleri atar. Donen: (filtrelenmis df, sinif isimleri listesi).
    """
    keep = df["diagnosis"].isin(task_map.keys())
    out = df.loc[keep].copy()
    out["label_name"] = out["diagnosis"].map(task_map)
    classes = sorted(out["label_name"].unique())
    mapping = {c: i for i, c in enumerate(classes)}
    out["label"] = out["label_name"].map(mapping).astype(int)
    return out.reset_index(drop=True), classes
