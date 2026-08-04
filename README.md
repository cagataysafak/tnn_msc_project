# Tensors and Neural Networks — MSc Project

**Dataset:** UCLA Consortium for Neuropsychiatric Phenomics, LA5c (OpenNeuro `ds000030`, v1.0.0)
**Method:** Partial Tucker decomposition (MPCA) → psychiatric diagnosis classification with a multi-layer perceptron (MLP)
**Target hardware:** CPU-only laptop (i7-13650HX, 16 GB RAM). CUDA **not required**.

---

## 1. Dataset structure (brief overview)

A dataset organized according to the BIDS ("Brain Imaging Data Structure") standard.
What the folders mean:

| Path | Contents |
|---|---|
| `participants.tsv` | One row per subject: `participant_id`, `diagnosis`, `age`, `gender`, `ghost_NoGhost`, which scans are available |
| `sub-XXXXX/anat/` | **T1w structural MRI** — a single 3D volume, brain anatomy (~1 mm resolution) |
| `sub-XXXXX/func/` | fMRI — 4D (x, y, z, **time**). A separate file per task |
| `sub-XXXXX/dwi/` | Diffusion MRI — for white matter tracts, `.bval`/`.bvec` direction info |
| `sub-XXXXX/beh/` | Behavioral test records collected outside the scanner |
| `derivatives/` | Quality control outputs (MRIQC), plots — not raw data |
| `phenotype/` | Neuropsychological test scores |

Diagnosis groups: `CONTROL` (138), `SCHZ` (58), `BIPOLAR` (49), `ADHD` (45).

**Why does this project use only `anat/*_T1w.nii.gz`?**

1. **Natural tensor structure:** Each T1w file is a single 3-way tensor `(x, y, z)`.
   Stacking all subjects together gives a 4-way tensor `(subject, x, y, z)` — ideal
   for Tucker decomposition.
2. **Size:** The full dataset is tens of GB; T1w alone is ≈ **2.5–3 GB**.
3. **Preprocessing:** Using fMRI would require motion correction + MNI registration
   (fMRIPrep) — on CPU that's ~1–2 hours per subject, i.e. weeks for 265 subjects.
   For T1w, reasonable spatial normalization is possible without registration software.

> If you want to build an fMRI connectivity tensor, I've described it in §8
> under "possible extensions" — but that's for after the MSc project submission.

---

## 2. Methodology — why exactly "Tucker + MLP" and not something else?

Your proposed Tucker + MLP idea is correct. I refined it in two places:

### 2.1 **Partial Tucker (MPCA)** instead of full Tucker

The data tensor is `X ∈ R^{N × d₁ × d₂ × d₃}` (N subjects). Full Tucker would also
compress the **subject mode** — but then when a new subject arrives, you cannot
project it into the same space, because the subject-mode factor belongs only to
the training subjects. That's why the subject mode is **not compressed**:

```
X ≈ C ×₁ U₁ ×₂ U₂ ×₃ U₃        U_k ∈ R^{d_k × r_k},  U_kᵀU_k = I

Feature for each subject:
C_n = (X_n − X̄) ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ   ∈ R^{r₁ × r₂ × r₃}
```

This is exactly **Multilinear PCA** (Lu, Plataniotis & Venetsanopoulos, 2008)
and the standard form of TensorFace/Tucker-based feature extraction.
The `U_k` matrices are found via **HOOI** (higher-order orthogonal iteration);
if `n_iter=0` is given, it falls back to single-pass **HOSVD** — you can
compare the two.

**Numerical gain:** `64³ = 262,144` voxels → `8³ = 512` features.
A 512× compression, making it feasible to work with ~160 training examples.

### 2.2 Avoiding **data leakage** — the most critical point

The vast majority of neuroimaging studies in the literature make a mistake here:
they perform the decomposition **on the entire dataset** once and then run CV.
This leaks information about the test subjects into the factor matrices and
artificially inflates performance.

In this project, `MPCA.fit()` is called **only on the training split** within
**each fold**; the test split is only ever `transform()`-ed. The same rule
applies to `StandardScaler` and `PCA`. Be sure to emphasize this in your
report — the committee will ask about it.

### 2.3 Methods compared

| Method | What it tests |
|---|---|
| **Tucker+MLP** | MAIN METHOD |
| Tucker+LogReg | Does the gain come from the NN, or from the features? |
| PCA+MLP | Is preserving tensor structure better than flattening and doing PCA? |
| Voxel-MLP | **Decomposition-free baseline** — pooled raw voxels → MLP |
| Dummy | Chance level (accuracy is misleading due to class imbalance) |

### 2.4 Statistical comparison

- 5-fold cross-validation × 3 repeats = 15 folds, **stratified**.
- Metrics: accuracy, **balanced accuracy**, **macro F1**, weighted F1,
  Cohen's κ, ROC-AUC. Since the classes are imbalanced, macro F1 and
  balanced accuracy are the primary metrics.
- **Nadeau–Bengio corrected paired t-test**: in repeated CV, folds are
  not independent (training sets overlap), so a plain t-test gives
  overly optimistic p-values. This correction adds a `1/k + n_test/n_train`
  term to the variance.
- The distribution-free **Wilcoxon signed-rank test** is also reported.
- **Holm–Bonferroni** correction for multiple comparisons.
- Effect size: paired Cohen's d.

---

## 3. Setup

```bash
# Virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**About PyTorch:** On Windows, `pip install torch` already installs the CPU version.
On Linux, if you want CPU-only:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`
If PyTorch is not installed at all, the project automatically falls back to
`sklearn.neural_network.MLPClassifier` and still works. **CUDA is not required.**

---

## 4. Running

### Step 0a — Smoke test first (without downloading data, ~2 minutes)

Verify that the pipeline runs error-free on your laptop:

```bash
python run_all.py --smoke
```

This generates 80 synthetic subjects under `data/synthetic/` with the same
structure as ds000030, runs all steps, and writes figures + tables to `results/`.
The results are meaningless (the data is synthetic) — **the goal is only to
confirm the code runs**.

### Step 0b — Download the real data (~2.5–3 GB)

```bash
pip install openneuro-py
python step0_download_t1w.py --target-dir data/ds000030
```

If that doesn't work, alternatives (also documented inside `step0_download_t1w.py`):

```bash
# AWS CLI — no account/registration needed
aws s3 sync --no-sign-request s3://openneuro.org/ds000030 data/ds000030 \
    --exclude "*" --include "participants.tsv" --include "*_T1w.nii.gz"

# DataLad
datalad clone https://github.com/OpenNeuroDatasets/ds000030.git data/ds000030
cd data/ds000030 && datalad get "sub-*/anat/*_T1w.nii.gz"
```

Or one by one from the browser: <https://openneuro.org/datasets/ds000030/versions/1.0.0>

### Step 1 — Build the tensor dataset (~5–15 min)

```bash
python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 --n-jobs 4
```

Output: `derivatives/X_64.npy` — `(N, 64, 64, 64)` float32, ~270 MB.

> `--n-jobs` requires ~0.5 GB RAM per process. 4–6 is comfortable on 16 GB.
> If you run low on RAM, use `--size 48` (volume shrinks by 2.4×).

### Step 2 — Run all analyses with a single command

```bash
python run_all.py --bids-dir data/ds000030
```

This runs all of the following in sequence:

| | What it does |
|---|---|
| STEP 1 | Builds the tensor dataset — twice: the main cohort (ghost-artifact subjects excluded) and a sensitivity cohort (`--keep-ghost`, `derivatives_ghost/`) |
| STEP 2 | Tucker+MLP, baselines, statistics, figures, rank sweep for each task |
| STEP 3 | Confounder and selection-bias checks |
| STEP 4 | Covariate regression + out-of-scanner generalization |
| STEP 5 | Collects everything into a single report (`THESIS_REPORT.md` + LaTeX tables) |

Default runs: `schz_vs_control`, `patient_vs_control`, `4class` (main cohort)
and `schz_keepghost` (sensitivity analysis).

Useful options:

```bash
# Skip preprocessing if already done
python run_all.py --bids-dir data/ds000030 --skip-step1

# Quick version: single task, fewer repeats, no rank sweep or sensitivity run
python run_all.py --bids-dir data/ds000030 --tasks schz_vs_control \
    --repeats 2 --no-rank-sweep --no-ghost-run

# Regenerate the report only (no analysis, takes seconds)
python run_all.py --only-report

# See what would be run (runs nothing)
python run_all.py --dry-run

# If you're tight on RAM
python run_all.py --bids-dir data/ds000030 --size 48 --skip-pca
```

If a run fails, the pipeline **does not stop**; the summary table at the end
shows which step failed and how long it took. You can re-run the failed step
on its own — the command lines are printed to the screen.

### If you want to run steps individually

```bash
python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 --n-jobs 4
python step2_run_experiments.py --task schz_vs_control --ranks 8 8 8 --repeats 3
python step3_confound_checks.py --bids-dir data/ds000030 --task schz_vs_control
python step4_deconfound.py --task schz_vs_control
python step5_make_report.py
```

Tasks: `schz_vs_control`, `patient_vs_control`, `4class`.
`--tag` writes each run's outputs to separate files (available in step2/3/4).

---

## 5. Outputs

```
derivatives/
  X_64.npy                     (N, 64, 64, 64) tensor dataset
  meta_64.csv                  subject, diagnosis, age, gender + preprocessing QC metrics
results/
  report_<task>.md             automatic summary report (can be pasted directly into the thesis)
  run_info_<task>.json         all hyperparameters (for reproducibility)
  tables/
    fold_metrics_*.csv         raw metrics per fold × per method
    summary_*.csv              mean ± sd + 95% CI
    statistical_tests_*.csv    t-test, Wilcoxon, Cohen's d, Holm correction
    per_class_f1_*.csv         per-class F1
    compression_*.csv          rank vs reconstruction error
  figures/
    fig01 demographics         class/age/gender distribution
    fig02 preprocessing QC     slice montage of processed volumes
    fig03 class means          group mean volumes + differences
    fig04 MPCA spectrum        per-mode eigenvalues, cumulative variance
    fig05 rank vs error        compression quality curve
    fig06 reconstruction       visual comparison at different ranks
    fig07 eigen-volumes        u₁∘u₂∘u₃ rank-1 components
    fig08 embedding            PCA + t-SNE, colored by class
    fig09 training curve       MLP loss curves
    fig10 confusion matrix     for each method
    fig11 box plots            fold-level performance distribution
    fig12 ROC                  for binary tasks
    fig13 rank sweep           with --rank-sweep
```

---

## 6. What does the preprocessing actually do?

`tnn/preprocessing.py` applies a coarse spatial normalization using plain
numpy/scipy, **without** registration software:

1. **Canonical RAS orientation** (`nibabel.as_closest_canonical`) → axis order
   is the same for all subjects.
2. **Otsu thresholding + largest connected component + hole filling** → head
   mask, background noise discarded.
3. **Crop to bounding box** → head position normalized.
4. **Pad to a cube in mm** → anatomical aspect ratio preserved.
5. **Resample to a `64³` grid** → head size normalized.
6. **Intensity scaling by the 99th percentile within the mask** → reduces
   scanner-to-scanner differences.

### Limitations (make sure to include these in your report)

- This is **not full MNI registration**. Voxel-level anatomical correspondence
  across subjects is approximate; fine structural differences get blurred.
- The Otsu mask also includes the skull and scalp (this is not true
  *skull-stripping*). Could be improved with FSL BET / ANTs.
- Because head size is normalized, **brain volume difference information is
  lost** — which is actually a known marker in schizophrenia. You could skip
  the `pad_to_cube_mm` step and use a fixed mm box instead to preserve this
  information (would make for a nice ablation experiment).
- ~20% of ds000030 has ghosting (aliasing) artifacts in the T1w scans; by
  default these are excluded via the `ghost_NoGhost` column (included with
  `--keep-ghost`).

---

## 7. An honest warning about expected results

Classifying psychiatric diagnosis from raw structural MRI is a **hard
problem**. Even with proper registration + voxel-based morphometry, the
literature typically reports **65–80% accuracy** for schizophrenia-vs-control;
performance on the 4-class problem is much closer to chance level.

So the scientific value of this project is not "high accuracy" but a
**controlled comparison**: *does tensor decomposition provide a statistically
significant gain over flattening / a decomposition-free baseline?* Even if
the answer is "no," that is still a valid and reportable finding — as long
as the methodology is sound (no leakage, appropriate metrics, appropriate
tests). The code satisfies all of these.

That's why the `Dummy` baseline exists: even if accuracy comes out at 68%,
if 68% of the classes are CONTROL, the model may have learned nothing.
**Look at balanced accuracy and macro F1.**

---

## 8. Possible extensions (optional)

1. **Rank ablation** — `--rank-sweep` already exists; include `fig13` in the report.
2. **HOSVD vs HOOI** — set `MPCA_N_ITER = 0` in `tnn/config.py` and re-run.
3. **CP (PARAFAC) decomposition** — compare against Tucker.
4. **3D CNN** — instead of MLP; slow on CPU but feasible with `--size 48`.
5. **fMRI connectivity tensor** — `rest` fMRI → atlas ROI time series →
   `(subject × ROI × ROI)` correlation tensor → Tucker. Requires fMRIPrep
   for registration; alternatively could be tried with GPU on Google Colab.
6. **Multimodal fusion** — decompose T1w + DWI features separately and
   concatenate them at the MLP input.

---

## 9. If you want to use Google Colab

Not required — the project runs comfortably on CPU. But if you want to try it:

```python
!pip install nibabel openneuro-py -q
!git clone <this-project's-repo-address> tnn_project   # or upload the files
%cd tnn_project
!python step0_download_t1w.py --target-dir data/ds000030
!python run_all.py --bids-dir data/ds000030 --size 64 --n-jobs 2
```

**Note:** The disk is wiped when the Colab session closes; you'll need to
re-download the ~3 GB of data every time. It's more practical to save
`derivatives/X_64.npy` (~270 MB) to Google Drive and load it from there in
subsequent sessions.

---

## 10. File map

```
step0_download_t1w.py        downloads only T1w from OpenNeuro
step0_make_synthetic_bids.py generates fake BIDS data (smoke test)
step1_build_tensor_dataset.py  T1w → (N,64,64,64) tensor
step2_run_experiments.py     CV + Tucker+MLP + baselines + statistics + figures
step3_confound_checks.py     selection bias + demographic baseline + feature content
step4_deconfound.py          covariate regression + out-of-scanner generalization
step5_make_report.py         collects all results into thesis-ready md + LaTeX tables
run_all.py                   runs everything in sequence
tnn/
  config.py         all settings in one place
  data.py           participants.tsv reading, task definitions
  preprocessing.py  Otsu mask, bbox cropping, resampling
  tensor_utils.py   unfold / n-mode product / MPCA (HOOI) — with self-tests
  nn_utils.py       PyTorch MLP + training loop (with sklearn fallback)
  evaluation.py     metrics + Nadeau-Bengio test + Holm correction
  viz_utils.py      all figures
```

To verify the tensor algebra:

```bash
python -m tnn.tensor_utils
```

(Runs 6 tests, including unfold/n-mode product consistency, zero error at
full rank, HOOI ≤ HOSVD error.)

---

## 11. Step 3 — confound checks

`step2` gives you a performance number. `step3` asks **where that number
comes from** — four questions that will come up during the defense:

```bash
python step3_confound_checks.py --bids-dir data/ds000030 --task schz_vs_control
python step3_confound_checks.py --bids-dir data/ds000030 --task 4class
```

| Check | Question | Why it's critical |
|---|---|---|
| **A** | Is the ghost-artifact exclusion biased by diagnosis? (chi-square) | If biased, there is selection bias; the `--keep-ghost` sensitivity analysis becomes essential |
| **B** | Are the groups balanced in terms of age / gender / scanner? | If not, the model may be learning demographics rather than diagnosis |
| **C** | What do you get with **demographics only** (age+gender+scanner, no images at all) on the same folds? | The one honest test that measures what MRI adds on top of demographics |
| **D** | What do the Tucker features actually encode — age, scanner, or diagnosis? | The features may be carrying nuisance variables rather than diagnosis |

Outputs: `results/confound_report_<task>.md`,
`results/tables/confound_*.csv`, `results/figures/fig14_feature_content_*.png`.

**How to interpret:** if in step C no imaging method significantly beats the
demographic baseline, the reported success may largely stem from demographic
differences — you must state this as a limitation. In step D, if the CV R²
for age is higher than the value obtained for diagnosis, it means the
features predominantly encode age.

---

## 12. Recommended additional runs

```bash
# 1. Ghost-exclusion sensitivity analysis (sample size ~217 -> ~265)
python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 \
    --n-jobs 4 --keep-ghost --out-dir derivatives_ghost
python step2_run_experiments.py --deriv-dir derivatives_ghost --size 64 \
    --task schz_vs_control --tag schz_keepghost

# 2. The most balanced contrast (~107 patients vs ~110 controls) -- highest power
python step2_run_experiments.py --size 64 --task patient_vs_control --repeats 5

# 3. Test for overfitting: smaller rank
python step2_run_experiments.py --size 64 --task 4class --ranks 4 4 4 \
    --tag 4class_r4

# 4. HOSVD vs HOOI ablation -- set MPCA_N_ITER = 0 in tnn/config.py
```

`--tag` writes each run's outputs to separate files (available in step2, step3 and step4), so they don't
overwrite each other.

---

## 13. Step 4 — deconfounded analysis

If `step3` finds a confounder (which it does for this dataset), `step4`
answers this single question: **once the effects of age, gender, and scanner
are removed from the features, does any signal about diagnosis remain in the
brain data?**

```bash
python step4_deconfound.py --size 64 --task schz_vs_control
python step4_deconfound.py --size 64 --task 4class
```

It performs three analyses:

1. **Covariate regression (residualization).** For each feature, a model
   `f = b₀ + b₁·age + b₂·gender + b₃·scanner + e` is fit using **only the
   training data**, and the residual `e` is used for classification.
2. **Comparative set of methods** (same folds, paired tests):
   raw Tucker · deconfounded Tucker · demographics only ·
   image+demographics combined · majority-class dummy.
   For each method, a one-sample corrected t-test checks whether balanced
   accuracy is significantly above chance level.
3. **Out-of-scanner generalization (leave-one-scanner-out).** Train on one
   scanner, test on another. If the model learned anatomy, it should still
   work when the scanner changes; if it learned site effects, it collapses.

Output: `results/deconfound_report_<task>.md`,
`results/tables/deconf_*.csv`, `results/figures/fig15_deconfound_*.png`.


### Reporting note — confidence intervals

The 95% confidence intervals produced by `step4` must be **Nadeau–Bengio
corrected**, i.e. they share the same assumptions as the p-values. A plain
t-interval (`sd/√k`) assumes the folds are independent; since training sets
overlap in repeated CV, this interval comes out **narrower** than it actually
is and can contradict the corrected p-value (the interval may exclude chance
while p remains non-significant). The uncorrected narrow intervals are kept
in the CSV under `*_uncorrected` columns for comparison — **use the corrected
one in the thesis.**

### AUC is the primary metric in LOSO

Because the scanners have very different class distributions, the `argmax`
decision carries the training prior, so balanced accuracy can collapse while
AUC remains robust (prior shift). For this reason, `step4` reports both the
raw and the **prior-corrected** balanced accuracy (the posterior is divided
by the training prior — test labels are not used) as well as AUC, and bases
its interpretation on **AUC**.

### A conceptual warning — put this in the report

Scanner and diagnosis are strongly associated in this dataset (χ²=21.7,
p<0.001). Regressing out a variable that is associated with the outcome also
removes some of the diagnosis signal (**over-correction**). So the
deconfounded result is a **lower bound**:

- If it's still significantly above chance after deconfounding → there is
  structural signal that cannot be explained by site.
- If it's indistinguishable from chance → *"in this sample, once linear
  confound effects are removed, no measurable structural diagnostic signal
  remains."* This is a valid finding; it does not mean the method is bad.

`step4`'s correctness was validated with two synthetic scenarios: when a
real signal is present, the signal is preserved after deconfounding
(p<0.001); when the signal comes only from a site effect, it drops to chance
level after deconfounding (p=0.12).

---

## 14. Step 5 — thesis-ready combined report

The results from steps 1–4 are spread across 15+ CSV files. `step5` scans
all of them and collects everything into a single report:

```bash
python step5_make_report.py --size 64
```

Outputs:

| File | Contents |
|---|---|
| `results/THESIS_REPORT.md` | Numbered markdown tables + a list of factual findings |
| `results/latex/*.tex` | Booktabs version of the same tables (added to the thesis via `\input{}`) |

Tags (`--tag`) are found automatically, so if you ran additional runs, they
will also appear in the report. Missing files are silently skipped — it also
works with partial results.

The LaTeX tables require `\usepackage{booktabs}` and are pdfLaTeX-compatible
(Unicode characters like `$\pm$`, `$<$` are converted to math mode).

**This script does not perform new analysis and does not generate
interpretation.** It only aggregates existing numbers and builds conditional
factual sentences based on p-values ("… p=0.0806 → not significant"). You
will write the discussion section yourself; these are ready-made materials.
