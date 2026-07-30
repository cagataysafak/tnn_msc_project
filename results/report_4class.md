# Sonuclar -- gorev: 4class

- Denek sayisi: 217  {'ADHD': 36, 'BIPOLAR': 41, 'CONTROL': 110, 'SCHZ': 30}
- Tensor: (217, 64, 64, 64), Tucker rank (8, 8, 8) (512 ozellik)
- CV: 5-fold x 3 tekrar, seed 42
- NN backend: PyTorch 2.13.0+cpu

## Ozet

| method        |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   kappa |   auc |
|:--------------|-----------:|--------------------:|-----------:|--------------:|--------:|------:|
| Tucker+MLP    |      0.357 |               0.396 |      0.349 |         0.353 |   0.142 | 0.652 |
| Tucker+LogReg |      0.475 |               0.412 |      0.418 |         0.479 |   0.207 | 0.651 |
| PCA+MLP       |      0.286 |               0.332 |      0.242 |         0.242 |   0.081 | 0.601 |
| Voxel-MLP     |      0.360 |               0.381 |      0.344 |         0.362 |   0.135 | 0.647 |
| Dummy         |      0.347 |               0.273 |      0.263 |         0.338 |   0.004 | 0.510 |

## Istatistiksel testler (f1_macro)

| B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+LogReg |     -0.0692 |    -1.0955 |              0.0736 |         0.2946 |       0.0004 |            0.0017 |
| PCA+MLP       |      0.1072 |     1.0847 |              0.0762 |         0.2946 |       0.0012 |            0.0035 |
| Voxel-MLP     |      0.0047 |     0.0648 |              0.9105 |         0.9105 |       0.8904 |            0.8904 |
| Dummy         |      0.0855 |     0.9350 |              0.1211 |         0.2946 |       0.0043 |            0.0085 |