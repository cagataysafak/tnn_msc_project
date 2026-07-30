# Sonuclar -- gorev: schz_vs_control

- Denek sayisi: 175  {'CONTROL': 125, 'SCHZ': 50}
- Tensor: (175, 64, 64, 64), Tucker rank (8, 8, 8) (512 ozellik)
- CV: 5-fold x 3 tekrar, seed 42
- NN backend: PyTorch 2.13.0+cpu

## Ozet

| method        |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   kappa |   auc |
|:--------------|-----------:|--------------------:|-----------:|--------------:|--------:|------:|
| Tucker+MLP    |      0.735 |               0.701 |      0.685 |         0.737 |   0.380 | 0.766 |
| Tucker+LogReg |      0.709 |               0.636 |      0.631 |         0.702 |   0.273 | 0.710 |
| PCA+MLP       |      0.530 |               0.561 |      0.433 |         0.452 |   0.113 | 0.637 |
| Voxel-MLP     |      0.703 |               0.688 |      0.665 |         0.711 |   0.345 | 0.763 |
| Dummy         |      0.621 |               0.529 |      0.526 |         0.616 |   0.058 | 0.529 |

## Istatistiksel testler (f1_macro)

| B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+LogReg |      0.0548 |     0.7097 |              0.2278 |         0.4557 |       0.0157 |            0.0313 |
| PCA+MLP       |      0.2525 |     1.3943 |              0.0266 |         0.0798 |       0.0012 |            0.0037 |
| Voxel-MLP     |      0.0205 |     0.2533 |              0.6596 |         0.6596 |       0.5098 |            0.5098 |
| Dummy         |      0.1597 |     1.7687 |              0.0072 |         0.0288 |       0.0001 |            0.0005 |