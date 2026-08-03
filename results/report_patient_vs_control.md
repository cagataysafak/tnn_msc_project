# Sonuclar -- gorev: patient_vs_control

- Denek sayisi: 217  {'CONTROL': 110, 'PATIENT': 107}
- Tensor: (217, 64, 64, 64), Tucker rank (8, 8, 8) (512 ozellik)
- CV: 5-fold x 3 tekrar, seed 42
- NN backend: PyTorch 2.13.0+cpu

## Ozet

| method        |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   kappa |   auc |
|:--------------|-----------:|--------------------:|-----------:|--------------:|--------:|------:|
| Tucker+MLP    |      0.544 |               0.544 |      0.517 |         0.517 |   0.089 | 0.561 |
| Tucker+LogReg |      0.607 |               0.607 |      0.603 |         0.603 |   0.214 | 0.609 |
| PCA+MLP       |      0.525 |               0.526 |      0.466 |         0.466 |   0.053 | 0.552 |
| Voxel-MLP     |      0.602 |               0.603 |      0.600 |         0.600 |   0.205 | 0.615 |
| Dummy         |      0.518 |               0.517 |      0.515 |         0.515 |   0.034 | 0.517 |

## Istatistiksel testler (f1_macro)

| B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+LogReg |     -0.0862 |    -0.9478 |              0.1166 |         0.4662 |       0.0026 |            0.0105 |
| PCA+MLP       |      0.0510 |     0.3546 |              0.5415 |         1.0000 |       0.0962 |            0.1924 |
| Voxel-MLP     |     -0.0833 |    -0.8971 |              0.1357 |         0.4662 |       0.0035 |            0.0105 |
| Dummy         |      0.0022 |     0.0182 |              0.9749 |         1.0000 |       0.9780 |            0.9780 |