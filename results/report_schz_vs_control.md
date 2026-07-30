# Sonuclar -- gorev: schz_vs_control

- Denek sayisi: 140  {'CONTROL': 110, 'SCHZ': 30}
- Tensor: (140, 64, 64, 64), Tucker rank (8, 8, 8) (512 ozellik)
- CV: 5-fold x 3 tekrar, seed 42
- NN backend: PyTorch 2.13.0+cpu

## Ozet

| method        |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   kappa |   auc |
|:--------------|-----------:|--------------------:|-----------:|--------------:|--------:|------:|
| Tucker+MLP    |      0.700 |               0.656 |      0.622 |         0.719 |   0.259 | 0.683 |
| Tucker+LogReg |      0.738 |               0.611 |      0.610 |         0.737 |   0.223 | 0.683 |
| PCA+MLP       |      0.533 |               0.541 |      0.400 |         0.471 |   0.062 | 0.599 |
| Voxel-MLP     |      0.664 |               0.637 |      0.592 |         0.687 |   0.217 | 0.715 |
| Dummy         |      0.669 |               0.490 |      0.486 |         0.661 |  -0.021 | 0.490 |

## Istatistiksel testler (f1_macro)

| B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+LogReg |      0.0120 |     0.1118 |              0.8454 |         1.0000 |       0.7332 |            0.7332 |
| PCA+MLP       |      0.2217 |     1.1314 |              0.0640 |         0.2562 |       0.0037 |            0.0149 |
| Voxel-MLP     |      0.0297 |     0.3321 |              0.5645 |         1.0000 |       0.2209 |            0.4418 |
| Dummy         |      0.1351 |     0.9868 |              0.1014 |         0.3041 |       0.0063 |            0.0189 |