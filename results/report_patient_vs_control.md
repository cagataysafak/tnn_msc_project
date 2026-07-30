# Sonuclar -- gorev: patient_vs_control

- Denek sayisi: 217  {'CONTROL': 110, 'PATIENT': 107}
- Tensor: (217, 64, 64, 64), Tucker rank (8, 8, 8) (512 ozellik)
- CV: 5-fold x 5 tekrar, seed 42
- NN backend: PyTorch 2.13.0+cpu

## Ozet

| method        |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   kappa |   auc |
|:--------------|-----------:|--------------------:|-----------:|--------------:|--------:|------:|
| Tucker+MLP    |      0.541 |               0.542 |      0.520 |         0.520 |   0.084 | 0.559 |
| Tucker+LogReg |      0.594 |               0.595 |      0.591 |         0.591 |   0.189 | 0.609 |
| PCA+MLP       |      0.512 |               0.513 |      0.462 |         0.461 |   0.027 | 0.545 |
| Voxel-MLP     |      0.602 |               0.602 |      0.598 |         0.598 |   0.204 | 0.615 |
| Dummy         |      0.513 |               0.513 |      0.510 |         0.510 |   0.025 | 0.513 |

## Istatistiksel testler (f1_macro)

| B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+LogReg |     -0.0707 |    -0.6895 |              0.2159 |         0.7681 |       0.0042 |            0.0125 |
| PCA+MLP       |      0.0585 |     0.4631 |              0.4018 |         0.8035 |       0.0164 |            0.0328 |
| Voxel-MLP     |     -0.0776 |    -0.7283 |              0.1920 |         0.7681 |       0.0015 |            0.0061 |
| Dummy         |      0.0107 |     0.0831 |              0.8796 |         0.8796 |       0.7712 |            0.7712 |