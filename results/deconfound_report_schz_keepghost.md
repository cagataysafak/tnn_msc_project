# Arindirilmis analiz -- gorev: schz_vs_control

- Denek: 175, siniflar ['CONTROL', 'SCHZ'] [125, 50]
- Karistiricilar: age, gender=M, scanner=35426.0

## 1-2) Yontem karsilastirmasi

```
             method  balanced_accuracy  ba_ci_low  ba_ci_high  ba_ci_low_uncorrected  ba_ci_high_uncorrected  f1_macro  f1_macro_sd      auc  p_vs_chance
         Tucker+MLP           0.700667   0.619003    0.782330               0.663197                0.738136  0.685461 5.627205e-02 0.766133     0.000059
Tucker(arinmis)+MLP           0.637333   0.544462    0.730205               0.594721                0.679946  0.611326 1.020453e-01 0.670933     0.003397
        Demo+LogReg           0.693333   0.605625    0.781042               0.653090                0.733577  0.668129 6.462820e-02 0.758400     0.000162
    Tucker+Demo+MLP           0.694000   0.610530    0.777470               0.655701                0.732299  0.674677 6.935084e-02 0.763733     0.000100
    Dummy(cogunluk)           0.500000   0.500000    0.500000               0.500000                0.500000  0.416667 1.149190e-16 0.500000     1.000000
```

```
                  A                   B   mean_A   mean_B     fark  p_corrected_ttest  p_wilcoxon  p_ttest_holm  p_wilcoxon_holm
         Tucker+MLP Tucker(arinmis)+MLP 0.685461 0.611326 0.074135       2.161544e-01    0.015076      0.864618         0.060303
         Tucker+MLP         Demo+LogReg 0.685461 0.668129 0.017331       7.608438e-01    0.389404      1.000000         0.991607
         Tucker+MLP     Tucker+Demo+MLP 0.685461 0.674677 0.010784       7.440812e-01    0.330536      1.000000         0.991607
         Tucker+MLP     Dummy(cogunluk) 0.685461 0.416667 0.268794       6.824615e-07    0.000653      0.000004         0.003920
Tucker(arinmis)+MLP     Dummy(cogunluk) 0.611326 0.416667 0.194659       4.400848e-03    0.000982      0.022004         0.004908
    Tucker+Demo+MLP         Demo+LogReg 0.674677 0.668129 0.006548       9.128933e-01    0.678772      1.000000         0.991607
```

## 3) Tarayici-disi genelleme

```
test_scanner  n_train  n_test  balanced_accuracy  balanced_accuracy_prior_adj  f1_macro      auc
     35343.0       48     127           0.717451                     0.647255  0.665496 0.738824
     35426.0      127      48           0.673043                     0.704348  0.661376 0.758261
```
