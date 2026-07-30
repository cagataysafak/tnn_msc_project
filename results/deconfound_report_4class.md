# Arindirilmis analiz -- gorev: 4class

- Denek: 217, siniflar ['ADHD', 'BIPOLAR', 'CONTROL', 'SCHZ'] [36, 41, 110, 30]
- Karistiricilar: age, gender=M, scanner=35426.0

## 1-2) Yontem karsilastirmasi

```
             method  balanced_accuracy  ba_ci_low  ba_ci_high  ba_ci_low_uncorrected  ba_ci_high_uncorrected  f1_macro  f1_macro_sd      auc  p_vs_chance
         Tucker+MLP           0.395551   0.305421    0.485681               0.354476                0.436625  0.348739     0.069040 0.652198     0.001900
Tucker(arinmis)+MLP           0.346305   0.280700    0.411911               0.316407                0.376203  0.298817     0.047349 0.613533     0.003557
        Demo+LogReg           0.356178   0.263267    0.449088               0.313836                0.398519  0.325198     0.067499 0.608061     0.013993
    Tucker+Demo+MLP           0.402116   0.295946    0.508287               0.353732                0.450501  0.344889     0.101287 0.658146     0.004132
    Dummy(cogunluk)           0.250000   0.250000    0.250000               0.250000                0.250000  0.168205     0.001300 0.500000     1.000000
```

```
                  A                   B   mean_A   mean_B     fark  p_corrected_ttest  p_wilcoxon  p_ttest_holm  p_wilcoxon_holm
         Tucker+MLP Tucker(arinmis)+MLP 0.348739 0.298817 0.049922           0.239656    0.002625      0.958623         0.010498
         Tucker+MLP         Demo+LogReg 0.348739 0.325198 0.023541           0.697567    0.454285      1.000000         0.990784
         Tucker+MLP     Tucker+Demo+MLP 0.348739 0.344889 0.003850           0.937080    1.000000      1.000000         1.000000
         Tucker+MLP     Dummy(cogunluk) 0.348739 0.168205 0.180534           0.000384    0.000061      0.001919         0.000366
Tucker(arinmis)+MLP     Dummy(cogunluk) 0.298817 0.168205 0.130612           0.000207    0.000061      0.001242         0.000366
    Tucker+Demo+MLP         Demo+LogReg 0.344889 0.325198 0.019691           0.788874    0.330261      1.000000         0.990784
```

## 3) Tarayici-disi genelleme

```
test_scanner  n_train  n_test  balanced_accuracy  balanced_accuracy_prior_adj  f1_macro      auc
     35343.0       78     139           0.319192                     0.295581  0.231888 0.611970
     35426.0      139      78           0.273538                     0.361393  0.242484 0.614159
```
