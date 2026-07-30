# Arindirilmis analiz -- gorev: schz_vs_control

- Denek: 140, siniflar ['CONTROL', 'SCHZ'] [110, 30]
- Karistiricilar: age, gender=M, scanner=35426.0

## 1-2) Yontem karsilastirmasi

```
             method  balanced_accuracy  ba_ci_low  ba_ci_high  ba_ci_low_uncorrected  ba_ci_high_uncorrected  f1_macro  f1_macro_sd      auc  p_vs_chance
         Tucker+MLP           0.655556   0.543525    0.767586               0.604152                0.706959  0.621559 7.692973e-02 0.682828     0.004988
Tucker(arinmis)+MLP           0.587879   0.460487    0.715270               0.529428                0.646330  0.548394 8.753906e-02 0.650505     0.080571
        Demo+LogReg           0.742929   0.591008    0.894850               0.673223                0.812636  0.689581 1.098802e-01 0.777525     0.002033
    Tucker+Demo+MLP           0.667677   0.573357    0.761996               0.624400                0.710954  0.615906 5.714863e-02 0.722222     0.000951
    Dummy(cogunluk)           0.500000   0.500000    0.500000               0.500000                0.500000  0.440000 1.723785e-16 0.500000     1.000000
```

```
                  A                   B   mean_A   mean_B      fark  p_corrected_ttest  p_wilcoxon  p_ttest_holm  p_wilcoxon_holm
         Tucker+MLP Tucker(arinmis)+MLP 0.621559 0.548394  0.073165           0.280788    0.055359      0.793073         0.110718
         Tucker+MLP         Demo+LogReg 0.621559 0.689581 -0.068022           0.264358    0.027708      0.793073         0.083124
         Tucker+MLP     Tucker+Demo+MLP 0.621559 0.615906  0.005653           0.898209    0.875291      0.898209         0.875291
         Tucker+MLP     Dummy(cogunluk) 0.621559 0.440000  0.181559           0.000901    0.000645      0.005407         0.003871
Tucker(arinmis)+MLP     Dummy(cogunluk) 0.548394 0.440000  0.108394           0.045066    0.001763      0.225329         0.008813
    Tucker+Demo+MLP         Demo+LogReg 0.615906 0.689581 -0.073674           0.195930    0.008362      0.783720         0.033447
```

## 3) Tarayici-disi genelleme

```
test_scanner  n_train  n_test  balanced_accuracy  balanced_accuracy_prior_adj  f1_macro      auc
     35343.0       39     101           0.770202                     0.759091  0.614013 0.794949
     35426.0      101      39           0.581579                     0.525000  0.538462 0.757895
```
