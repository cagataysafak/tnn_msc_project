# Arindirilmis analiz -- gorev: patient_vs_control

- Denek: 217, siniflar ['CONTROL', 'PATIENT'] [110, 107]
- Karistiricilar: age, gender=M, scanner=35426.0

## 1-2) Yontem karsilastirmasi

```
             method  balanced_accuracy  ba_ci_low  ba_ci_high  ba_ci_low_uncorrected  ba_ci_high_uncorrected  f1_macro  f1_macro_sd      auc  p_vs_chance
         Tucker+MLP           0.544300   0.475767    0.612833               0.513068                0.575532  0.517163     0.086419 0.560659     0.093655
Tucker(arinmis)+MLP           0.535498   0.476292    0.594704               0.508516                0.562479  0.491757     0.091532 0.531254     0.109663
        Demo+LogReg           0.667244   0.589843    0.744645               0.631971                0.702517  0.662986     0.063494 0.717982     0.000193
    Tucker+Demo+MLP           0.526984   0.439112    0.614856               0.486939                0.567030  0.505090     0.088675 0.548564     0.260410
    Dummy(cogunluk)           0.500000   0.500000    0.500000               0.500000                0.500000  0.336410     0.002600 0.500000     1.000000
```

```
                  A                   B   mean_A   mean_B      fark  p_corrected_ttest  p_wilcoxon  p_ttest_holm  p_wilcoxon_holm
         Tucker+MLP Tucker(arinmis)+MLP 0.517163 0.491757  0.025407           0.661731    0.278707      1.000000         0.557415
         Tucker+MLP         Demo+LogReg 0.517163 0.662986 -0.145822           0.005230    0.000061      0.025189         0.000366
         Tucker+MLP     Tucker+Demo+MLP 0.517163 0.505090  0.012073           0.742427    0.432626      1.000000         0.557415
         Tucker+MLP     Dummy(cogunluk) 0.517163 0.336410  0.180753           0.002405    0.000653      0.014432         0.002613
Tucker(arinmis)+MLP     Dummy(cogunluk) 0.491757 0.336410  0.155347           0.009121    0.001201      0.027364         0.003602
    Tucker+Demo+MLP         Demo+LogReg 0.505090 0.662986 -0.157896           0.005038    0.000122      0.025189         0.000610
```

## 3) Tarayici-disi genelleme

```
test_scanner  n_train  n_test  balanced_accuracy  balanced_accuracy_prior_adj  f1_macro      auc
     35343.0       78     139           0.422336                     0.500000  0.403283 0.407483
     35426.0      139      78           0.572414                     0.616379  0.460119 0.618103
```
