# Sonuclar -- birlesik rapor

Bu dosya `step5_make_report.py` tarafindan mevcut CSV ciktilarindan
otomatik uretildi. Tablolar `results/latex/` altinda LaTeX olarak da
mevcuttur (`\usepackage{booktabs}` gerekir).

Kosular: `4class`, `patient_vs_control`, `schz_keepghost`, `schz_vs_control`

## Tablo 1. Capraz dogrulama performansi (fold ortalamasi ± sd) -- 4class

| method        | balanced_accuracy   | f1_macro      | auc           | accuracy      | kappa         |
|:--------------|:--------------------|:--------------|:--------------|:--------------|:--------------|
| Tucker+MLP    | 0.396 ± 0.074       | 0.349 ± 0.069 | 0.652 ± 0.049 | 0.357 ± 0.079 | 0.142 ± 0.092 |
| Tucker+LogReg | 0.412 ± 0.060       | 0.418 ± 0.065 | 0.651 ± 0.050 | 0.475 ± 0.067 | 0.207 ± 0.093 |
| PCA+MLP       | 0.332 ± 0.068       | 0.242 ± 0.091 | 0.601 ± 0.068 | 0.286 ± 0.096 | 0.081 ± 0.071 |
| Voxel-MLP     | 0.381 ± 0.066       | 0.344 ± 0.076 | 0.647 ± 0.050 | 0.360 ± 0.084 | 0.135 ± 0.078 |
| Dummy         | 0.273 ± 0.053       | 0.263 ± 0.053 | 0.510 ± 0.036 | 0.347 ± 0.052 | 0.004 ± 0.080 |

## Tablo 2. Capraz dogrulama performansi (fold ortalamasi ± sd) -- patient_vs_control

| method        | balanced_accuracy   | f1_macro      | auc           | accuracy      | kappa         |
|:--------------|:--------------------|:--------------|:--------------|:--------------|:--------------|
| Tucker+MLP    | 0.542 ± 0.075       | 0.520 ± 0.093 | 0.559 ± 0.081 | 0.541 ± 0.076 | 0.084 ± 0.149 |
| Tucker+LogReg | 0.595 ± 0.055       | 0.591 ± 0.056 | 0.609 ± 0.055 | 0.594 ± 0.055 | 0.189 ± 0.111 |
| PCA+MLP       | 0.513 ± 0.068       | 0.462 ± 0.096 | 0.545 ± 0.087 | 0.512 ± 0.068 | 0.027 ± 0.135 |
| Voxel-MLP     | 0.602 ± 0.073       | 0.598 ± 0.075 | 0.615 ± 0.074 | 0.602 ± 0.072 | 0.204 ± 0.145 |
| Dummy         | 0.513 ± 0.079       | 0.510 ± 0.081 | 0.513 ± 0.079 | 0.513 ± 0.078 | 0.025 ± 0.158 |

## Tablo 3. Capraz dogrulama performansi (fold ortalamasi ± sd) -- schz_keepghost

| method        | balanced_accuracy   | f1_macro      | auc           | accuracy      | kappa         |
|:--------------|:--------------------|:--------------|:--------------|:--------------|:--------------|
| Tucker+MLP    | 0.701 ± 0.068       | 0.685 ± 0.056 | 0.766 ± 0.081 | 0.735 ± 0.050 | 0.380 ± 0.112 |
| Tucker+LogReg | 0.636 ± 0.068       | 0.631 ± 0.071 | 0.710 ± 0.069 | 0.709 ± 0.054 | 0.273 ± 0.132 |
| PCA+MLP       | 0.561 ± 0.097       | 0.433 ± 0.187 | 0.637 ± 0.131 | 0.530 ± 0.210 | 0.113 ± 0.193 |
| Voxel-MLP     | 0.688 ± 0.083       | 0.665 ± 0.084 | 0.763 ± 0.091 | 0.703 ± 0.085 | 0.345 ± 0.160 |
| Dummy         | 0.529 ± 0.067       | 0.526 ± 0.069 | 0.529 ± 0.067 | 0.621 ± 0.062 | 0.058 ± 0.137 |

## Tablo 4. Capraz dogrulama performansi (fold ortalamasi ± sd) -- schz_vs_control

| method        | balanced_accuracy   | f1_macro      | auc           | accuracy      | kappa          |
|:--------------|:--------------------|:--------------|:--------------|:--------------|:---------------|
| Tucker+MLP    | 0.656 ± 0.093       | 0.622 ± 0.077 | 0.683 ± 0.091 | 0.700 ± 0.069 | 0.259 ± 0.148  |
| Tucker+LogReg | 0.611 ± 0.085       | 0.610 ± 0.089 | 0.683 ± 0.059 | 0.738 ± 0.067 | 0.223 ± 0.177  |
| PCA+MLP       | 0.541 ± 0.080       | 0.400 ± 0.179 | 0.599 ± 0.162 | 0.533 ± 0.248 | 0.062 ± 0.133  |
| Voxel-MLP     | 0.637 ± 0.091       | 0.592 ± 0.074 | 0.715 ± 0.091 | 0.664 ± 0.081 | 0.217 ± 0.136  |
| Dummy         | 0.490 ± 0.073       | 0.486 ± 0.080 | 0.490 ± 0.073 | 0.669 ± 0.056 | -0.021 ± 0.160 |

## Tablo 5. Eslesmis karsilastirmalar, macro F1 -- 4class

| A          | B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:-----------|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+MLP | Tucker+LogReg |     -0.0692 |    -1.0955 |              0.0736 |         0.2946 |       0.0004 |            0.0017 |
| Tucker+MLP | PCA+MLP       |      0.1072 |     1.0847 |              0.0762 |         0.2946 |       0.0012 |            0.0035 |
| Tucker+MLP | Voxel-MLP     |      0.0047 |     0.0648 |              0.9105 |         0.9105 |       0.8904 |            0.8904 |
| Tucker+MLP | Dummy         |      0.0855 |     0.9350 |              0.1211 |         0.2946 |       0.0043 |            0.0085 |

*p(t): Nadeau-Bengio duzeltilmis eslesmis t-testi; Holm = coklu karsilastirma duzeltmesi.*

## Tablo 6. Eslesmis karsilastirmalar, macro F1 -- patient_vs_control

| A          | B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:-----------|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+MLP | Tucker+LogReg |     -0.0707 |    -0.6895 |              0.2159 |         0.7681 |       0.0042 |            0.0125 |
| Tucker+MLP | PCA+MLP       |      0.0585 |     0.4631 |              0.4018 |         0.8035 |       0.0164 |            0.0328 |
| Tucker+MLP | Voxel-MLP     |     -0.0776 |    -0.7283 |              0.1920 |         0.7681 |       0.0015 |            0.0061 |
| Tucker+MLP | Dummy         |      0.0107 |     0.0831 |              0.8796 |         0.8796 |       0.7712 |            0.7712 |

*p(t): Nadeau-Bengio duzeltilmis eslesmis t-testi; Holm = coklu karsilastirma duzeltmesi.*

## Tablo 7. Eslesmis karsilastirmalar, macro F1 -- schz_keepghost

| A          | B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:-----------|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+MLP | Tucker+LogReg |      0.0548 |     0.7097 |              0.2278 |         0.4557 |       0.0157 |            0.0313 |
| Tucker+MLP | PCA+MLP       |      0.2525 |     1.3943 |              0.0266 |         0.0798 |       0.0012 |            0.0037 |
| Tucker+MLP | Voxel-MLP     |      0.0205 |     0.2533 |              0.6596 |         0.6596 |       0.5098 |            0.5098 |
| Tucker+MLP | Dummy         |      0.1597 |     1.7687 |              0.0072 |         0.0288 |       0.0001 |            0.0005 |

*p(t): Nadeau-Bengio duzeltilmis eslesmis t-testi; Holm = coklu karsilastirma duzeltmesi.*

## Tablo 8. Eslesmis karsilastirmalar, macro F1 -- schz_vs_control

| A          | B             |   mean_diff |   cohens_d |   p_corrected_ttest |   p_ttest_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:-----------|:--------------|------------:|-----------:|--------------------:|---------------:|-------------:|------------------:|
| Tucker+MLP | Tucker+LogReg |      0.0120 |     0.1118 |              0.8454 |         1.0000 |       0.7332 |            0.7332 |
| Tucker+MLP | PCA+MLP       |      0.2217 |     1.1314 |              0.0640 |         0.2562 |       0.0037 |            0.0149 |
| Tucker+MLP | Voxel-MLP     |      0.0297 |     0.3321 |              0.5645 |         1.0000 |       0.2209 |            0.4418 |
| Tucker+MLP | Dummy         |      0.1351 |     0.9868 |              0.1014 |         0.3041 |       0.0063 |            0.0189 |

*p(t): Nadeau-Bengio duzeltilmis eslesmis t-testi; Holm = coklu karsilastirma duzeltmesi.*

## Tablo 9. Karistiricilardan arindirma etkisi -- 4class. Guven araliklari fold bagimliligina gore duzeltilmistir.

| method              | dengeli dogruluk [%95 GA]   | macro F1      |   AUC |   p (sans) |
|:--------------------|:----------------------------|:--------------|------:|-----------:|
| Tucker+MLP          | 0.396 [0.305, 0.486]        | 0.349 ± 0.069 | 0.652 |      0.002 |
| Tucker(arinmis)+MLP | 0.346 [0.281, 0.412]        | 0.299 ± 0.047 | 0.614 |      0.004 |
| Demo+LogReg         | 0.356 [0.263, 0.449]        | 0.325 ± 0.067 | 0.608 |      0.014 |
| Tucker+Demo+MLP     | 0.402 [0.296, 0.508]        | 0.345 ± 0.101 | 0.658 |      0.004 |
| Dummy(cogunluk)     | 0.250 [0.250, 0.250]        | 0.168 ± 0.001 | 0.500 |      1.000 |

## Tablo 10. Karistiricilardan arindirma etkisi -- patient_vs_control. Guven araliklari fold bagimliligina gore duzeltilmistir.

| method              | dengeli dogruluk [%95 GA]   | macro F1      |   AUC | p (sans)   |
|:--------------------|:----------------------------|:--------------|------:|:-----------|
| Tucker+MLP          | 0.544 [0.476, 0.613]        | 0.517 ± 0.086 | 0.561 | 0.0937     |
| Tucker(arinmis)+MLP | 0.535 [0.476, 0.595]        | 0.492 ± 0.092 | 0.531 | 0.1097     |
| Demo+LogReg         | 0.667 [0.590, 0.745]        | 0.663 ± 0.063 | 0.718 | <0.001     |
| Tucker+Demo+MLP     | 0.527 [0.439, 0.615]        | 0.505 ± 0.089 | 0.549 | 0.2604     |
| Dummy(cogunluk)     | 0.500 [0.500, 0.500]        | 0.336 ± 0.003 | 0.500 | 1.0000     |

## Tablo 11. Karistiricilardan arindirma etkisi -- schz_keepghost. Guven araliklari fold bagimliligina gore duzeltilmistir.

| method              | dengeli dogruluk [%95 GA]   | macro F1      |   AUC | p (sans)   |
|:--------------------|:----------------------------|:--------------|------:|:-----------|
| Tucker+MLP          | 0.701 [0.619, 0.782]        | 0.685 ± 0.056 | 0.766 | <0.001     |
| Tucker(arinmis)+MLP | 0.637 [0.544, 0.730]        | 0.611 ± 0.102 | 0.671 | 0.0034     |
| Demo+LogReg         | 0.693 [0.606, 0.781]        | 0.668 ± 0.065 | 0.758 | <0.001     |
| Tucker+Demo+MLP     | 0.694 [0.611, 0.777]        | 0.675 ± 0.069 | 0.764 | <0.001     |
| Dummy(cogunluk)     | 0.500 [0.500, 0.500]        | 0.417 ± 0.000 | 0.500 | 1.0000     |

## Tablo 12. Karistiricilardan arindirma etkisi -- schz_vs_control. Guven araliklari fold bagimliligina gore duzeltilmistir.

| method              | dengeli dogruluk [%95 GA]   | macro F1      |   AUC | p (sans)   |
|:--------------------|:----------------------------|:--------------|------:|:-----------|
| Tucker+MLP          | 0.656 [0.544, 0.768]        | 0.622 ± 0.077 | 0.683 | 0.0050     |
| Tucker(arinmis)+MLP | 0.588 [0.460, 0.715]        | 0.548 ± 0.088 | 0.651 | 0.0806     |
| Demo+LogReg         | 0.743 [0.591, 0.895]        | 0.690 ± 0.110 | 0.778 | 0.0020     |
| Tucker+Demo+MLP     | 0.668 [0.573, 0.762]        | 0.616 ± 0.057 | 0.722 | <0.001     |
| Dummy(cogunluk)     | 0.500 [0.500, 0.500]        | 0.440 ± 0.000 | 0.500 | 1.0000     |

## Tablo 13. Tarayici-disi genelleme (leave-one-scanner-out) -- patient_vs_control. Birincil olcut AUC; dengeli dogruluk onsel kaymasina duyarlidir.

|   test_scanner |   n_train |   n_test |   balanced_accuracy |   balanced_accuracy_prior_adj |   f1_macro |   auc |
|---------------:|----------:|---------:|--------------------:|------------------------------:|-----------:|------:|
|      35343.000 |    78.000 |  139.000 |               0.422 |                         0.500 |      0.403 | 0.407 |
|      35426.000 |   139.000 |   78.000 |               0.572 |                         0.616 |      0.460 | 0.618 |

## Tablo 14. Tarayici-disi genelleme (leave-one-scanner-out) -- schz_keepghost. Birincil olcut AUC; dengeli dogruluk onsel kaymasina duyarlidir.

|   test_scanner |   n_train |   n_test |   balanced_accuracy |   balanced_accuracy_prior_adj |   f1_macro |   auc |
|---------------:|----------:|---------:|--------------------:|------------------------------:|-----------:|------:|
|      35343.000 |    48.000 |  127.000 |               0.717 |                         0.647 |      0.665 | 0.739 |
|      35426.000 |   127.000 |   48.000 |               0.673 |                         0.704 |      0.661 | 0.758 |

## Tablo 15. Tucker ozniteliklerinin kodladigi bilgi -- 4class. Yas icin CV $R^2$, digerleri icin dengeli dogruluk.

| hedef    | olcut            |   deger |
|:---------|:-----------------|--------:|
| yas      | CV R^2           |   0.049 |
| cinsiyet | dengeli dogruluk |   0.793 |
| tarayici | dengeli dogruluk |   0.767 |
| tani     | dengeli dogruluk |   0.391 |

## Tablo 16. Tucker ozniteliklerinin kodladigi bilgi -- schz_vs_control. Yas icin CV $R^2$, digerleri icin dengeli dogruluk.

| hedef    | olcut            |   deger |
|:---------|:-----------------|--------:|
| yas      | CV R^2           |  -0.007 |
| cinsiyet | dengeli dogruluk |   0.739 |
| tarayici | dengeli dogruluk |   0.668 |
| tani     | dengeli dogruluk |   0.588 |

## Tablo 17. Tucker rank vs sikistirma kalitesi -- 4class

|    rank |   test_recon_error |   train_explained_energy |
|--------:|-------------------:|-------------------------:|
|  2.0000 |             0.5355 |                   0.1102 |
|  4.0000 |             0.5039 |                   0.2116 |
|  6.0000 |             0.4721 |                   0.3044 |
|  8.0000 |             0.4462 |                   0.3786 |
| 10.0000 |             0.4220 |                   0.4445 |
| 12.0000 |             0.3992 |                   0.5018 |
| 16.0000 |             0.3589 |                   0.5972 |

## Tablo 18. Tucker rank vs sikistirma kalitesi -- patient_vs_control

|    rank |   test_recon_error |   train_explained_energy |
|--------:|-------------------:|-------------------------:|
|  2.0000 |             0.5281 |                   0.1106 |
|  4.0000 |             0.4975 |                   0.2122 |
|  6.0000 |             0.4676 |                   0.3060 |
|  8.0000 |             0.4421 |                   0.3800 |
| 10.0000 |             0.4185 |                   0.4460 |
| 12.0000 |             0.3961 |                   0.5033 |
| 16.0000 |             0.3566 |                   0.5986 |

## Tablo 19. Tucker rank vs sikistirma kalitesi -- schz_keepghost

|    rank |   test_recon_error |   train_explained_energy |
|--------:|-------------------:|-------------------------:|
|  2.0000 |             0.5230 |                   0.1077 |
|  4.0000 |             0.4959 |                   0.2156 |
|  6.0000 |             0.4660 |                   0.3093 |
|  8.0000 |             0.4415 |                   0.3828 |
| 10.0000 |             0.4179 |                   0.4477 |
| 12.0000 |             0.3960 |                   0.5046 |
| 16.0000 |             0.3563 |                   0.5996 |

## Tablo 20. Tucker rank vs sikistirma kalitesi -- schz_vs_control

|    rank |   test_recon_error |   train_explained_energy |
|--------:|-------------------:|-------------------------:|
|  2.0000 |             0.5406 |                   0.1075 |
|  4.0000 |             0.5078 |                   0.2120 |
|  6.0000 |             0.4743 |                   0.3059 |
|  8.0000 |             0.4474 |                   0.3795 |
| 10.0000 |             0.4233 |                   0.4454 |
| 12.0000 |             0.4003 |                   0.5022 |
| 16.0000 |             0.3597 |                   0.5975 |

---

## Olgusal bulgu listesi

Asagidaki maddeler CSV ciktilarindan dogrudan uretildi. Yorum eklenmemistir -- tezdeki tartisma bolumunu bunlarin uzerine sen yazacaksin.

1. [4class] En yuksek dengeli dogruluk: **Tucker+LogReg** (0.412).
2. [patient_vs_control] En yuksek dengeli dogruluk: **Voxel-MLP** (0.602).
3. [schz_keepghost] En yuksek dengeli dogruluk: **Tucker+MLP** (0.701).
4. [schz_vs_control] En yuksek dengeli dogruluk: **Tucker+MLP** (0.656).
5. [schz_keepghost] Tucker+MLP vs Dummy: fark +0.160 macro F1, p=0.0288 (Holm) -- ANLAMLI.
6. [4class] Arindirma sonrasi dengeli dogruluk 0.346 (ham 0.396, degisim -0.049); sansa karsi p=0.0036 -> ANLAMLI.
7. [4class] Sadece demografi 0.356 vs ham goruntu 0.396 (fark +0.039).
8. [patient_vs_control] Arindirma sonrasi dengeli dogruluk 0.535 (ham 0.544, degisim -0.009); sansa karsi p=0.1097 -> anlamli degil.
9. [patient_vs_control] Sadece demografi 0.667 vs ham goruntu 0.544 (fark -0.123).
10. [schz_keepghost] Arindirma sonrasi dengeli dogruluk 0.637 (ham 0.701, degisim -0.063); sansa karsi p=0.0034 -> ANLAMLI.
11. [schz_keepghost] Sadece demografi 0.693 vs ham goruntu 0.701 (fark +0.007).
12. [schz_vs_control] Arindirma sonrasi dengeli dogruluk 0.588 (ham 0.656, degisim -0.068); sansa karsi p=0.0806 -> anlamli degil.
13. [schz_vs_control] Sadece demografi 0.743 vs ham goruntu 0.656 (fark -0.087).
14. [patient_vs_control] LOSO ortalama AUC 0.513 (sans 0.500); dengeli dogruluk 0.497, onsel-duzeltilmis 0.558.
15. [schz_keepghost] LOSO ortalama AUC 0.749 (sans 0.500); dengeli dogruluk 0.695, onsel-duzeltilmis 0.676.
16. [4class] Oznitelikler **cinsiyet** bilgisini (0.793) taniyi (0.391) kodladigindan daha iyi kodluyor.
17. [4class] En guclu demografi baseline (Demografi+LogReg, F1=0.323) vs en iyi goruntu yontemi (Tucker+LogReg, F1=0.418): fark +0.095, p=0.1285 -> anlamli degil.
18. [schz_vs_control] Oznitelikler **cinsiyet** bilgisini (0.739) taniyi (0.588) kodladigindan daha iyi kodluyor.
19. [schz_vs_control] En guclu demografi baseline (Demografi+LogReg, F1=0.685) vs en iyi goruntu yontemi (Tucker+MLP, F1=0.622): fark -0.064, p=0.2890 -> anlamli degil.
20. [4class] Rank taramasinda en iyi: rank 12 (1728 oznitelik), macro F1 0.390 ± 0.097.
21. [schz_vs_control] Rank taramasinda en iyi: rank 8 (512 oznitelik), macro F1 0.654 ± 0.106.

---

## Raporlama hatirlatmalari

- Sinif dengesizligi nedeniyle birincil olcut **dengeli dogruluk** ve **macro F1** olmali; accuracy yaniltici (cogunluk sinifi tahmini yuksek accuracy verir).
- Guven araliklari ve p-degerleri Nadeau-Bengio duzeltmelidir; tekrarlı CV'de fold'lar bagimsiz degildir.
- LOSO'da birincil olcut AUC'dir (esikten bagimsiz, onsel kaymasindan etkilenmez).
- Tarayici ile tani iliskili oldugu icin kovaryant regresyonu asiri duzeltme yapar; arindirilmis sonuclar **alt sinir**dir.
- Tensor ayrisimi her fold'da yalnizca egitim verisiyle fit edilmistir (veri sizintisi yok).
