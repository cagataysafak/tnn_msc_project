# Karistirici kontrolleri -- gorev: patient_vs_control

## A) Hayalet artefakti elemesi

```
ghost_flag  temiz  hayalet  toplam  hayalet_%
diagnosis                                    
ADHD           36        5      41       12.2
BIPOLAR        41        8      49       16.3
CONTROL       110       15     125       12.0
SCHZ           30       20      50       40.0
```

ki-kare: chi2=20.371, p=<0.001

## B) Demografik denge

```
degisken           test  istatistik            p
     yas Kruskal-Wallis    5.753184 1.645880e-02
  gender        ki-kare    1.371300 2.415886e-01
 scanner        ki-kare   29.026146 7.140797e-08
```

## C) Demografi baseline vs goruntu

```
goruntu_yontemi demografi_yontemi  F1_goruntu  F1_demografi      fark  p_corrected_ttest  p_wilcoxon
     Tucker+MLP  Demografi+LogReg    0.517163      0.661467 -0.144304           0.006245    0.000061
     Tucker+MLP     Demografi+MLP    0.517163      0.650980 -0.133816           0.086239    0.005371
  Tucker+LogReg  Demografi+LogReg    0.603409      0.661467 -0.058058           0.259451    0.023790
  Tucker+LogReg     Demografi+MLP    0.603409      0.650980 -0.047571           0.515750    0.041327
      Voxel-MLP  Demografi+LogReg    0.600498      0.661467 -0.060969           0.154323    0.010254
      Voxel-MLP     Demografi+MLP    0.600498      0.650980 -0.050481           0.420710    0.041327
```

## D) Ozellikler neyi kodluyor?

```
   hedef            olcut    deger
     yas           CV R^2 0.038939
cinsiyet dengeli dogruluk 0.797542
tarayici dengeli dogruluk 0.755442
    tani dengeli dogruluk 0.594393
```
