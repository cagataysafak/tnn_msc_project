# Karistirici kontrolleri -- gorev: 4class

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
     yas Kruskal-Wallis    9.503544 2.329369e-02
  gender        ki-kare    5.779961 1.228204e-01
 scanner        ki-kare   32.091610 5.005895e-07
```

## C) Demografi baseline vs goruntu

```
goruntu_yontemi demografi_yontemi  F1_goruntu  F1_demografi     fark  p_corrected_ttest  p_wilcoxon
     Tucker+MLP  Demografi+LogReg    0.348739      0.322533 0.026207           0.680815    0.454285
     Tucker+MLP     Demografi+MLP    0.348739      0.246705 0.102035           0.084587    0.000122
  Tucker+LogReg  Demografi+LogReg    0.417918      0.322533 0.095385           0.128505    0.003357
  Tucker+LogReg     Demografi+MLP    0.417918      0.246705 0.171213           0.001401    0.000061
      Voxel-MLP  Demografi+LogReg    0.344087      0.322533 0.021554           0.716071    0.421204
      Voxel-MLP     Demografi+MLP    0.344087      0.246705 0.097382           0.066747    0.001526
```

## D) Ozellikler neyi kodluyor?

```
   hedef            olcut    deger
     yas           CV R^2 0.048633
cinsiyet dengeli dogruluk 0.792591
tarayici dengeli dogruluk 0.767478
    tani dengeli dogruluk 0.391211
```
