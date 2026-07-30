# Karistirici kontrolleri -- gorev: schz_vs_control

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
degisken           test  istatistik        p
     yas Kruskal-Wallis    8.340273 0.003878
  gender        ki-kare    4.648001 0.031090
 scanner        ki-kare   21.717249 0.000003
```

## C) Demografi baseline vs goruntu

```
goruntu_yontemi demografi_yontemi  F1_goruntu  F1_demografi      fark  p_corrected_ttest  p_wilcoxon
     Tucker+MLP  Demografi+LogReg    0.621559      0.685430 -0.063871           0.289041    0.033047
     Tucker+MLP     Demografi+MLP    0.621559      0.589171  0.032388           0.734880    0.846924
  Tucker+LogReg  Demografi+LogReg    0.609540      0.685430 -0.075890           0.351351    0.084284
  Tucker+LogReg     Demografi+MLP    0.609540      0.589171  0.020369           0.791973    0.550924
      Voxel-MLP  Demografi+LogReg    0.591825      0.685430 -0.093605           0.194974    0.008362
      Voxel-MLP     Demografi+MLP    0.591825      0.589171  0.002654           0.978818    0.678772
```

## D) Ozellikler neyi kodluyor?

```
   hedef            olcut     deger
     yas           CV R^2 -0.006851
cinsiyet dengeli dogruluk  0.739309
tarayici dengeli dogruluk  0.668190
    tani dengeli dogruluk  0.587879
```
