# Karistirici kontrolleri -- gorev: schz_vs_control

## B) Demografik denge

```
degisken           test  istatistik        p
     yas Kruskal-Wallis   10.827414 0.001000
  gender        ki-kare    7.039460 0.007973
 scanner        ki-kare   16.363907 0.000052
```

## C) Demografi baseline vs goruntu

```
goruntu_yontemi demografi_yontemi  F1_goruntu  F1_demografi      fark  p_corrected_ttest  p_wilcoxon
     Tucker+MLP  Demografi+LogReg    0.685461      0.668129  0.017331           0.760844    0.389404
     Tucker+MLP     Demografi+MLP    0.685461      0.627143  0.058317           0.309655    0.051589
  Tucker+LogReg  Demografi+LogReg    0.630653      0.668129 -0.037476           0.557106    0.359131
  Tucker+LogReg     Demografi+MLP    0.630653      0.627143  0.003510           0.958741    0.678772
      Voxel-MLP  Demografi+LogReg    0.664959      0.668129 -0.003171           0.952993    0.700703
      Voxel-MLP     Demografi+MLP    0.664959      0.627143  0.037815           0.586884    0.221330
```

## D) Ozellikler neyi kodluyor?

```
   hedef            olcut    deger
     yas           CV R^2 0.017988
cinsiyet dengeli dogruluk 0.795978
tarayici dengeli dogruluk 0.618930
    tani dengeli dogruluk 0.636000
```
