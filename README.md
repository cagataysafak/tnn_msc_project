# Tensors and Neural Networks — MSc Projesi

**Veri seti:** UCLA Consortium for Neuropsychiatric Phenomics, LA5c (OpenNeuro `ds000030`, v1.0.0)
**Yöntem:** Kısmi Tucker ayrışımı (MPCA) → çok katmanlı algılayıcı (MLP) ile psikiyatrik tanı sınıflandırması
**Donanım hedefi:** CPU-only dizüstü (i7-13650HX, 16 GB RAM). CUDA **gerekmiyor**.

---

## 1. Veri setinin yapısı (kısa özet)

BIDS ("Brain Imaging Data Structure") standardında düzenlenmiş bir veri seti.
Klasörlerin anlamı:

| Yol | İçerik |
|---|---|
| `participants.tsv` | Her deneğin satırı: `participant_id`, `diagnosis`, `age`, `gender`, `ghost_NoGhost`, hangi taramaların mevcut olduğu |
| `sub-XXXXX/anat/` | **T1w yapısal MR** — tek bir 3B hacim, beynin anatomisi (~1 mm çözünürlük) |
| `sub-XXXXX/func/` | fMRI — 4B (x, y, z, **zaman**). Her görev için ayrı dosya |
| `sub-XXXXX/dwi/` | Difüzyon MR — beyaz cevher yolakları için, `.bval`/`.bvec` yön bilgisi |
| `sub-XXXXX/beh/` | Tarayıcı dışı davranışsal test kayıtları |
| `derivatives/` | Kalite kontrol çıktıları (MRIQC), grafikler — ham veri değil |
| `phenotype/` | Nöropsikolojik test skorları |

Tanı grupları: `CONTROL` (138), `SCHZ` (58), `BIPOLAR` (49), `ADHD` (45).

**Bu proje neden sadece `anat/*_T1w.nii.gz` kullanıyor?**

1. **Tensor yapısı doğal:** Her T1w dosyası tek bir 3-yollu tensor `(x, y, z)`.
   Tüm denekler üst üste konunca 4-yollu tensor `(denek, x, y, z)` — Tucker
   ayrışımı için ideal.
2. **Boyut:** Tüm veri seti onlarca GB; sadece T1w ≈ **2.5–3 GB**.
3. **Ön işleme:** fMRI kullanmak için hareket düzeltme + MNI kaydı (fMRIPrep)
   gerekir — bu, CPU'da denek başına ~1–2 saat, yani 265 denek için haftalar.
   T1w'de kayıt yazılımı olmadan da makul bir uzamsal normalizasyon yapılabilir.

> fMRI ile bağlantı (connectivity) tensörü yapmak isterseniz bunu §8'de
> "olası uzantılar" bölümünde anlattım — ama MSc projesinin tesliminden sonra.

---

## 2. Metodoloji — neden "Tucker + MLP" değil de tam olarak bu?

Senin önerdiğin Tucker + MLP fikri doğru. Ben iki noktada rafine ettim:

### 2.1 Tam Tucker yerine **kısmi Tucker (MPCA)**

Veri tensörü `X ∈ R^{N × d₁ × d₂ × d₃}` (N denek). Tam Tucker **denek modunu da**
sıkıştırır — ama o zaman yeni bir denek geldiğinde onu aynı uzaya projekte
edemezsin, çünkü denek modu faktörü sadece eğitimdeki deneklere ait.
Bu yüzden denek modu **sıkıştırılmaz**:

```
X ≈ C ×₁ U₁ ×₂ U₂ ×₃ U₃        U_k ∈ R^{d_k × r_k},  U_kᵀU_k = I

Her denek için özellik:
C_n = (X_n − X̄) ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ   ∈ R^{r₁ × r₂ × r₃}
```

Bu tam olarak **Multilinear PCA** (Lu, Plataniotis & Venetsanopoulos, 2008)
ve TensorFace/Tucker tabanlı öznitelik çıkarımının standart formu.
`U_k`'lar **HOOI** (higher-order orthogonal iteration) ile bulunuyor;
`n_iter=0` verilirse tek geçişlik **HOSVD**'ye düşüyor — ikisini
karşılaştırabilirsin.

**Sayısal kazanç:** `64³ = 262 144` voksel → `8³ = 512` özellik.
512× sıkıştırma, ~160 eğitim örneğiyle çalışabilir hâle geliyor.

### 2.2 **Veri sızıntısını (data leakage) önleme** — en kritik nokta

Literatürdeki nöro-görüntüleme çalışmalarının çok büyük bir kısmı burada
hata yapar: ayrışımı **tüm veri** üzerinde bir kere yapıp sonra CV yapar.
Bu, test deneklerinin bilgisinin faktör matrislerine sızmasına ve
performansın yapay olarak şişmesine yol açar.

Bu projede **her fold'da** `MPCA.fit()` yalnızca eğitim bölümüyle çağrılır;
test bölümü yalnızca `transform()` edilir. `StandardScaler` ve `PCA` için de
aynı kural geçerli. Bunu raporunda ayrıca vurgula — jüri sorar.

### 2.3 Karşılaştırılan yöntemler

| Yöntem | Ne test ediyor |
|---|---|
| **Tucker+MLP** | ANA YÖNTEM |
| Tucker+LogReg | Kazanç NN'den mi geliyor, yoksa özniteliklerden mi? |
| PCA+MLP | Tensor yapısını korumak, düzleştirip PCA yapmaktan iyi mi? |
| Voxel-MLP | **Ayrışımsız baseline** — havuzlanmış ham vokseller → MLP |
| Dummy | Şans seviyesi (sınıf dengesizliği yüzünden accuracy yanıltıcı) |

### 2.4 İstatistiksel karşılaştırma

- 5-kat çapraz doğrulama × 3 tekrar = 15 fold, **stratified**.
- Metrikler: accuracy, **balanced accuracy**, **macro F1**, weighted F1,
  Cohen's κ, ROC-AUC. Sınıflar dengesiz olduğu için macro F1 ve balanced
  accuracy birincil metrik.
- **Nadeau–Bengio düzeltilmiş eşleşmiş t-testi**: tekrarlı CV'de fold'lar
  bağımsız değildir (eğitim kümeleri örtüşür), düz t-testi p-değerlerini
  aşırı iyimser verir. Bu düzeltme varyansa `1/k + n_test/n_train` terimini
  ekler.
- Dağılım varsayımsız **Wilcoxon işaretli sıra testi** de raporlanır.
- Çoklu karşılaştırma için **Holm–Bonferroni** düzeltmesi.
- Etki büyüklüğü: eşleşmiş Cohen's d.

---

## 3. Kurulum

```bash
# Sanal ortam (önerilir)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**PyTorch hakkında:** Windows'ta `pip install torch` zaten CPU sürümünü kurar.
Linux'ta CPU-only istersen:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`
PyTorch hiç kurulmazsa proje otomatik olarak `sklearn.neural_network.MLPClassifier`'a
düşer ve yine çalışır. **CUDA gerekmiyor.**

---

## 4. Çalıştırma

### Adım 0a — Önce duman testi (veri indirmeden, ~2 dakika)

Pipeline'ın laptopunda hatasız çalıştığını doğrula:

```bash
python run_all.py --smoke
```

Bu, `data/synthetic/` altında ds000030 ile aynı yapıda 80 sahte denek üretir,
tüm adımları çalıştırır ve `results/` altına figür + tablo yazar.
Sonuçlar anlamsızdır (veri sentetik), **amaç sadece kodun çalıştığını görmek**.

### Adım 0b — Gerçek veriyi indir (~2.5–3 GB)

```bash
pip install openneuro-py
python step0_download_t1w.py --target-dir data/ds000030
```

Çalışmazsa alternatifler (`step0_download_t1w.py` içinde de yazılı):

```bash
# AWS CLI — hesap/kayıt gerekmez
aws s3 sync --no-sign-request s3://openneuro.org/ds000030 data/ds000030 \
    --exclude "*" --include "participants.tsv" --include "*_T1w.nii.gz"

# DataLad
datalad clone https://github.com/OpenNeuroDatasets/ds000030.git data/ds000030
cd data/ds000030 && datalad get "sub-*/anat/*_T1w.nii.gz"
```

Ya da tarayıcıdan tek tek: <https://openneuro.org/datasets/ds000030/versions/1.0.0>

### Adım 1 — Tensor veri setini kur (~5–15 dk)

```bash
python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 --n-jobs 4
```

Çıktı: `derivatives/X_64.npy` — `(N, 64, 64, 64)` float32, ~270 MB.

> `--n-jobs` işlem başına ~0.5 GB RAM ister. 16 GB'de 4–6 rahat.
> RAM sıkıntısı olursa `--size 48` kullan (hacim 2.4× küçülür).

### Adım 2 — Bütün analizleri tek komutla çalıştır

```bash
python run_all.py --bids-dir data/ds000030
```

Bu, aşağıdakilerin hepsini sırayla çalıştırır:

| | Ne yapar |
|---|---|
| ADIM 1 | Tensör veri setini kurar — iki kez: ana küme (hayaletliler elenmiş) ve duyarlılık kümesi (`--keep-ghost`, `derivatives_ghost/`) |
| ADIM 2 | Her görev için Tucker+MLP, baseline'lar, istatistik, figürler, rank taraması |
| ADIM 3 | Karıştırıcı ve seçim yanlılığı kontrolleri |
| ADIM 4 | Kovaryant regresyonu + tarayıcı-dışı genelleme |
| ADIM 5 | Hepsini tek rapora toplar (`THESIS_REPORT.md` + LaTeX tablolar) |

Varsayılan koşular: `schz_vs_control`, `patient_vs_control`, `4class` (ana küme)
ve `schz_keepghost` (duyarlılık analizi).

Yararlı seçenekler:

```bash
# Ön işleme zaten yapıldıysa atla
python run_all.py --bids-dir data/ds000030 --skip-step1

# Hızlı sürüm: tek görev, az tekrar, rank taraması ve duyarlılık koşusu yok
python run_all.py --bids-dir data/ds000030 --tasks schz_vs_control \
    --repeats 2 --no-rank-sweep --no-ghost-run

# Sadece raporu yeniden üret (analiz yapmaz, saniyeler sürer)
python run_all.py --only-report

# Ne çalıştırılacağını görmek için (hiçbir şey çalıştırmaz)
python run_all.py --dry-run

# RAM sıkışırsa
python run_all.py --bids-dir data/ds000030 --size 48 --skip-pca
```

Bir koşu hata verirse pipeline **durmaz**; sonundaki özet tabloda hangi
adımın başarısız olduğu ve ne kadar sürdüğü görünür. Başarısız adımı tek
başına tekrar çalıştırabilirsin — komut satırları ekranda basılıyor.

### Tek tek çalıştırmak istersen

```bash
python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 --n-jobs 4
python step2_run_experiments.py --task schz_vs_control --ranks 8 8 8 --repeats 3
python step3_confound_checks.py --bids-dir data/ds000030 --task schz_vs_control
python step4_deconfound.py --task schz_vs_control
python step5_make_report.py
```

Görevler: `schz_vs_control`, `patient_vs_control`, `4class`.
`--tag` ile her koşunun çıktıları ayrı dosyalara yazılır (step2/3/4'te var).

---

## 5. Çıktılar

```
derivatives/
  X_64.npy                     (N, 64, 64, 64) tensor veri seti
  meta_64.csv                  denek, tanı, yaş, cinsiyet + ön işleme QC metrikleri
results/
  report_<görev>.md            otomatik özet rapor (rapora doğrudan yapıştırılabilir)
  run_info_<görev>.json        tüm hiperparametreler (tekrar üretilebilirlik)
  tables/
    fold_metrics_*.csv         her fold × her yöntem için ham metrikler
    summary_*.csv              ortalama ± sd + %95 GA
    statistical_tests_*.csv    t-testi, Wilcoxon, Cohen's d, Holm düzeltmesi
    per_class_f1_*.csv         sınıf bazlı F1
    compression_*.csv          rank vs yeniden kurma hatası
  figures/
    fig01 demografi            sınıf/yaş/cinsiyet dağılımı
    fig02 ön işleme QC         işlenmiş hacimlerin dilim montajı
    fig03 sınıf ortalamaları   grup ortalama hacimleri + farkları
    fig04 MPCA spektrumu       mod bazlı özdeğerler, kümülatif varyans
    fig05 rank vs hata         sıkıştırma kalitesi eğrisi
    fig06 yeniden kurma        farklı ranklarda görsel karşılaştırma
    fig07 öz-hacimler          u₁∘u₂∘u₃ rank-1 bileşenleri
    fig08 gömme (embedding)    PCA + t-SNE, sınıf renkli
    fig09 eğitim eğrisi        MLP kayıp eğrileri
    fig10 karışıklık matrisi   her yöntem için
    fig11 kutu grafikleri      fold bazlı performans dağılımı
    fig12 ROC                  ikili görevlerde
    fig13 rank taraması        --rank-sweep ile
```

---

## 6. Ön işleme neyi nasıl yapıyor?

`tnn/preprocessing.py`, kayıt (registration) yazılımı **olmadan**, saf
numpy/scipy ile kaba bir uzamsal normalizasyon uygular:

1. **Kanonik RAS yönelimi** (`nibabel.as_closest_canonical`) → eksen sırası
   tüm denekler için aynı.
2. **Otsu eşiği + en büyük bağlantılı bileşen + delik doldurma** → kafa maskesi,
   arka plan gürültüsü atılır.
3. **Sınır kutusuna (bounding box) kırpma** → kafa konumu normalize edilir.
4. **mm cinsinden küpe tamamlama** → anatomik en-boy oranı korunur.
5. **`64³` gridine yeniden örnekleme** → kafa boyutu normalize edilir.
6. **Maske içi %99 persentil ile yoğunluk ölçekleme** → tarayıcı farkı azalır.

### Sınırlılıklar (raporuna mutlaka yaz)

- Bu **tam bir MNI kaydı değil**. Voksel düzeyinde denekler arası anatomik
  karşılık yaklaşıktır; ince yapısal farklar bulanıklaşır.
- Otsu maskesi kafatası ve skalpı da içerir (gerçek bir *skull-stripping*
  değil). FSL BET / ANTs ile iyileştirilebilir.
- Kafa boyutu normalize edildiği için **beyin hacmi farkı bilgisi kayboluyor** —
  bu aslında şizofrenide bilinen bir işaret. `pad_to_cube_mm` adımını atlayıp
  sabit mm'lik bir kutu kullanarak bu bilgiyi koruyabilirsin (bir ablasyon
  deneyi olarak güzel olur).
- ds000030'un ~%20'sinde T1w'de hayalet (aliasing) artefaktı var; varsayılan
  olarak `ghost_NoGhost` sütunuyla eleniyorlar (`--keep-ghost` ile dahil edilir).

---

## 7. Beklenen sonuçlar hakkında dürüst bir uyarı

Ham yapısal MR'dan psikiyatrik tanı sınıflandırması **zor bir problemdir**.
Literatürde, düzgün kayıt + voksel-tabanlı morfometri ile bile
şizofreni-vs-kontrol için tipik olarak **%65–80 doğruluk** bildirilir;
4 sınıflı problemde performans şans seviyesine çok daha yakındır.

Bu yüzden projenin bilimsel değeri "yüksek accuracy" değil, **kontrollü
karşılaştırma**dır: *tensor ayrışımı, düzleştirmeye/ayrışımsız baseline'a göre
istatistiksel olarak anlamlı bir kazanç sağlıyor mu?* Sonuç "hayır" çıksa bile
bu geçerli ve raporlanabilir bir bulgudur — yeter ki metodoloji sağlam olsun
(sızıntı yok, uygun metrik, uygun test). Kodun bunların hepsini karşılıyor.

`Dummy` baseline'ı bu yüzden var: accuracy %68 çıksa bile, eğer sınıfların
%68'i CONTROL ise model hiçbir şey öğrenmemiş olabilir. **Balanced accuracy ve
macro F1'e bak.**

---

## 8. Olası uzantılar (istersen)

1. **Rank ablasyonu** — `--rank-sweep` zaten var; `fig13`'ü rapora koy.
2. **HOSVD vs HOOI** — `tnn/config.py` içinde `MPCA_N_ITER = 0` yapıp tekrar çalıştır.
3. **CP (PARAFAC) ayrışımı** — Tucker yerine CP ile karşılaştırma.
4. **3B CNN** — MLP yerine; CPU'da yavaş ama `--size 48` ile mümkün.
5. **fMRI bağlantı tensörü** — `rest` fMRI → atlas ROI zaman serileri →
   `(denek × ROI × ROI)` korelasyon tensörü → Tucker. Kayıt için fMRIPrep
   gerekir; alternatif olarak Google Colab'da GPU ile denenebilir.
6. **Çok modlu birleştirme** — T1w + DWI özniteliklerini ayrı ayrı ayrıştırıp
   MLP'nin girişinde birleştirmek.

---

## 9. Google Colab kullanmak istersen

Gerekmiyor — proje CPU'da rahat çalışıyor. Ama denemek istersen:

```python
!pip install nibabel openneuro-py -q
!git clone <bu-projenin-repo-adresi> tnn_project   # ya da dosyaları yükle
%cd tnn_project
!python step0_download_t1w.py --target-dir data/ds000030
!python run_all.py --bids-dir data/ds000030 --size 64 --n-jobs 2
```

**Dikkat:** Colab oturumu kapanınca disk silinir; ~3 GB'lık veriyi her
seferinde yeniden indirmen gerekir. `derivatives/X_64.npy` (~270 MB) dosyasını
Google Drive'a kaydedip sonraki oturumlarda oradan yüklemek daha pratiktir.

---

## 10. Dosya haritası

```
step0_download_t1w.py        OpenNeuro'dan sadece T1w indirir
step0_make_synthetic_bids.py sahte BIDS üretir (duman testi)
step1_build_tensor_dataset.py  T1w → (N,64,64,64) tensor
step2_run_experiments.py     CV + Tucker+MLP + baseline'lar + istatistik + figürler
step3_confound_checks.py     seçim yanlılığı + demografi baseline + öznitelik içeriği
step4_deconfound.py          kovaryant regresyonu + tarayıcı-dışı genelleme
step5_make_report.py         tüm sonuçları teze hazır md + LaTeX tablolara toplar
run_all.py                   hepsini sırayla çalıştırır
tnn/
  config.py         tüm ayarlar tek yerde
  data.py           participants.tsv okuma, görev tanımları
  preprocessing.py  Otsu maske, bbox kırpma, yeniden örnekleme
  tensor_utils.py   unfold / n-mod çarpım / MPCA (HOOI) — self-test'li
  nn_utils.py       PyTorch MLP + eğitim döngüsü (sklearn yedeği ile)
  evaluation.py     metrikler + Nadeau-Bengio testi + Holm düzeltmesi
  viz_utils.py      tüm figürler
```

Tensor cebrini doğrulamak için:

```bash
python -m tnn.tensor_utils
```

(unfold/n-mod çarpım tutarlılığı, tam rankta sıfır hata, HOOI ≤ HOSVD hatası
gibi 6 test çalıştırır.)

---

## 11. Adım 3 — karıştırıcı (confound) kontrolleri

`step2` bir performans sayısı verir. `step3` o sayının **nereden geldiğini**
sorar — savunmada sorulacak dört soru:

```bash
python step3_confound_checks.py --bids-dir data/ds000030 --task schz_vs_control
python step3_confound_checks.py --bids-dir data/ds000030 --task 4class
```

| Kontrol | Soru | Neden kritik |
|---|---|---|
| **A** | Hayalet artefaktı elemesi tanıya göre yanlı mı? (ki-kare) | Yanlıysa seçim yanlılığı var; `--keep-ghost` duyarlılık analizi şart |
| **B** | Gruplar yaş / cinsiyet / tarayıcı bakımından dengeli mi? | Değilse model tanıdan çok demografiyi öğreniyor olabilir |
| **C** | **Sadece demografiyle** (yaş+cinsiyet+tarayıcı, hiç görüntü yok) aynı fold'larda ne alınır? | MR'ın demografinin üstüne ne kattığını ölçen tek dürüst test |
| **D** | Tucker öznitelikleri neyi kodluyor — yaş mı, tarayıcı mı, tanı mı? | Öznitelikler tanıyı değil nuisance değişkenleri taşıyor olabilir |

Çıktılar: `results/confound_report_<görev>.md`,
`results/tables/confound_*.csv`, `results/figures/fig14_feature_content_*.png`.

**Nasıl okunacak:** C adımında hiçbir görüntü yöntemi demografi baseline'ını
anlamlı olarak geçemiyorsa, bildirilen başarı büyük ölçüde demografik
farklardan kaynaklanıyor olabilir — bunu sınırlılık olarak yazmak zorundasın.
D adımında yaş için CV R² değeri tanı için elde edilen değerden yüksekse,
öznitelikler ağırlıkla yaşı kodluyor demektir.

---

## 12. Önerilen ek koşular

```bash
# 1. Hayalet elemesi duyarlılık analizi (örneklem ~217 -> ~265)
python step1_build_tensor_dataset.py --bids-dir data/ds000030 --size 64 \
    --n-jobs 4 --keep-ghost --out-dir derivatives_ghost
python step2_run_experiments.py --deriv-dir derivatives_ghost --size 64 \
    --task schz_vs_control --tag schz_keepghost

# 2. En dengeli kontrast (~107 hasta vs ~110 kontrol) -- en yüksek güç
python step2_run_experiments.py --size 64 --task patient_vs_control --repeats 5

# 3. Aşırı öğrenmeyi test et: daha küçük rank
python step2_run_experiments.py --size 64 --task 4class --ranks 4 4 4 \
    --tag 4class_r4

# 4. HOSVD vs HOOI ablasyonu -- tnn/config.py içinde MPCA_N_ITER = 0 yap
```

`--tag` her koşunun çıktılarını ayrı dosyalara yazar (step2, step3 ve step4'te var), böylece birbirini
ezmezler.

---

## 13. Adım 4 — karıştırıcılardan arındırılmış analiz

`step3` bir karıştırıcı bulursa (bu veri setinde buluyor), `step4` şu tek
soruyu yanıtlar: **yaş, cinsiyet ve tarayıcı etkisi özniteliklerden
çıkarıldıktan sonra beyinde tanı hakkında sinyal kalıyor mu?**

```bash
python step4_deconfound.py --size 64 --task schz_vs_control
python step4_deconfound.py --size 64 --task 4class
```

Üç analiz yapar:

1. **Kovaryant regresyonu (residualization).** Her öznitelik için
   `f = b₀ + b₁·yaş + b₂·cinsiyet + b₃·tarayıcı + e` modeli **sadece eğitim
   verisiyle** kurulur, sınıflandırmada artık `e` kullanılır.
2. **Karşılaştırmalı yöntem seti** (aynı fold'lar, eşleşmiş testler):
   ham Tucker · arındırılmış Tucker · sadece demografi ·
   görüntü+demografi birleşik · çoğunluk sınıfı dummy.
   Her yöntem için dengeli doğruluğun şans seviyesinden anlamlı yüksek olup
   olmadığı tek-örneklemli düzeltilmiş t-testiyle sınanır.
3. **Tarayıcı-dışı genelleme (leave-one-scanner-out).** Bir tarayıcıda eğit,
   diğerinde test et. Model anatomi öğrendiyse tarayıcı değişince de
   çalışmalı; site etkisi öğrendiyse çöker.

Çıktı: `results/deconfound_report_<görev>.md`,
`results/tables/deconf_*.csv`, `results/figures/fig15_deconfound_*.png`.


### Raporlama notu — güven aralıkları

`step4`'ün verdiği %95 güven aralıkları **Nadeau–Bengio düzeltmelidir**, yani
p-değerleriyle aynı varsayımları paylaşır. Düz t-aralığı (`sd/√k`) fold'ların
bağımsız olduğunu varsayar; tekrarlı CV'de eğitim kümeleri örtüştüğü için bu
aralık gerçekte olduğundan **dar** çıkar ve düzeltilmiş p-değeriyle çelişebilir
(aralık şansı dışlarken p anlamsız kalabilir). Düzeltilmemiş dar aralıklar
karşılaştırma için CSV'de `*_uncorrected` sütunlarında saklanır — **tezde
düzeltilmiş olanı kullan.**

### LOSO'da birincil ölçüt AUC

Tarayıcıların sınıf dağılımları çok farklı olduğu için `argmax` kararı eğitim
önselini taşır ve dengeli doğruluk çökebilirken AUC sağlam kalır (önsel kayması
/ prior shift). `step4` bu yüzden hem ham hem **önsel-düzeltilmiş** dengeli
doğruluğu (posterior eğitim önseline bölünür — test etiketleri kullanılmaz)
hem AUC'yi raporlar ve yorumunu **AUC'ye** dayandırır.

### Kavramsal uyarı — bunu rapora yaz

Tarayıcı ile tanı bu veri setinde güçlü ilişkili (χ²=21.7, p<0.001).
İlişkili bir değişkeni regresyonla çıkarmak tanı sinyalinin bir kısmını da
siler (**over-correction**). Dolayısıyla arındırılmış sonuç bir **alt sınır**dır:

- Arındırmadan sonra hâlâ şanstan anlamlı yüksekse → site ile açıklanamayan
  bir yapısal sinyal var.
- Şanstan ayırt edilemiyorsa → *"bu örneklemde doğrusal karıştırıcı etkileri
  çıkarıldığında ölçülebilir bir yapısal tanı sinyali kalmıyor"*. Bu geçerli
  bir bulgudur; yöntemin kötü olduğu anlamına gelmez.

`step4`'ün doğru çalıştığı iki sentetik senaryoyla doğrulandı: gerçek sinyal
varken arındırma sonrası sinyali koruyor (p<0.001), sinyal yalnızca site
etkisinden geliyorken arındırma sonrası şans seviyesine düşüyor (p=0.12).

---

## 14. Adım 5 — teze hazır birleşik rapor

step1-4 sonuçları 15+ CSV'ye dağılır. `step5` hepsini tarayıp tek bir rapora
toplar:

```bash
python step5_make_report.py --size 64
```

Çıktılar:

| Dosya | İçerik |
|---|---|
| `results/THESIS_REPORT.md` | Numaralanmış markdown tablolar + olgusal bulgu listesi |
| `results/latex/*.tex` | Aynı tabloların booktabs sürümü (`\input{}` ile teze eklenir) |

Etiketleri (`--tag`) otomatik bulur, yani ek koşular yaptıysan onlar da rapora
girer. Eksik dosyalar sessizce atlanır — kısmi sonuçlarla da çalışır.

LaTeX tabloları `\usepackage{booktabs}` gerektirir ve pdfLaTeX uyumludur
(Unicode karakterler `$\pm$`, `$<$` gibi matematik moduna çevrilir).

**Bu script yeni analiz yapmaz ve yorum üretmez.** Yalnızca mevcut sayıları
toplar ve p-değerlerine göre koşullu olgusal cümleler kurar ("… p=0.0806 →
anlamlı değil"). Tartışma bölümünü sen yazacaksın; bunlar hazır malzeme.

---

## 15. Kaynaklar

- Poldrack et al. (2016). *A phenome-wide examination of neural and cognitive
  function.* Scientific Data 3:160110. — ds000030'un veri makalesi, **atıf ver**.
- Tucker, L. R. (1966). *Some mathematical notes on three-mode factor analysis.*
- De Lathauwer, De Moor & Vandewalle (2000). *A multilinear SVD* / *Best rank-(R₁,R₂,R₃)
  approximation* — HOSVD ve HOOI.
- Lu, Plataniotis & Venetsanopoulos (2008). *MPCA: Multilinear PCA of tensor objects.*
  IEEE TNN 19(1):18–39.
- Kolda & Bader (2009). *Tensor decompositions and applications.* SIAM Review 51(3).
- Nadeau & Bengio (2003). *Inference for the generalization error.* ML 52:239–281.
- Bouckaert & Frank (2004). *Evaluating the replicability of significance tests.*