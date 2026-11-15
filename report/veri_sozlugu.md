# UT-HAR Veri Sözlüğü ve Öznitelikler

## 1. Veri seti özeti
| Özellik | Değer |
|---|---|
| Ad | UT-HAR (Yousefi vd., IEEE Communications Magazine, 2017); SenseFi (Patterns, 2023) işlenmiş sürümü |
| Donanım | Intel 5300 Wi-Fi kartı, 5 GHz, 1 verici – 3 alıcı anten |
| Veri türü | Sayısal: CSI **genlik** değerleri (faz yok) |
| Örnek sayısı | 4973 → eğitim 3977 · doğrulama 496 · test 500 |
| Örnek boyutu | 250 zaman adımı (paket) × 90 kanal = 22.500 sayı |
| Eksik değer | Yok |
| Dosya biçimi | Uzantısı `.csv` ama içerik NumPy ikili dosyası → `np.load` ile okunur (`src/ortak.py`) |

## 2. Ham öznitelikler: 90 kanal
- 90 kanal = **3 alıcı anten × 30 alt taşıyıcı (subcarrier)**.
- Adlandırma: `a{anten}_t{alt taşıyıcı}` → `a1_t01` … `a3_t30` (`ozellik.kanal_adlari()`).
- Her değer, o pakette o frekansta Wi-Fi sinyalinin alıcıya ne kadar güçlü ulaştığıdır. İnsan hareket edince yansımalar değişir ve genlik dalgalanır; modelin öğrendiği şey bu dalgalanma desenidir.
- **Zaman ekseni (250):** hareketin seyri. **Alt taşıyıcı ekseni (30):** farklı frekanslar harekete farklı tepki verir. **Anten ekseni (3):** farklı konumlardan bakış (3 farklı “görüş açısı”).
- Komşu alt taşıyıcılar yüksek korelasyonludur → 90 kanalda tekrar eden bilgi var (H2 analizi, bölüm 6).

## 3. Etiketler
| Kod | Sınıf | Açıklama |
|---|---|---|
| 0 | yatma (lie down) | yere/yatağa uzanma |
| 1 | **düşme (fall)** | ani düşme — ana hedef sınıf, ~%9 |
| 2 | yürüme (walk) | sürekli, ritmik hareket |
| 3 | eğilip alma (pick up) | eğilip yerden nesne alma |
| 4 | koşma (run) | hızlı, ritmik hareket |
| 5 | oturma (sit down) | ayaktan oturmaya geçiş |
| 6 | kalkma (stand up) | oturmadan ayağa kalkma |

İkili görev: `düşme = 1`, diğer her şey `0` (`ortak.ikili_etiket`).

## 4. Klasik ML için türetilmiş öznitelikler (910 adet, `src/ozellik.py`)
Klasik algoritmalar 22.500 ham sayıyla iyi çalışmaz (boyut laneti, zaman kaymasına duyarlılık). Her örnek zaman ekseni boyunca özetlenir:

| Grup | Adet | Formül (her kanal için, 250 adım üzerinde) | Ne anlatır / neye yarar |
|---|---|---|---|
| `ort` | 90 | ortalama | Kanalın genel seviyesi → kişinin konumu ve duruşu |
| `medyan` | 90 | ortanca | `ort` ile aynı ama ani sıçramalardan etkilenmez |
| `std` | 90 | standart sapma | Hareketin **büyüklüğü**; hareketsiz sınıflarda düşük |
| `min`, `maks` | 180 | en küçük / en büyük | Sinyalin uç değerleri |
| `aralik` | 90 | maks − min | Toplam dalgalanma genişliği |
| `enerji` | 90 | ortalama(x²) | Toplam sinyal gücü |
| `ort_mutlak_fark` | 90 | ortalama \|x(t+1) − x(t)\| | Hareketin **hızı** (koşma > yürüme > oturma) |
| `carpiklik` | 90 | 3. moment | Değişim tek yönlü mü (oturma/kalkma gibi geçişler) |
| `basiklik` | 90 | 4. moment − 3 | Kısa, sert, **tek seferlik** sıçramalar → **düşme** |
| `hareket_dilim1-5` | 5 | 250 adım 5 parçaya bölünür; her parçada tüm kanalların ort. mutlak farkı | Hareket **ne zaman** oldu: sürekli mi (yürüme) yoksa tek patlama mı (düşme) |
| `fft_bant1-5` | 5 | Fourier güç spektrumu 5 frekans bandına bölünür (log) | Hareket **yavaş mı, hızlı/tekrarlı mı** (adım ritmi) |

Toplam: 90 × 10 + 5 + 5 = **910**. Ad biçimi: `a2_t17__std` = anten 2, alt taşıyıcı 17'nin standart sapması.

## 5. Sızıntı kuralları
- Öznitelikler örnek içinde hesaplanır, başka örneklerin bilgisini kullanmaz.
- Min-max normalizasyon ve `StandardScaler` **yalnızca eğitim kümesinden** öğrenilir.
- Hiperparametre **doğrulama** kümesinde seçilir; **test** kümesine yalnızca en sonda bakılır.

## 6. Klasik ML algoritma dağılımı (H2)
| Kişi | Algoritma | `ALGORITMA` |
|---|---|---|
| Fatih Kaya | Random Forest | `rf` |
| Taha Yasir Güzeldal | SVM (RBF çekirdek) | `svm` |
| Yunus Emir Gemici | K-En Yakın Komşu | `knn` |
| Mümün Can Yıldız | Lojistik Regresyon | `lr` |
| Mehmet Selim Özdemir | Gradient Boosting (HistGradientBoosting) | `gb` |
| Muhammet Emin Zora | Karar Ağacı | `dt` |

Herkes `notebooks/H2_ML_sablon.ipynb` dosyasını kullanır; sonuçlar Drive'daki `results/sonuclar.csv` dosyasına eklenir.
