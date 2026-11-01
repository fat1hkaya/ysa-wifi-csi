# Wi-Fi CSI ile Kamerasız İnsan Aktivitesi Tanıma ve Düşme Algılama

Marmara Üniversitesi · Yapay Sinir Ağları dersi grup projesi (2026)

## Amaç
Wi-Fi Kanal Durum Bilgisi (CSI) verisinden derin sinir ağlarıyla 7 aktiviteyi sınıflandırmak ve düşmeyi yüksek duyarlılıkla (recall) yakalamak.

## Veri
- **UT-HAR** (ana veri): Intel 5300 NIC, 7 sınıf, örnek boyutu 250 paket × 90 (3 anten × 30 alt taşıyıcı).
  Kaynak: [SenseFi deposu](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
- **WiFall** (opsiyonel): ESP32-S3, 10 kişi. Kaynak: [Hugging Face](https://huggingface.co/datasets/RS2002/WiFall)

Veri depoya **eklenmez**. Herkes Google Drive'da aynı yolu kullanır:
`MyDrive/ysa-wifi-csi/data/UT_HAR/` (içinde `data/` ve `label/` klasörleri)

## Klasör yapısı
| Klasör | İçerik |
|---|---|
| `src/` | Ortak kod (`ortak.py`). Yalnızca Fatih değiştirir. |
| `notebooks/` | Herkesin deney notebook'ları. Ad: `<hafta>_<ekip>_<is>.ipynb` (örn. `H3_B_bilstm.ipynb`) |
| `results/` | `sonuclar.csv` ve grafikler |
| `report/` | Rapor taslakları, veri sözlüğü |
| `literatur/` | Makale notları (`<ad>.md`) |

## Başlangıç (Colab)
1. `notebooks/H1_A_veri_kontrol.ipynb` dosyasını Colab'da aç.
2. Hücreleri sırayla çalıştır. Son hücre veri şekillerini ve sınıf dağılımını yazdırmalı.

## Ortak sabitler
`SEED = 42` (nihai sonuçlar 42, 0, 123 ile) · Adam, lr = 1e-3 · batch 64 · en fazla 100 epoch, erken durdurma (sabır 15)
Ana metrik: 7 sınıfta **macro-F1**, düşme görevinde **düşme recall'u**.

## Ekip
Fatih Kaya · Taha Yasir Güzeldal · Yunus Emir Gemici · Mümün Can Yıldız · Mehmet Selim Özdemir · Muhammet Emin Zora
