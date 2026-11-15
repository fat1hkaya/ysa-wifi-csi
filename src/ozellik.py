"""UT-HAR icin klasik makine ogrenmesi oznitelikleri.

Ham ornek: (250 zaman adimi, 90 kanal). 90 kanal = 3 anten x 30 alt tasiyici CSI genligi.
Klasik algoritmalar (RF, SVM, KNN, LR...) 22.500 ham degeri dogrudan iyi kullanamaz;
bu dosya her ornegi 910 sayilik bir oznitelik vektorune ozetler:

  1) 90 kanal x 10 istatistik = 900   (zaman ekseni boyunca)
  2) 5 zaman dilimi hareket yogunlugu = 5   (hareket ne zaman oldu?)
  3) 5 frekans bandi gucu = 5   (hareket hizli mi yavas mi?)

Kullanim:
    from ozellik import ozellik_cikar
    F_train, isimler = ozellik_cikar(v["X_train"])
Oznitelikler ornek-ici hesaplandigi icin sizinti yaratmaz; olcekleme (StandardScaler)
yine YALNIZCA egitim verisinden ogrenilmelidir.
"""
import numpy as np

ANTEN, ALT_TASIYICI = 3, 30
ISTATISTIKLER = ["ort", "std", "min", "maks", "aralik", "medyan",
                 "carpiklik", "basiklik", "enerji", "ort_mutlak_fark"]
N_DILIM, N_BANT = 5, 5


def kanal_adlari():
    """90 kanalin adi: a1_t01 ... a3_t30 (anten, alt tasiyici). Sira UT-HAR ile aynidir."""
    return [f"a{a + 1}_t{t + 1:02d}" for a in range(ANTEN) for t in range(ALT_TASIYICI)]


def ozellik_isimleri():
    adlar = [f"{k}__{s}" for s in ISTATISTIKLER for k in kanal_adlari()]
    adlar += [f"hareket_dilim{i + 1}" for i in range(N_DILIM)]
    adlar += [f"fft_bant{i + 1}" for i in range(N_BANT)]
    return adlar


def _parca(X):
    """(n, 250, 90) -> (n, 910)."""
    X = X.astype("float64")
    ort = X.mean(1)
    std = X.std(1)
    mn, mx = X.min(1), X.max(1)
    medyan = np.median(X, 1)
    merkez = X - ort[:, None, :]
    m2 = (merkez ** 2).mean(1)
    m3 = (merkez ** 3).mean(1)
    m4 = (merkez ** 4).mean(1)
    guvenli = np.where(m2 > 1e-12, m2, 1.0)
    carpiklik = np.where(m2 > 1e-12, m3 / guvenli ** 1.5, 0.0)
    basiklik = np.where(m2 > 1e-12, m4 / guvenli ** 2 - 3.0, 0.0)   # fazla basiklik
    enerji = (X ** 2).mean(1)
    fark = np.abs(np.diff(X, axis=1)).mean(1)
    kanal = np.concatenate([ort, std, mn, mx, mx - mn, medyan,
                            carpiklik, basiklik, enerji, fark], axis=1)   # (n, 900)

    # Zaman dilimleri: tum kanallarin ortalama ardisik farki, 5 esit parcada
    adim = np.abs(np.diff(X, axis=1)).mean(2)                            # (n, 249)
    dilim = np.stack([d.mean(1) for d in np.array_split(adim, N_DILIM, axis=1)], 1)

    # Frekans bantlari: ortalamasi cikarilmis sinyalin guc spektrumu, kanallar ortalamasi
    guc = (np.abs(np.fft.rfft(merkez, axis=1)) ** 2).mean(2)[:, 1:]      # DC haric (n, 125)
    bant = np.stack([b.sum(1) for b in np.array_split(guc, N_BANT, axis=1)], 1)
    bant = np.log1p(bant)

    return np.concatenate([kanal, dilim, bant], axis=1)


def ozellik_cikar(X, parti=256):
    """X: (N, 250, 90) -> (F: (N, 910) float32, isimler). Bellek icin parti parti calisir."""
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1:] != (250, 90):
        raise ValueError(f"Beklenen sekil (N, 250, 90), gelen {X.shape}")
    F = np.concatenate([_parca(X[i:i + parti]) for i in range(0, len(X), parti)], 0)
    return F.astype("float32"), ozellik_isimleri()


def grup_ozet(isimler, onem):
    """Oznitelik onemlerini istatistik turu ve anten bazinda toplar (yorum icin)."""
    onem = np.asarray(onem, dtype=float)
    tur, anten = {}, {}
    for ad, o in zip(isimler, onem):
        if "__" in ad:
            k, s = ad.split("__")
            tur[s] = tur.get(s, 0) + o
            anten[k[:2]] = anten.get(k[:2], 0) + o
        else:
            g = ad.rsplit("_", 1)[0] if ad.startswith("fft") else "hareket_dilim"
            tur[g] = tur.get(g, 0) + o
    return tur, anten
