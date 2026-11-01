"""Projenin ortak fonksiyonlari. Herkes veriyi bu dosyayla okur, boler ve degerlendirir.
Bu dosyayi yalnizca Fatih degistirir; degisiklik istegi WhatsApp grubuna yazilir."""
import glob
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

SEED = 42
VERI_YOLU = "/content/drive/MyDrive/ysa-wifi-csi/data/UT_HAR"
# UT-HAR sinif sirasi; H1 veri kontrolunden sonra dogrulanacak
SINIFLAR = ["yatma", "dusme", "yurume", "egilip_alma", "kosma", "oturma", "kalkma"]
DUSME = 1  # dusme sinifinin etiket numarasi; H1 veri kontrolunden sonra dogrulanacak


def uthar_yukle(kok=VERI_YOLU):
    """UT-HAR'i SenseFi ile ayni bicimde okur: X_* -> (N, 250, 90), y_* -> (N,).
    SenseFi'dan farkli olarak normalizasyon yalnizca EGITIM verisinden ogrenilir (sizinti onleme)."""
    v = {}
    for yol in glob.glob(f"{kok}/data/*.csv"):
        ad = yol.split("/")[-1].split(".")[0]          # X_train, X_val, X_test
        with open(yol, "rb") as f:
            v[ad] = np.load(f).reshape(-1, 250, 90).astype("float32")
    for yol in glob.glob(f"{kok}/label/*.csv"):
        ad = yol.split("/")[-1].split(".")[0]          # y_train, y_val, y_test
        with open(yol, "rb") as f:
            v[ad] = np.load(f).astype("int64")
    if "X_train" not in v:
        raise FileNotFoundError(f"UT-HAR bulunamadi: {kok}. Drive yolunu kontrol et.")
    mn, mx = v["X_train"].min(), v["X_train"].max()
    for k in [k for k in v if k.startswith("X_")]:
        v[k] = (v[k] - mn) / (mx - mn)
    return v


def ikili_etiket(y, dusme=DUSME):
    """7 sinifi dusme(1) / dusme degil(0) olarak ikiye indirir."""
    return (np.asarray(y) == dusme).astype("int64")


def degerlendir(y_gercek, y_tahmin, ikili=False):
    """Ortak metrikler. 7 sinif: accuracy + macro-F1; ikili gorevde dusme metrikleri de eklenir."""
    s = {"accuracy": accuracy_score(y_gercek, y_tahmin),
         "macro_f1": f1_score(y_gercek, y_tahmin, average="macro"),
         "karisiklik": confusion_matrix(y_gercek, y_tahmin)}
    if ikili:
        s["dusme_recall"] = recall_score(y_gercek, y_tahmin, pos_label=1)
        s["dusme_precision"] = precision_score(y_gercek, y_tahmin, pos_label=1, zero_division=0)
        s["dusme_f1"] = f1_score(y_gercek, y_tahmin, pos_label=1)
    return s
