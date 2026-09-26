#!/usr/bin/env python3
""""Eru Ilúvatar" hesabı için profil resmi üretir.

Ortada Eru'nun ışığı (Sönmez Alev), çevresinde altın bir yüzük; yüzüğün
üzerinde ateşle parlayan Tengwar yazısı (Quenya kipi):
  Eru Ilúvatar · Eä · Ainulindalë

Yazı tipleri: Alcarin Tengwar (OFL), Cinzel (OFL).

Kullanım:
  python3 profil.py --fontlar FONTLAR/ --cikti marka/
"""
import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 2048  # süper örnekleme boyutu; çıktı 1024 px

# Quenya kipi (Free Tengwar / CSUR eşlemesi)
KISA, UZUN = "", ""
A, E, I, U = "", "", "", ""
ERU = KISA + E + "" + U                                            # E-ru
ILUVATAR = KISA + I + "" + UZUN + U + "" + A + "" + A + ""   # I-l-ú-va-ta-r
EA = KISA + E + KISA + A                                                  # E-ä
AINULINDALE = "" + A + "" + U + "" + I + "" + A + "" + E  # ai-nu-li-nda-lë
AYRAC = "    "
YAZI = ERU + " " + ILUVATAR + AYRAC + EA + AYRAC + AINULINDALE + AYRAC


def fraktal(n, tohum):
    rng = np.random.default_rng(tohum)
    c = np.zeros((n, n), np.float32)
    g, top = 1.0, 0.0
    for s in (96, 48, 24, 12, 6):
        x = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), s)
        c += g * x / (x.std() + 1e-6)
        top += g
        g *= 0.55
    c -= c.min()
    return c / c.max()


def serit(font_yolu, yukseklik, cevre):
    """Yazıyı çevreyi dolduracak kadar tekrarlayan yatay şerit (maske)."""
    boyut = int(yukseklik * 0.62)
    fnt = ImageFont.truetype(str(font_yolu), boyut)
    tek = fnt.getlength(YAZI)
    tekrar = max(1, int(round(cevre / tek)))
    metin = YAZI * tekrar
    uzunluk = fnt.getlength(metin)
    tuval = Image.new("L", (int(uzunluk) + 4, yukseklik), 0)
    ImageDraw.Draw(tuval).text((0, yukseklik * 0.5), metin, font=fnt, fill=255, anchor="lm")
    x = np.asarray(tuval, np.float32) / 255.0
    return cv2.resize(x, (int(cevre), yukseklik), interpolation=cv2.INTER_AREA)


def ciz(font_klasoru, isim=False):
    f = pathlib.Path(font_klasoru)
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    X = (xs - S / 2) / (S / 2)
    Y = (ys - S / 2) / (S / 2)
    R = np.sqrt(X * X + Y * Y)
    TH = np.arctan2(X, -Y)  # tepeden saat yönünde
    img = np.zeros((S, S, 3), np.float32)

    # arka plan: gece mavisi, nebula, yıldızlar
    img += np.array([0.035, 0.035, 0.09]) * np.exp(-R ** 2 / 1.2)[..., None]
    neb = fraktal(S, 3)
    img += (np.clip((neb - 0.5) * 2.5, 0, 1) ** 2 * 0.10)[..., None] * np.array([0.35, 0.3, 0.8])
    rng = np.random.default_rng(9)
    for _ in range(420):
        x, y = rng.uniform(0, S, 2)
        r = np.hypot(x - S / 2, y - S / 2) / (S / 2)
        if 0.52 < r < 0.80:
            continue
        b = rng.uniform(0.2, 1.0) ** 2
        cv2.circle(img, (int(x), int(y)), int(rng.choice([1, 2, 2, 3])), (b, b, b * 1.1), -1, cv2.LINE_AA)

    # merkezdeki ışık: Sönmez Alev
    sicak = np.array([1.0, 0.86, 0.62])
    img += np.exp(-R ** 2 / 0.12)[..., None] * np.array([1.0, 0.62, 0.30]) * 0.16
    img += np.exp(-R ** 2 / 0.03)[..., None] * sicak * 1.2
    img += np.exp(-R ** 2 / 0.002)[..., None] * np.array([1, 1, 1]) * 6.0
    isin = np.zeros_like(R)
    for k in range(8):
        aci = k * np.pi / 4
        fark = np.angle(np.exp(1j * (TH - aci)))
        uzun = 1.0 if k % 2 == 0 else 0.55
        isin += np.exp(-(fark / (0.012 + 0.02 * R)) ** 2) * np.exp(-R / (0.28 * uzun))
    for k in range(48):
        aci = k * np.pi / 24 + 0.07
        fark = np.angle(np.exp(1j * (TH - aci)))
        isin += 0.18 * np.exp(-(fark / 0.02) ** 2) * np.exp(-R / 0.22)
    isin *= np.clip(R / 0.02, 0, 1) * (0.25 + 0.75 * np.clip((0.60 - R) / 0.08, 0, 1))
    img += (isin * 1.6)[..., None] * sicak

    # altın yüzük
    r_ic, r_dis = 0.565, 0.755
    t = (R - r_ic) / (r_dis - r_ic)
    bant = np.clip(np.minimum(t, 1 - t) * (r_dis - r_ic) * S / 2 / 1.5, 0, 1)
    tt = np.clip(t, 0, 1)
    nz = np.sin(np.pi * tt)                     # yüzeye dik bileşen
    nr = -np.cos(np.pi * tt)                    # iç kenar içe, dış kenar dışa bakar
    nx, ny = nr * X / np.maximum(R, 1e-3), nr * Y / np.maximum(R, 1e-3)
    L = np.array([-0.55, -0.65, 0.45]); L /= np.linalg.norm(L)
    lambert = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    Hh = L + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
    parlama = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) ** 60
    ic_isik = np.clip(-nr, 0, 1) ** 2 * 0.9     # merkezdeki ışığın iç kenardaki yansıması
    koyu, orta, acik = np.array([0.16, 0.07, 0.015]), np.array([0.80, 0.50, 0.14]), np.array([1.0, 0.92, 0.66])
    yansima = np.exp(-((nz - 0.55) / 0.08) ** 2) * 0.35 * (0.6 + 0.4 * np.cos(TH + 0.8))
    yuzuk = koyu + (orta - koyu) * lambert[..., None] ** 1.4 + acik * (parlama * 1.8 + ic_isik * 0.55 + yansima)[..., None]
    yuzuk += np.array([0.25, 0.12, 0.03]) * (0.5 + 0.5 * np.cos(2 * TH - 0.6))[..., None] * 0.35
    golge = np.exp(-((R - r_dis - 0.01) / 0.02) ** 2) + np.exp(-((R - r_ic + 0.01) / 0.02) ** 2)
    img *= (1 - 0.6 * golge)[..., None]
    img = img * (1 - bant[..., None]) + yuzuk * bant[..., None]

    # ateş yazısı
    r_y_dis, r_y_ic = 0.725, 0.595
    y_px = int((r_y_dis - r_y_ic) * S / 2)
    cevre = 2 * np.pi * (r_y_dis + r_y_ic) / 2 * S / 2
    seritm = serit(f / "AlcarinTengwar-Bold.ttf", y_px, cevre)
    harita_x = ((TH % (2 * np.pi)) / (2 * np.pi) * (seritm.shape[1] - 1)).astype(np.float32)
    harita_y = ((r_y_dis - R) / (r_y_dis - r_y_ic) * (y_px - 1)).astype(np.float32)
    yazi = cv2.remap(seritm, harita_x, harita_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    yazi *= ((R > r_y_ic) & (R < r_y_dis))
    parilti = cv2.GaussianBlur(yazi, (0, 0), 10) * 1.3 + cv2.GaussianBlur(yazi, (0, 0), 3) * 0.8
    img *= (1 - np.clip(cv2.GaussianBlur(yazi, (0, 0), 6) * 0.9, 0, 0.7))[..., None]
    img += parilti[..., None] * np.array([1.0, 0.30, 0.03]) * 1.3
    img = img * (1 - yazi[..., None] * 0.9) + yazi[..., None] * np.array([1.0, 0.80, 0.35]) * 1.9

    # parıltı, vinyet, ton eşleme
    img += cv2.GaussianBlur(img, (0, 0), 30) * 0.18
    img *= (1 - 0.35 * np.clip(R - 0.85, 0, 1) ** 1.5 * 2)[..., None]
    img = 1 - np.exp(-np.maximum(img, 0) * 1.25)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    if isim:
        pil = Image.fromarray(out)
        d = ImageDraw.Draw(pil)
        fnt = ImageFont.truetype(str(f / "Cinzel[wght].ttf"), 150)
        fnt.set_variation_by_axes([700])
        d.text((S / 2, S * 0.935), "ERU ILÚVATAR", font=fnt, fill=(255, 220, 150), anchor="mm",
               stroke_width=6, stroke_fill=(20, 10, 5))
        out = np.asarray(pil)
    return cv2.resize(out, (1024, 1024), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fontlar", required=True)
    ap.add_argument("--cikti", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.cikti.mkdir(parents=True, exist_ok=True)
    for isim, ad in ((False, "profil-resmi.png"), (True, "profil-resmi-isimli.png")):
        img = ciz(a.fontlar, isim)
        cv2.imwrite(str(a.cikti / ad), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(a.cikti / ad)


if __name__ == "__main__":
    main()
