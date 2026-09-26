#!/usr/bin/env python3
"""Bölüm videosunun görüntüsünü üretir ve sesle birleştirir.

Görseller tamamen koddan (prosedürel) üretilir: ışık, parçacıklar, şeritler,
siluetler. Üzerine sahne başlıkları, zamanlanmış Türkçe altyazılar ve kapanış
kartı eklenir. Kareler 4 süreçte paralel işlenir ve ffmpeg ile kodlanır.

Kullanım:
  python3 goruntu.py bolumler/bolum01.json --klasor build/ --fontlar fonts/ --cikti bolum01.mp4
  python3 goruntu.py ... --kare 15.5 --kare 60   # yalnızca önizleme PNG'leri
"""
import argparse
import json
import math
import pathlib
import re
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
w, h = W // 2, H // 2
FPS = 30
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ALTIN = np.array([1.0, 0.68, 0.26], np.float32)
GUMUS = np.array([0.70, 0.83, 1.0], np.float32)
SICAK = np.array([1.0, 0.88, 0.70], np.float32)
KIZIL = np.array([1.0, 0.09, 0.03], np.float32)
KOR = np.array([1.0, 0.34, 0.07], np.float32)
PALET = [np.array(c, np.float32) for c in [
    (1.0, 0.72, 0.30), (0.70, 0.83, 1.0), (0.40, 0.62, 1.0), (0.78, 0.52, 1.0),
    (1.0, 0.55, 0.70), (0.38, 0.95, 0.85), (1.0, 0.86, 0.48)]]
MERKEZ = (0.0, -0.05)


def puruzsuz(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def pencere(t, bas, bit, gecis=0.5):
    """bas–bit arasında 1, kenarlarda yumuşak geçişli ağırlık."""
    return float(puruzsuz((t - bas) / gecis) * puruzsuz((bit - t) / gecis))


def hece(metin):
    return sum(1 for c in metin if c in "aeıioöuüâîûëéáíóúAEIİOÖUÜÂÎÛËÉÁÍÓÚ")


# ------------------------------------------------------------------ zamanlama

class Zaman:
    def __init__(self, bolum, zaman):
        self.bolum = bolum
        self.sure = zaman["sure"]
        self.birim = zaman["birimler"]
        self.sahne = zaman["sahneler"]
        self.sinir = [0.0]
        for i in range(1, len(self.sahne)):
            if bolum["sahneler"][i]["gorsel"] == "akor":
                self.sinir.append(self.sahne[i - 1]["bit"] + 0.35)
            else:
                self.sinir.append((self.sahne[i - 1]["bit"] + self.sahne[i]["bas"]) / 2)
        self.sinir.append(self.sure)
        self.akor = next((self.sinir[i] for i, s in enumerate(bolum["sahneler"]) if s["gorsel"] == "akor"), None)
        self.kapanis = self.birim[-1]["bit"] + 0.15

    def c(self, sahne, cumle=0):
        return next(b["bas"] for b in self.birim if b["sahne"] == sahne and b["cumle"] == cumle)

    def e(self, sahne, cumle=0):
        return [b["bit"] for b in self.birim if b["sahne"] == sahne and b["cumle"] == cumle][-1]

    def agirlik(self, i, t, gecis=0.45):
        bas, bit = self.sinir[i], self.sinir[i + 1]
        g1 = 0.02 if self.bolum["sahneler"][i]["gorsel"] == "akor" else gecis
        g2 = 0.02 if i + 1 < len(self.sahne) and self.bolum["sahneler"][i + 1]["gorsel"] == "akor" else gecis
        if i == 0:
            bas -= 1
        if i == len(self.sahne) - 1:
            bit += 1
        return float(puruzsuz((t - bas + g1) / (2 * g1)) * puruzsuz((bit - t + g2) / (2 * g2)))


# ------------------------------------------------------------------ yazılar

class Yazici:
    def __init__(self, font_klasoru):
        f = pathlib.Path(font_klasoru)
        self.dosya = {"cinzel": f / "Cinzel[wght].ttf", "mont": f / "Montserrat[wght].ttf",
                      "cormorant": f / "CormorantGaramond[wght].ttf"}
        self._onbellek = {}

    def font(self, ad, boyut, kalinlik):
        anahtar = (ad, boyut, kalinlik)
        if anahtar not in self._onbellek:
            fnt = ImageFont.truetype(str(self.dosya[ad]), boyut)
            fnt.set_variation_by_axes([kalinlik])
            self._onbellek[anahtar] = fnt
        return self._onbellek[anahtar]

    @staticmethod
    def aralikli_genislik(fnt, metin, aralik):
        return sum(fnt.getlength(ch) for ch in metin) + aralik * (len(metin) - 1)

    def aralikli_yaz(self, ciz, x, y, metin, fnt, aralik, dolgu):
        for ch in metin:
            ciz.text((x, y), ch, font=fnt, fill=dolgu)
            x += fnt.getlength(ch) + aralik

    def sigdir(self, ad, metin, boyut, kalinlik, aralik, en):
        while boyut > 20:
            fnt = self.font(ad, boyut, kalinlik)
            if self.aralikli_genislik(fnt, metin, aralik) <= en:
                return fnt
            boyut -= 2
        return self.font(ad, boyut, kalinlik)

    def baslik(self, satirlar, en=880):
        """[(metin, stil)] → premultiplied RGBA (float32). stil: 'buyuk' | 'alt' | 'etiket'."""
        tuval = Image.new("L", (W, 900), 0)
        ciz = ImageDraw.Draw(tuval)
        y = 40
        cizgi_y = None
        for metin, stil in satirlar:
            if not metin:
                continue
            if stil == "buyuk":
                fnt = self.sigdir("cinzel", metin, 96, 700, 6, en)
            elif stil == "alt":
                if cizgi_y is None:
                    cizgi_y = y + 6
                    y += 26
                fnt = self.sigdir("cormorant", metin, 60, 600, 2, en)
            else:
                fnt = self.sigdir("cinzel", metin, 34, 600, 8, en)
            aralik = {"buyuk": 6, "alt": 2, "etiket": 8}[stil]
            genislik = self.aralikli_genislik(fnt, metin, aralik)
            self.aralikli_yaz(ciz, (W - genislik) / 2, y, metin, fnt, aralik, 255)
            bb = fnt.getbbox("ŞİÜĞ")
            y += bb[3] + (18 if stil == "buyuk" else 12)
        if cizgi_y is not None:
            for x in range(W // 2 - 150, W // 2 + 150):
                a = 1 - abs(x - W // 2) / 150
                ciz.line([(x, cizgi_y), (x, cizgi_y + 1)], fill=int(200 * a ** 1.5))
        maske = np.asarray(tuval, np.float32)[: y + 30] / 255.0
        return self._altin_boya(maske)

    @staticmethod
    def _altin_boya(maske):
        hh = maske.shape[0]
        grad = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
        ust = np.array([1.0, 0.93, 0.70], np.float32)
        alt = np.array([0.93, 0.66, 0.30], np.float32)
        renk = ust * (1 - grad) + alt * grad
        parilti = cv2.GaussianBlur(maske, (0, 0), 14) * 0.85
        golge = cv2.GaussianBlur(np.roll(maske, 5, axis=0), (0, 0), 6)
        rgba = np.zeros(maske.shape + (4,), np.float32)
        hale = cv2.GaussianBlur(cv2.dilate(maske, np.ones((9, 9), np.uint8)), (0, 0), 26)
        a_golge = np.clip(np.maximum(golge * 0.85, hale * 0.75), 0, 1)
        rgba[..., 3] = a_golge
        a_par = np.clip(parilti * 0.55, 0, 1)
        par_renk = np.array([1.0, 0.55, 0.18], np.float32)
        rgba[..., :3] = rgba[..., :3] * (1 - a_par[..., None]) + par_renk * a_par[..., None]
        rgba[..., 3] = rgba[..., 3] * (1 - a_par) + a_par
        rgba[..., :3] = rgba[..., :3] * (1 - maske[..., None]) + renk * maske[..., None]
        rgba[..., 3] = rgba[..., 3] * (1 - maske) + maske
        return rgba

    def altyazi(self, metin, vurgu, en=820):
        fnt = self.font("mont", 60, 800)
        kelimeler = metin.split()
        satirlar, satir = [], []
        for k in kelimeler:
            deneme = " ".join(satir + [k])
            if satir and fnt.getlength(deneme) > en:
                satirlar.append(satir)
                satir = [k]
            else:
                satir.append(k)
        satirlar.append(satir)
        if len(satirlar) == 2 and len(satirlar[1]) == 1 and len(satirlar[0]) > 2:
            satirlar = [satirlar[0][:-1], [satirlar[0][-1]] + satirlar[1]]
        satir_y = 78
        tuval = Image.new("RGBA", (W, satir_y * len(satirlar) + 40), (0, 0, 0, 0))
        ciz = ImageDraw.Draw(tuval)
        bosluk = fnt.getlength(" ")
        for i, s in enumerate(satirlar):
            genislik = fnt.getlength(" ".join(s))
            x = (W - genislik) / 2
            for k in s:
                sade = k.strip(".,;:!?\"“”…")
                renk = (255, 210, 110, 255) if sade in vurgu else (255, 255, 255, 255)
                ciz.text((x, 12 + i * satir_y), k, font=fnt, fill=renk, stroke_width=6, stroke_fill=(0, 0, 0, 255))
                x += fnt.getlength(k) + bosluk
        dizi = np.asarray(tuval, np.float32) / 255.0
        alfa = dizi[..., 3]
        golge = cv2.GaussianBlur(np.roll(alfa, 4, axis=0), (0, 0), 7) * 0.6
        rgba = np.zeros_like(dizi)
        rgba[..., 3] = golge
        rgba[..., :3] = dizi[..., :3] * alfa[..., None]
        rgba[..., 3] = alfa + golge * (1 - alfa)
        return rgba


def bindir(kare, rgba, y_merkez, alfa=1.0, olcek=1.0, x_merkez=W / 2):
    if alfa <= 0.003:
        return
    if abs(olcek - 1) > 1e-3:
        rgba = cv2.resize(rgba, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_LINEAR)
    hh, ww = rgba.shape[:2]
    y0 = int(round(y_merkez - hh / 2))
    x0 = int(round(x_merkez - ww / 2))
    ky0, ky1 = max(0, y0), min(H, y0 + hh)
    kx0, kx1 = max(0, x0), min(W, x0 + ww)
    if ky1 <= ky0 or kx1 <= kx0:
        return
    kaynak = rgba[ky0 - y0: ky1 - y0, kx0 - x0: kx1 - x0]
    bolge = kare[ky0:ky1, kx0:kx1].astype(np.float32)
    a = kaynak[..., 3:4] * alfa
    bolge = bolge * (1 - a) + kaynak[..., :3] * 255.0 * alfa
    kare[ky0:ky1, kx0:kx1] = np.clip(bolge, 0, 255).astype(np.uint8)


def altyazi_parcalari(birimler):
    """Her seslendirme birimini kısa altyazı parçalarına böler ve hece oranıyla zamanlar."""
    parcalar = []
    for b in birimler:
        kelimeler = b["metin"].split()
        gruplar, grup = [], []
        for k in kelimeler:
            grup.append(k)
            uzunluk = len(" ".join(grup))
            if uzunluk >= 26 or (re.search(r"[,;:]$", k) and uzunluk >= 12):
                gruplar.append(grup)
                grup = []
        if grup:
            if gruplar and len(" ".join(grup)) < 9:
                gruplar[-1] += grup
            else:
                gruplar.append(grup)
        agirlik = [hece(" ".join(g)) + 1.5 for g in gruplar]
        toplam = sum(agirlik)
        t = b["bas"]
        for g, a in zip(gruplar, agirlik):
            d = (b["bit"] - b["bas"]) * a / toplam
            parcalar.append({"metin": " ".join(g), "bas": t, "bit": t + d})
            t += d
    for i in range(len(parcalar) - 1):
        bosluk = parcalar[i + 1]["bas"] - parcalar[i]["bit"]
        parcalar[i]["bit"] += min(bosluk, 0.25)
    return parcalar


# ------------------------------------------------------------------ görsel araçlar

def fraktal(n, tohum):
    rng = np.random.default_rng(tohum)
    cikti = np.zeros((n, n), np.float32)
    genlik, toplam = 1.0, 0.0
    for sigma in (48, 24, 12, 6, 3):
        g = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), sigma)
        cikti += genlik * g / (g.std() + 1e-6)
        toplam += genlik
        genlik *= 0.55
    cikti -= cikti.min()
    return cikti / cikti.max()


class Baglam:
    def __init__(self, zaman, klasor):
        self.z = zaman
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self.U = (xs - w / 2) / h
        self.V = (ys - h / 2) / h
        du, dv = self.U - MERKEZ[0], self.V - MERKEZ[1]
        self.R = np.sqrt(du * du + dv * dv)
        self.TH = np.arctan2(dv, du)
        self.vinyet = (1 - 0.55 * np.clip(np.sqrt(self.U ** 2 + (self.V * 0.8) ** 2) / 0.62, 0, 1) ** 2)[..., None]
        doku = klasor / "doku.npy"
        if not doku.exists():
            np.save(doku, np.stack([fraktal(1400, 1), fraktal(1400, 2)]))
        self.doku = np.load(doku)
        ses, sr = sf.read(klasor / "ses.wav", dtype="float32")
        pencere_n = sr // FPS
        n = len(ses) // pencere_n
        rms = np.sqrt((ses[: n * pencere_n].reshape(n, pencere_n) ** 2).mean(1))
        rms = rms / (np.percentile(rms, 95) + 1e-6)
        yumusak = np.zeros_like(rms)
        for i in range(1, n):
            k = 0.5 if rms[i] > yumusak[i - 1] else 0.12
            yumusak[i] = yumusak[i - 1] + k * (rms[i] - yumusak[i - 1])
        self.zarf = np.clip(yumusak, 0, 1.3)
        rng = np.random.default_rng(11)
        self.p = {
            "u": rng.uniform(-0.32, 0.32, 700).astype(np.float32),
            "v": rng.uniform(-0.52, 0.52, 700).astype(np.float32),
            "hiz": rng.uniform(0.3, 1.0, 700).astype(np.float32),
            "faz": rng.uniform(0, 2 * np.pi, 700).astype(np.float32),
            "boy": rng.choice([1, 1, 1, 2, 2, 3], 700),
            "a": rng.uniform(0.35, 1.0, 700).astype(np.float32),
        }
        self.figurler = self._figur_dizilimi()

    def ses(self, t):
        i = int(t * FPS)
        return float(self.zarf[min(max(i, 0), len(self.zarf) - 1)])

    def nebula(self, taban, t, renk, guc):
        if guc <= 0:
            return
        for k, (hx, hy, olcek) in enumerate(((3.0, 1.2, 1.0), (-2.2, 1.8, 0.7))):
            M = np.float32([[olcek, 0, 200 + hx * t], [0, olcek, 200 + hy * t]])
            n = cv2.warpAffine(self.doku[k], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)
            n = np.clip((n - 0.45) * 2.2, 0, 1) ** 2
            taban += (n[..., None] * renk * guc * (0.6 if k else 1.0)).astype(np.float32)

    def _figur_dizilimi(self):
        rng = np.random.default_rng(5)
        figurler = []
        for sayi, v, olcek in ((9, -0.10, 0.55), (7, -0.015, 0.75), (5, 0.075, 1.0)):
            yayilim = 0.25 if sayi == 9 else (0.22 if sayi == 7 else 0.18)
            for i in range(sayi):
                u = -yayilim + 2 * yayilim * i / (sayi - 1)
                figurler.append({"u": u, "v": v + 0.01 * math.cos(u * 9), "olcek": olcek,
                                 "renk": PALET[rng.integers(len(PALET))], "faz": rng.uniform(0, 6.28)})
        sira = rng.permutation(len(figurler))
        for k, i in enumerate(sira):
            figurler[i]["sira"] = k
        return figurler


def lekele(taban, u, v, su, sv, renk, guc):
    """Ayrılabilir Gauss lekesi (yalnızca etki alanında hesaplanır)."""
    if guc <= 0.002:
        return
    cx, cy = w / 2 + u * h, h / 2 + v * h
    sx, sy = max(su * h, 0.6), max(sv * h, 0.6)
    x0, x1 = int(max(0, cx - 4 * sx)), int(min(w, cx + 4 * sx + 1))
    y0, y1 = int(max(0, cy - 4 * sy)), int(min(h, cy + 4 * sy + 1))
    if x1 <= x0 or y1 <= y0:
        return
    gx = np.exp(-0.5 * ((np.arange(x0, x1, dtype=np.float32) - cx) / sx) ** 2)
    gy = np.exp(-0.5 * ((np.arange(y0, y1, dtype=np.float32) - cy) / sy) ** 2)
    taban[y0:y1, x0:x1] += (gy[:, None, None] * gx[None, :, None]) * (renk * guc)


def halka(taban, B, cu, cv, r, gen, renk, guc):
    if guc <= 0.002:
        return
    R = r + 4 * gen
    x0, x1 = int(max(0, w / 2 + (cu - R) * h)), int(min(w, w / 2 + (cu + R) * h + 1))
    y0, y1 = int(max(0, h / 2 + (cv - R) * h)), int(min(h, h / 2 + (cv + R) * h + 1))
    if x1 <= x0 or y1 <= y0:
        return
    uu = B.U[y0:y1, x0:x1] - cu
    vv = B.V[y0:y1, x0:x1] - cv
    d = np.sqrt(uu * uu + vv * vv)
    taban[y0:y1, x0:x1] += np.exp(-(((d - r) / gen) ** 2))[..., None] * (renk * guc)


def isinlar(taban, B, t, renk, guc, yaricap=0.25):
    if guc <= 0.002:
        return
    th = B.TH
    ray = 0.6 * (0.5 + 0.5 * np.cos(9 * th + 0.12 * t)) ** 6 + 0.4 * (0.5 + 0.5 * np.cos(14 * th - 0.09 * t + 1.3)) ** 8
    f = ray * np.exp(-B.R / yaricap) * np.clip(B.R / 0.03, 0, 1)
    taban += f[..., None] * (renk * guc)


def pikselle(u, v):
    return np.stack([W / 2 + u * H, H / 2 + v * H], -1)


def parcaciklar(ayrinti, B, t, kip, renk, guc, adet=500, t0=0.0):
    if guc <= 0.01:
        return
    p = B.p
    u, v, hiz, faz = p["u"][:adet], p["v"][:adet], p["hiz"][:adet], p["faz"][:adet]
    tl = t - t0
    parla = p["a"][:adet] * (0.55 + 0.45 * np.sin(2.2 * hiz * t + faz))
    if kip == "suzul":
        uu = u + 0.006 * np.sin(0.25 * t + faz)
        vv = (v - 0.008 * hiz * t + 0.52) % 1.04 - 0.52
    elif kip == "disa":
        r = (0.03 + 0.06 * hiz * tl + faz / 6.28 * 0.55) % 0.62
        uu = MERKEZ[0] + r * np.cos(faz * 7)
        vv = MERKEZ[1] + r * np.sin(faz * 7)
        parla = parla * np.clip(1 - r / 0.62, 0, 1) * np.clip(r / 0.05, 0, 1)
    elif kip == "kor":
        vv = (v - 0.05 * hiz * t + 0.52) % 1.04 - 0.52
        uu = u * 0.6 + 0.012 * np.sin(1.7 * t + faz)
        parla = parla * np.clip((0.35 - vv) / 0.5, 0, 1)
    elif kip == "dus":
        uu = u + 0.004 * np.sin(0.5 * t + faz)
        vv = (v + 0.012 * hiz * tl + 0.52) % 1.04 - 0.52
    else:  # sarmal
        r = 0.5 * ((faz / 6.28 - 0.05 * hiz * tl) % 1.0) + 0.02
        aci = faz * 5 + 1.8 * np.log(r + 0.01) * -1 + 0.3 * tl
        uu = MERKEZ[0] + r * np.cos(aci)
        vv = MERKEZ[1] + r * np.sin(aci) * 0.85
        parla = parla * np.clip(r / 0.08, 0, 1)
    xy = pikselle(uu, vv)
    renk255 = renk * 255 * guc
    for (x, y), b, s in zip(xy, parla, p["boy"][:adet]):
        if 0 <= x < W and 0 <= y < H and b > 0.05:
            cv2.circle(ayrinti, (int(x * 16), int(y * 16)), int(s), tuple(float(c) for c in renk255 * b), -1, cv2.LINE_AA, 4)


def serit_v(i, u, t, derin=0.0, bozulma=0.0, faz_kaymasi=0.0):
    rng = np.random.default_rng(100 + i)
    k1, k2 = rng.uniform(1.6, 3.6), rng.uniform(3.0, 6.0)
    w1, w2 = rng.uniform(0.25, 0.6), rng.uniform(-0.5, -0.2)
    f1, f2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    a1 = rng.uniform(0.035, 0.07) * (1 + 0.8 * derin)
    a2 = rng.uniform(0.008, 0.02) * (1 - 0.7 * derin)
    merkez = -0.06 + 0.011 * (i - 6)
    k1 = k1 * (1 - 0.45 * derin)
    v = merkez + a1 * np.sin(2 * np.pi * k1 * u + w1 * t + f1 + faz_kaymasi) + a2 * np.sin(2 * np.pi * k2 * u + w2 * t + f2)
    if bozulma:
        rs = np.random.default_rng(int(t * 15) * 31 + i)
        v = v + bozulma * 0.02 * rs.standard_normal(len(u))
    return v


def seritler(taban, ayrinti, B, t, guc, adet=12, acilis=1.0, derin=0.0, bozulma=0.0, altin_oran=0.5, kalin=1.0):
    if guc <= 0.01:
        return
    maske = [np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)]
    u = np.linspace(-0.33, 0.33, 90, dtype=np.float32)
    for i in range(adet):
        gorunur = np.abs(u) <= 0.02 + 0.34 * acilis
        if gorunur.sum() < 12:
            continue
        v = serit_v(i, u, t, derin, bozulma)
        uu, vv = u[gorunur], v[gorunur]
        grup = 0 if (i % 2 == 0) == (altin_oran >= 0.5) or (altin_oran >= 0.99) else 1
        yarim = np.stack([w / 2 + uu * h, h / 2 + vv * h], -1)
        cv2.polylines(maske[grup], [np.int32(yarim * 16)], False, 255, int(3 * kalin), cv2.LINE_AA, 4)
        renk = ALTIN if grup == 0 else GUMUS
        tam = pikselle(uu, vv)
        cv2.polylines(ayrinti, [np.int32(tam * 16)], False, tuple(float(c) for c in (renk * 0.5 + 0.5) * 235 * guc),
                      2, cv2.LINE_AA, 4)
    for grup, renk in ((0, ALTIN), (1, GUMUS)):
        m = maske[grup].astype(np.float32) / 255.0
        parilti = cv2.GaussianBlur(m, (0, 0), 5) * 1.4 + cv2.GaussianBlur(m, (0, 0), 16) * 1.2
        taban += parilti[..., None] * (renk * guc)


def zikzak(taban, ayrinti, B, t, guc, adet=5, tekrar=False, uyum=0.0, tohum=0):
    if guc <= 0.01:
        return
    maske = np.zeros((h, w), np.uint8)
    u = np.linspace(-0.34, 0.34, 70, dtype=np.float32)
    kare_no = int(t * 15)
    for i in range(adet):
        rs = np.random.default_rng(tohum * 1000 + i * 97 + (0 if tekrar else kare_no))
        merkez = -0.06 + 0.035 * (i - adet / 2) + 0.01 * math.sin(t * 2 + i)
        if tekrar:
            desen = np.array([0, 0.05, -0.04, 0.03, -0.05, 0.02, 0], np.float32)
            faz = (u * 10 + t * 2.8) % 1.0
            v_z = merkez + np.interp(faz, np.linspace(0, 1, 7), desen) * (0.9 + 0.3 * math.sin(t * 17))
        else:
            v_z = merkez + np.cumsum(rs.standard_normal(len(u)).astype(np.float32)) * 0.012
            v_z -= (v_z.mean() - merkez)
        if uyum > 0:
            v_z = v_z * (1 - uyum) + serit_v(i + 3, u, t) * uyum
        yarim = np.stack([w / 2 + u * h, h / 2 + v_z * h], -1)
        cv2.polylines(maske, [np.int32(yarim * 16)], False, 255, 3, cv2.LINE_AA, 4)
        renk = KIZIL * (1 - uyum) + ALTIN * uyum
        cekirdek = np.clip((renk * 0.6 + np.array([0.4, 0.3, 0.2])) * 255 * guc, 0, 255)
        cv2.polylines(ayrinti, [np.int32(pikselle(u, v_z) * 16)], False, tuple(float(c) for c in cekirdek), 2, cv2.LINE_AA, 4)
    m = maske.astype(np.float32) / 255.0
    parilti = cv2.GaussianBlur(m, (0, 0), 4) * 1.6 + cv2.GaussianBlur(m, (0, 0), 14) * 1.0
    renk = KIZIL * (1 - uyum) + ALTIN * uyum
    taban += parilti[..., None] * (renk * guc)


_SAG = [(0, -0.07), (0.03, 0.01), (0.06, -0.05), (0.09, 0.02), (0.13, -0.03), (0.12, 0.05), (0.125, 0.09),
        (0.10, 0.14), (0.09, 0.16), (0.22, 0.17), (0.30, 0.13), (0.33, 0.19), (0.40, 0.22), (0.43, 0.30),
        (0.44, 0.42), (0.47, 0.60), (0.52, 0.80), (0.58, 0.97), (0.46, 0.93), (0.40, 1.0), (0.28, 0.94),
        (0.18, 1.0), (0.05, 0.95)]
SILUET = np.array(_SAG + [(-x, y) for x, y in reversed(_SAG[1:])], np.float32)


def melkor(taban, ayrinti, B, t, guc, cu, ust, olcek=1.0, kor_guc=1.0):
    if guc <= 0.01:
        return
    lekele(taban, cu, ust + 0.2 * olcek, 0.16 * olcek, 0.20 * olcek, np.array([0.55, 0.05, 0.03], np.float32), 0.9 * guc)
    nokta = np.stack([w / 2 + (cu + SILUET[:, 0] * 0.17 * olcek) * h,
                      h / 2 + (ust + SILUET[:, 1] * 0.42 * olcek) * h], -1)
    maske = np.zeros((h, w), np.uint8)
    cv2.fillPoly(maske, [np.int32(nokta * 16)], 255, cv2.LINE_AA, 4)
    m = cv2.GaussianBlur(maske.astype(np.float32) / 255.0, (0, 0), 0.8)
    kenar = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
    kenar = cv2.GaussianBlur(kenar, (0, 0), 2.5) * 2.2
    M = np.float32([[1, 0, 300], [0, 1, 300 + 20 * t]])
    catlak = cv2.warpAffine(B.doku[1], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    catlak = 1 - np.abs(catlak * 2 - 1)
    catlak = np.clip((catlak - 0.975) * 40, 0, 1) * cv2.erode(m, np.ones((7, 7), np.uint8)) * (0.7 + 0.3 * math.sin(t * 9))
    taban *= (1 - m * guc)[..., None]
    taban += (kenar * guc * kor_guc)[..., None] * KOR
    taban += (cv2.GaussianBlur(catlak, (0, 0), 1.0) * 0.7 * guc * kor_guc)[..., None] * KIZIL
    for gx in (-0.0078, 0.0078):
        lekele(taban, cu + gx * olcek, ust + 0.036 * olcek, 0.0022 * olcek, 0.0014 * olcek, KOR, 7 * guc * kor_guc)


def kure(taban, B, t, guc, r):
    if guc <= 0.01:
        return
    cu, cv = MERKEZ
    x0, x1 = int(w / 2 + (cu - r * 1.4) * h), int(w / 2 + (cu + r * 1.4) * h)
    y0, y1 = int(h / 2 + (cv - r * 1.4) * h), int(h / 2 + (cv + r * 1.4) * h)
    uu = (B.U[y0:y1, x0:x1] - cu) / r
    vv = (B.V[y0:y1, x0:x1] - cv) / r
    d2 = uu * uu + vv * vv
    ic = d2 < 1
    zz = np.sqrt(np.clip(1 - d2, 0, 1))
    isik = np.clip(-0.55 * uu - 0.45 * vv + 0.7 * zz, 0, 1)
    kosinus = np.sqrt(np.clip(1 - vv * vv, 1e-4, 1))
    boylam = (np.arcsin(np.clip(uu / kosinus, -1, 1)) / np.pi + 0.5 + 0.02 * t) % 1.0
    enlem = np.arcsin(np.clip(vv, -1, 1)) / np.pi + 0.5
    doku = B.doku[0][(enlem * 1399).astype(int), (boylam * 1399).astype(int)]
    kara = np.clip((doku - 0.52) * 10, 0, 1)
    yuzey = (np.array([0.10, 0.22, 0.55]) * (1 - kara[..., None]) + np.array([0.75, 0.62, 0.30]) * kara[..., None])
    renk = yuzey * (0.15 + 1.1 * isik[..., None]) * ic[..., None]
    kenar = np.exp(-((np.sqrt(d2) - 1) / 0.06) ** 2) * (0.6 + 0.8 * np.clip(-uu - vv, 0, 1))
    renk = renk + kenar[..., None] * np.array([0.55, 0.75, 1.0])
    renk = renk + (np.clip(1 - d2 / 1.9, 0, 1) ** 2 * 0.25)[..., None] * ALTIN
    taban[y0:y1, x0:x1] += (renk * guc).astype(np.float32)


# ------------------------------------------------------------------ sahneler

def s_kanca(taban, ayrinti, B, t, a):
    z = B.z
    e = B.ses(t)
    cu, cv = MERKEZ
    lekele(taban, cu, cv, 0.005, 0.005, SICAK, 5.0 * a)
    lekele(taban, cu, cv, 0.035, 0.035, ALTIN, (0.35 + 0.5 * e) * a)
    lekele(taban, cu, cv, 0.30, 0.0035, SICAK, 0.35 * a)
    for k in range(12):
        t_cikis = 0.3 + k * 0.95
        yas = t - t_cikis
        if 0 < yas < 5:
            kirmizi = t_cikis > z.c(0, 1) - 0.5
            renk = KIZIL if kirmizi else ALTIN
            jit = 0.004 * math.sin(t * 40) if kirmizi else 0
            halka(taban, B, cu, cv, 0.02 + 0.075 * yas + jit, 0.004 + 0.004 * yas, renk,
                  0.6 * math.exp(-yas / 1.6) * a * (1.4 if kirmizi else 1))
    parcaciklar(ayrinti, B, t, "suzul", SICAK, 0.5 * a, 250)


def s_eru(taban, ayrinti, B, t, a):
    z = B.z
    cu, cv = MERKEZ
    dogus = puruzsuz((t - z.c(1, 2) + 0.3) / 2.5)
    e = B.ses(t)
    lekele(taban, cu, cv, 0.005, 0.005, SICAK, 4.0 * a)
    lekele(taban, cu, cv, 0.03 + 0.10 * dogus, 0.03 + 0.10 * dogus, SICAK, (0.3 + 0.75 * dogus + 0.2 * e) * a)
    lekele(taban, cu, cv, 0.07 + 0.05 * dogus, 0.07 + 0.05 * dogus, ALTIN, (0.2 + 0.45 * dogus) * a)
    isinlar(taban, B, t, SICAK, 0.75 * dogus * a, 0.10 + 0.12 * dogus)
    parcaciklar(ayrinti, B, t, "disa", SICAK, (0.2 + 0.8 * dogus) * a, 450, t0=z.c(1, 2))


def s_ainur(taban, ayrinti, B, t, a, sonuk=1.0):
    z = B.z
    kaynak = (0.0, -0.17)
    lekele(taban, kaynak[0], kaynak[1], 0.10, 0.10, SICAK, 0.35 * a * sonuk)
    lekele(taban, kaynak[0], kaynak[1], 0.02, 0.02, SICAK, 1.2 * a * sonuk)
    isinlar(taban, B, t, SICAK, 0.25 * a * sonuk, 0.25)
    bas = z.c(2, 0) + 0.3
    tek = z.c(2, 2) - 0.2
    grup = tek + 1.4
    sarki = z.c(2, 1)
    for i, f in enumerate(B.figurler):
        t_dogus = bas + f["sira"] * 0.14
        ilerleme = puruzsuz((t - t_dogus) / 1.4)
        if ilerleme <= 0:
            continue
        u = kaynak[0] + (f["u"] - kaynak[0]) * ilerleme + 0.002 * math.sin(0.7 * t + f["faz"])
        v = kaynak[1] + (f["v"] - kaynak[1]) * ilerleme
        s = f["olcek"] * (0.4 + 0.6 * ilerleme)
        parlak = 0.55 * ilerleme
        if t > sarki:
            parlak += 0.25 * max(0.0, math.sin(1.3 * t + f["faz"])) * puruzsuz(t - sarki)
        solo = f["sira"] % 21
        artis = 0.0
        if solo < 4:
            artis += 1.6 * math.exp(-((t - tek - solo * 0.35) / 0.2) ** 2)
        g = f["sira"] % 3
        artis += 1.3 * math.exp(-((t - grup - (f["sira"] // 3 % 4) * 0.45) / 0.22) ** 2) * (g == 0)
        parlak = (parlak + artis) * sonuk
        r = f["renk"]
        lekele(taban, u, v - 0.055 * s, 0.009 * s, 0.009 * s, r, 1.3 * parlak * a)
        lekele(taban, u, v - 0.055 * s, 0.0035 * s, 0.0035 * s, SICAK, 2.0 * parlak * a)
        lekele(taban, u, v, 0.012 * s, 0.048 * s, r, 0.9 * parlak * a)
        lekele(taban, u, v - 0.02 * s, 0.045 * s, 0.09 * s, r, 0.12 * parlak * a)
        if artis > 0.2:
            halka(taban, B, u, v - 0.055 * s, 0.01 + 0.05 * (1 - min(1, artis / 1.6)), 0.004, r, 0.4 * artis * a)
    parcaciklar(ayrinti, B, t, "suzul", SICAK, 0.4 * a * sonuk, 300)


def s_muzik(taban, ayrinti, B, t, a):
    z = B.z
    bas = z.c(3, 0) - 0.4
    zirve = z.c(3, 2)
    acilis = puruzsuz((t - bas) / 4.0)
    adet = int(4 + 8 * puruzsuz((t - bas) / 6.0))
    kabarma = 1 + 0.6 * math.exp(-((t - zirve - 1.2) / 1.4) ** 2)
    s_ainur(taban, ayrinti, B, t, a * 0.35)
    seritler(taban, ayrinti, B, t, 0.75 * a * kabarma * min(1.0, 0.1 + 1.2 * acilis), adet=adet, acilis=acilis)
    lekele(taban, 0, -0.06, 0.22, 0.10, ALTIN, 0.18 * a * kabarma)
    parcaciklar(ayrinti, B, t, "suzul", ALTIN, 0.6 * a, 400)


def s_melkor(taban, ayrinti, B, t, a):
    z = B.z
    bas, bit = z.sinir[4], z.sinir[5]
    ilerleme = (t - bas) / (bit - bas)
    seritler(taban, ayrinti, B, t, 0.35 * a * (1 - 0.7 * puruzsuz(ilerleme * 4)) + 0.25 * a * puruzsuz((t - z.c(4, 5)) / 2),
             adet=10, acilis=1.0)
    cu = 0.07 - 0.10 * ilerleme
    ust = -0.20 + 0.004 * math.sin(t * 1.3)
    alev_bas, alev_bit = z.c(4, 2) + 2.0, z.c(4, 3) + 0.4
    alev = pencere(t, alev_bas, alev_bit, 0.8)
    if alev > 0:
        titreme = 0.8 + 0.2 * math.sin(t * 23) * math.sin(t * 7)
        lekele(taban, 0.17, -0.31, 0.006, 0.012, SICAK, 3.0 * alev * titreme * a)
        lekele(taban, 0.17, -0.31, 0.03, 0.04, np.array([0.8, 0.9, 1.0], np.float32), 0.5 * alev * a)
    iluvatar = pencere(t, z.c(4, 4) - 0.2, z.e(4, 4) + 0.6, 0.6)
    lekele(taban, 0, -0.42, 0.18, 0.08, SICAK, 0.5 * iluvatar * a)
    melkor(taban, ayrinti, B, t, a, cu, ust)
    zik = puruzsuz((t - z.c(4, 5) - 0.8) / 2.0)
    zikzak(taban, ayrinti, B, t, 0.7 * zik * a, adet=2)
    parcaciklar(ayrinti, B, t, "kor", KOR, 0.9 * a, 350)


def s_uyumsuzluk(taban, ayrinti, B, t, a):
    z = B.z
    bozuk = puruzsuz((t - z.c(5, 0)) / 0.4)
    yeni = pencere(t, z.c(5, 2), z.c(5, 3), 0.4)
    bastir = puruzsuz((t - z.c(5, 3)) / 0.3)
    ucuncu = puruzsuz((t - z.c(5, 4)) / 1.0)
    gurultulu = pencere(t, z.c(5, 5), z.c(5, 6) + 0.3, 0.4)
    derin = pencere(t, z.c(5, 6), z.c(5, 7) + 0.3, 0.4)
    birles = puruzsuz((t - z.c(5, 7) - 0.8) / 3.2)
    doruk = puruzsuz((t - (z.sinir[6] - 1.4)) / 1.4)

    kaos = bozuk * (1 - yeni) * (1 - ucuncu) * (0.8 + 0.2 * bastir)
    seritler(taban, ayrinti, B, t, (0.55 + 0.5 * yeni + 0.3 * derin) * a * (1 + doruk),
             adet=12, derin=ucuncu, bozulma=1.4 * kaos, altin_oran=0.5 + 0.5 * ucuncu)
    if yeni > 0:
        lekele(taban, 0, -0.40, 0.25, 0.12, SICAK, 0.9 * yeni * a)
        isinlar(taban, B, t, SICAK, 0.5 * yeni * a, 0.3)
    zikzak(taban, ayrinti, B, t, (kaos * 1.0 + 0.6 * bastir * (1 - ucuncu)) * a, adet=5, tohum=1)
    if ucuncu > 0:
        kirmizi = ucuncu * (0.8 + 0.6 * gurultulu - 0.4 * derin) * (1 - 0.5 * birles)
        vurus = 0.75 + 0.25 * math.cos(2 * math.pi * t / 0.36)
        zikzak(taban, ayrinti, B, t, kirmizi * vurus * a * (1 + doruk), adet=4, tekrar=True, uyum=birles, tohum=2)
    lekele(taban, 0, -0.06, 0.25, 0.14, ALTIN, (0.1 + 0.25 * derin + 0.8 * doruk) * a)
    parcaciklar(ayrinti, B, t, "suzul", ALTIN, 0.4 * a, 300)
    parcaciklar(ayrinti, B, t, "kor", KOR, 0.5 * kaos * a, 200)


def s_akor(taban, ayrinti, B, t, a):
    z = B.z
    tl = t - z.akor
    dalga = 0.02 + 1.1 * (1 - math.exp(-tl / 0.5))
    halka(taban, B, MERKEZ[0], MERKEZ[1], dalga, 0.01 + 0.04 * tl, SICAK, 3.0 * math.exp(-tl / 0.9) * a)
    halka(taban, B, MERKEZ[0], MERKEZ[1], dalga * 0.8, 0.03, ALTIN, 1.2 * math.exp(-tl / 0.9) * a)
    donma = math.exp(-tl / 2.5)
    seritler(taban, ayrinti, B, z.akor, 0.8 * donma * a, adet=12, derin=1.0, altin_oran=1.0, kalin=0.7)
    geri = puruzsuz(tl / 3.0)
    melkor(taban, ayrinti, B, t, a * (1 - puruzsuz((tl - 1.5) / 2.5)), 0.08 + 0.12 * geri, -0.02 + 0.2 * geri,
           olcek=1.0 - 0.45 * geri, kor_guc=1 - geri)
    sakin = puruzsuz((tl - 1.0) / 3.0)
    konusma = pencere(t, z.c(6, 2), z.sinir[7], 1.0)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.14, 0.14, SICAK, (0.35 + 0.35 * konusma + 0.2 * B.ses(t)) * sakin * a)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.012, 0.012, SICAK, 2.5 * sakin * a)
    isinlar(taban, B, t, SICAK, (0.25 + 0.3 * konusma) * sakin * a, 0.28)
    parcaciklar(ayrinti, B, t, "dus", ALTIN, 0.7 * sakin * a, 450, t0=z.akor)


def s_dunya(taban, ayrinti, B, t, a):
    z = B.z
    bas = z.sinir[7]
    tl = t - bas
    olusum = puruzsuz((tl - 1.0) / 6.0)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.28, 0.22, ALTIN, 0.35 * a * (1 - 0.4 * olusum))
    parcaciklar(ayrinti, B, t, "sarmal", ALTIN, 0.9 * a, 650, t0=bas)
    kure(taban, B, t, 0.85 * olusum * a, 0.085 + 0.02 * olusum + 0.004 * tl)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.012, 0.012, SICAK, 2.0 * (1 - olusum) * a)


SAHNELER = {"kanca": s_kanca, "eru": s_eru, "ainur": s_ainur, "muzik": s_muzik, "melkor": s_melkor,
            "uyumsuzluk": s_uyumsuzluk, "akor": s_akor, "dunya": s_dunya}
NEBULA = {"kanca": ((0.10, 0.08, 0.20), 0.05), "eru": ((0.35, 0.22, 0.10), 0.07), "ainur": ((0.14, 0.10, 0.28), 0.08),
          "muzik": ((0.22, 0.15, 0.32), 0.09), "melkor": ((0.40, 0.05, 0.03), 0.10), "uyumsuzluk": ((0.35, 0.10, 0.12), 0.10),
          "akor": ((0.22, 0.18, 0.14), 0.07), "dunya": ((0.10, 0.15, 0.35), 0.10)}


# ------------------------------------------------------------------ kare

class Cizer:
    def __init__(self, bolum, zaman, klasor, font_klasoru):
        self.bolum = bolum
        self.z = Zaman(bolum, zaman)
        self.B = Baglam(self.z, klasor)
        yz = Yazici(font_klasoru)
        vurgu = set(bolum.get("vurgu", []))
        self.basliklar = []
        for i, s in enumerate(bolum["sahneler"]):
            if not any(s["ekran"]):
                continue
            if s["gorsel"] == "kanca":
                img = yz.baslik([(s["ekran"][0], "buyuk"), (s["ekran"][1], "buyuk")])
                self.basliklar.append((0.25, self.z.e(0, 0) + 0.3, img))
            else:
                t_on = self.z.c(i, s.get("ekran_cumle", 0)) - 0.15
                if s["gorsel"] == "akor":
                    t_on = self.z.akor + 0.5
                img = yz.baslik([(s["ekran"][0], "buyuk"), (s["ekran"][1], "alt")])
                self.basliklar.append((t_on, t_on + 3.8, img))
        self.etiket = yz.baslik([(f"TOLKIEN MİTOLOJİSİ  ·  BÖLÜM {bolum['bolum']}", "etiket")])
        sonraki = bolum.get("sonraki", ["", "", ""])
        self.son_kart = yz.baslik([(sonraki[0], "etiket"), (sonraki[1], "buyuk"), (sonraki[2], "alt")])
        self.altyazilar = [dict(p, img=yz.altyazi(p["metin"], vurgu)) for p in altyazi_parcalari(self.z.birim)]
        self.gorseller = [s["gorsel"] for s in bolum["sahneler"]]
        kapak = bolum.get("kapak", [bolum["baslik"].upper()])
        self.kapak_yazi = yz.baslik([(k, "buyuk") for k in kapak])

    def kare(self, t, yazilar=True):
        z, B = self.z, self.B
        taban = np.zeros((h, w, 3), np.float32)
        ayrinti = np.zeros((H, W, 3), np.uint8)
        agirliklar = [(i, z.agirlik(i, t)) for i in range(len(self.gorseller))]
        for i, a in agirliklar:
            if a > 0.001:
                renk, guc = NEBULA[self.gorseller[i]]
                B.nebula(taban, t, np.array(renk, np.float32), guc * a * 3)
        for i, a in agirliklar:
            if a > 0.001:
                SAHNELER[self.gorseller[i]](taban, ayrinti, B, t, a)
        parilti = cv2.GaussianBlur(taban, (0, 0), 10)
        taban = (taban + 0.25 * parilti) * B.vinyet
        taban = 1 - np.exp(-np.maximum(taban, 0) * 1.1)
        kare = cv2.resize((taban * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
        kare = cv2.add(kare, ayrinti)

        if "uyumsuzluk" in self.gorseller:
            i = self.gorseller.index("uyumsuzluk")
            titresim = z.agirlik(i, t) * (pencere(t, z.c(i, 0), z.c(i, 0) + 1.2, 0.2) + pencere(t, z.c(i, 3), z.c(i, 3) + 1.0, 0.2)
                                          + 0.35 * pencere(t, z.c(i, 0), z.c(i, 2), 0.3))
            if titresim > 0.02:
                rs = np.random.default_rng(int(t * FPS))
                dx, dy = rs.uniform(-10, 10) * titresim, rs.uniform(-10, 10) * titresim
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                kare = cv2.warpAffine(kare, M, (W, H), borderMode=cv2.BORDER_REFLECT)
                k = int(6 * titresim)
                if k:
                    kare[:, k:, 2] = kare[:, :-k, 2]
                    kare[:, :-k, 0] = kare[:, k:, 0]
        if z.akor is not None and t >= z.akor:
            flas = math.exp(-(t - z.akor) / 0.35)
            if flas > 0.01:
                kare = cv2.addWeighted(kare, 1 - flas, np.full_like(kare, 255), flas, 0)

        if not yazilar:
            return kare
        for t_on, t_off, img in self.basliklar:
            a = pencere(t, t_on, t_off, 0.5)
            if a > 0:
                yukselis = 14 * (1 - puruzsuz((t - t_on) / 0.8))
                bindir(kare, img, 0.215 * H + yukselis, a)
        bindir(kare, self.etiket, 0.125 * H, pencere(t, 0.25, z.e(0, 0) + 0.3, 0.5) * 0.85)
        son = pencere(t, z.kapanis, z.sure + 1, 0.6)
        if son > 0:
            bindir(kare, self.etiket, 0.125 * H, son * 0.85)
            bindir(kare, self.son_kart, 0.705 * H + 10 * (1 - son), son)
        for p in self.altyazilar:
            if p["bas"] <= t < p["bit"]:
                yas = t - p["bas"]
                bindir(kare, p["img"], 0.695 * H, min(1.0, yas / 0.08), 0.92 + 0.08 * puruzsuz(yas / 0.12))
        return kare


# ------------------------------------------------------------------ işleme

_CIZER = None


def _baslat(args):
    global _CIZER
    _CIZER = Cizer(*args)


def _parca_isle(is_):
    bas, bit, yol = is_
    kodlayici = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-g", "60", yol],
        stdin=subprocess.PIPE)
    for k in range(bas, bit):
        kodlayici.stdin.write(_CIZER.kare(k / FPS).tobytes())
    kodlayici.stdin.close()
    kodlayici.wait()
    return yol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bolum", type=pathlib.Path)
    ap.add_argument("--klasor", type=pathlib.Path, required=True, help="ses.wav, muzik.wav, zaman.json klasörü")
    ap.add_argument("--fontlar", type=pathlib.Path, required=True)
    ap.add_argument("--cikti", type=pathlib.Path)
    ap.add_argument("--kare", type=float, action="append", help="yalnızca bu saniyelerin PNG önizlemesi")
    ap.add_argument("--surec", type=int, default=4)
    ap.add_argument("--kapak", type=float, help="bu saniyedeki görüntüden kapak.png üret")
    a = ap.parse_args()
    bolum = json.loads(a.bolum.read_text(encoding="utf-8"))
    zaman = json.loads((a.klasor / "zaman.json").read_text(encoding="utf-8"))
    ayarlar = (bolum, zaman, a.klasor, a.fontlar)

    if a.kapak is not None:
        c = Cizer(*ayarlar)
        kare = c.kare(a.kapak, yazilar=False)
        bindir(kare, c.etiket, 0.125 * H, 0.85)
        bindir(kare, c.kapak_yazi, 0.25 * H)
        yol = a.klasor / "kapak.png"
        cv2.imwrite(str(yol), cv2.cvtColor(kare, cv2.COLOR_RGB2BGR))
        print(yol)
        return

    if a.kare:
        c = Cizer(*ayarlar)
        for t in a.kare:
            yol = a.klasor / f"onizleme_{t:06.2f}.png"
            cv2.imwrite(str(yol), cv2.cvtColor(c.kare(t), cv2.COLOR_RGB2BGR))
            print(yol)
        return

    Baglam(Zaman(bolum, zaman), a.klasor)  # doku önbelleği
    toplam = int(round(zaman["sure"] * FPS))
    parca_sayisi = a.surec * 3
    sinirlar = np.linspace(0, toplam, parca_sayisi + 1).astype(int)
    isler = [(sinirlar[i], sinirlar[i + 1], str(a.klasor / f"parca_{i:02d}.mp4")) for i in range(parca_sayisi)]
    from multiprocessing import Pool
    with Pool(a.surec, initializer=_baslat, initargs=(ayarlar,)) as havuz:
        for yol in havuz.imap(_parca_isle, isler):
            print("hazır:", yol, flush=True)
    liste = a.klasor / "parcalar.txt"
    liste.write_text("".join(f"file '{pathlib.Path(y).name}'\n" for _, _, y in isler))
    sessiz = a.klasor / "goruntu.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(liste), "-c", "copy", str(sessiz)], check=True)
    print("görüntü:", sessiz)


if __name__ == "__main__":
    main()
