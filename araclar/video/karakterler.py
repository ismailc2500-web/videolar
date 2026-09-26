"""Karakter ve manzara kütüphanesi (kenar ışıklı siluet çizim tarzı).

Karakterler yerel koordinatlarda tanımlanır: ayaklar (0, 0), başın tepesi
yaklaşık (0, -1); x ekseni boy birimindedir. Kalem, bu koordinatları yarım
çözünürlüklü tabana dönüştürür. Siluetler koyu dolgu + ışık yönüne göre
kenar ışığıyla, ışık varlıkları (Ainur, Valar) ise parlayan dolguyla çizilir.
"""
import math

import cv2
import numpy as np

from motor import H, W, h, lekele, w

GOK_MAVI = np.array([0.45, 0.70, 1.0], np.float32)
DENIZ_YESIL = np.array([0.30, 0.95, 0.85], np.float32)
KOR = np.array([1.0, 0.34, 0.07], np.float32)
KIZIL = np.array([1.0, 0.09, 0.03], np.float32)
SICAK = np.array([1.0, 0.88, 0.70], np.float32)
AY_ISIGI = np.array([0.70, 0.82, 1.0], np.float32)
BUZ = np.array([0.75, 0.92, 1.0], np.float32)
BEYAZ_KOPUK = np.array([0.85, 1.0, 1.0], np.float32)


def bezier(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3).tolist()


class Kalem:
    """Yerel karakter koordinatlarından yarım çözünürlüklü maskeye çizer."""

    def __init__(self, u, v, boy, yon=1):
        self.u, self.v, self.boy, self.yon = u, v, boy, yon
        self.m = np.zeros((h, w), np.uint8)
        self.kutu = [w, h, 0, 0]

    def px(self, pts):
        a = np.asarray(pts, np.float32).reshape(-1, 2)
        x = w / 2 + (self.u + a[:, 0] * self.boy * self.yon) * h
        y = h / 2 + (self.v + a[:, 1] * self.boy) * h
        self.kutu = [min(self.kutu[0], x.min()), min(self.kutu[1], y.min()),
                     max(self.kutu[2], x.max()), max(self.kutu[3], y.max())]
        return np.stack([x, y], -1)

    def poligon(self, pts, deger=255):
        cv2.fillPoly(self.m, [np.int32(self.px(pts) * 16)], deger, cv2.LINE_AA, 4)

    def elips(self, x, y, rx, ry, aci=0.0, deger=255):
        c = self.px([(x, y)])[0]
        eksen = (max(1, int(rx * self.boy * h * 16)), max(1, int(ry * self.boy * h * 16)))
        self.px([(x - rx, y - ry), (x + rx, y + ry)])
        cv2.ellipse(self.m, (int(c[0] * 16), int(c[1] * 16)), eksen, aci * self.yon, 0, 360, deger, -1, cv2.LINE_AA, 4)

    def cizgi(self, pts, kalinlik, deger=255):
        k = max(1, int(round(kalinlik * self.boy * h)))
        cv2.polylines(self.m, [np.int32(self.px(pts) * 16)], False, deger, k, cv2.LINE_AA, 4)

    def konik(self, pts, r0, r1, deger=255):
        """Kalınlığı r0'dan r1'e incelen eğri (boynuz, kuyruk, saç tutamı); pürüzsüz kenarlı şerit."""
        p = np.asarray(pts, np.float32)
        if len(p) < 2:
            self.elips(p[0, 0], p[0, 1], r0, r0, deger=deger)
            return
        d = np.gradient(p, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], -1)
        n /= np.maximum(1e-6, np.linalg.norm(n, axis=1, keepdims=True))
        r = np.linspace(r0, r1, len(p))[:, None]
        self.poligon(np.vstack([p + n * r, (p - n * r)[::-1]]), deger)
        self.elips(p[0, 0], p[0, 1], r0, r0, deger=deger)
        self.elips(p[-1, 0], p[-1, 1], r1, r1, deger=deger)

    def kirp(self, pay=18):
        x0, y0, x1, y1 = self.kutu
        x0, y0 = max(0, int(x0) - pay), max(0, int(y0) - pay)
        x1, y1 = min(w, int(x1) + pay), min(h, int(y1) + pay)
        return x0, y0, x1, y1


def isle(taban, kalem, kenar_renk, isik=(-0.6, -0.8), guc=1.0, dolgu=(0.015, 0.015, 0.03),
         kenar_guc=1.0, parlak=False, parlak_renk=None, opaklik=1.0):
    """Maskeyi tabana uygular: koyu siluet + yönlü kenar ışığı, ya da ışık varlığı."""
    if guc <= 0.01:
        return
    x0, y0, x1, y1 = kalem.kirp()
    if x1 <= x0 or y1 <= y0:
        return
    m = kalem.m[y0:y1, x0:x1].astype(np.float32) / 255.0
    bolge = taban[y0:y1, x0:x1]
    mb = cv2.GaussianBlur(m, (0, 0), 1.6)
    gx = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
    lx, ly = isik
    n = math.hypot(lx, ly) or 1
    yonlu = np.clip(-(gx * lx + gy * ly) / n, 0, None)
    kenar = np.sqrt(gx * gx + gy * gy)
    renk = np.asarray(kenar_renk, np.float32)
    if parlak:
        pr = np.asarray(parlak_renk if parlak_renk is not None else kenar_renk, np.float32)
        hale = cv2.GaussianBlur(m, (0, 0), 10)
        yumusak = cv2.GaussianBlur(m, (0, 0), 2.5)
        bolge += (hale * 0.55 + yumusak * 0.35)[..., None] * pr * guc
        bolge += (cv2.GaussianBlur(yonlu, (0, 0), 1.2) * 1.8)[..., None] * SICAK * guc * kenar_guc
        return
    bolge *= (1 - m * guc * opaklik)[..., None]
    bolge += m[..., None] * np.asarray(dolgu, np.float32) * guc
    ince = cv2.GaussianBlur(yonlu * 2.4 + kenar * 0.06, (0, 0), 0.7)
    ic = cv2.GaussianBlur(yonlu, (0, 0), 2.5) * m * 0.6
    dis = cv2.GaussianBlur(yonlu, (0, 0), 7) * (1 - m) * 0.7
    bolge += (ince + ic + dis)[..., None] * renk * guc * kenar_guc


# ------------------------------------------------------------------ karakterler

def pikselle_yerel(u, v, boy, yon, x, y):
    return (u + x * boy * yon, v + y * boy)


def uzuv(k, pts, r0, r1, n=18):
    """Bezier kontrol noktalarından incelen uzuv (kol, bacak, saç tutamı)."""
    k.konik(bezier(*pts, n), r0, r1)


def elf(taban, u, v, boy, t, renk=AY_ISIGI, isik=(-0.5, -0.8), guc=1.0, faz=0.0, yay=True, kadin=False, yon=1):
    k = Kalem(u, v, boy, yon)
    r = 0.012 * math.sin(1.3 * t + faz) + 0.01
    k.elips(0, -0.918, 0.038, 0.050)
    k.elips(0.004, -0.885, 0.028, 0.028)
    for s in (-1, 1):
        k.poligon([(0.030 * s, -0.925), (0.066 * s, -0.948), (0.058 * s, -0.936), (0.034 * s, -0.905)])
    if kadin:
        for s in (-1, 1):
            k.poligon([(0.035 * s, -0.95)] + bezier((0.045 * s, -0.90), (0.07 * s + r, -0.78), (0.085 * s + r, -0.62),
                                                     (0.07 * s + r * 1.5, -0.50), 10) + [(0.03 * s, -0.62), (0.03 * s, -0.86)])
    else:
        for s in (-1, 1):
            k.poligon([(0.036 * s, -0.95)] + bezier((0.046 * s, -0.90), (0.058 * s + r, -0.82), (0.06 * s + r, -0.74),
                                                     (0.05 * s + r, -0.68), 8) + [(0.03 * s, -0.76), (0.03 * s, -0.87)])
    k.poligon([(-0.016, -0.875), (0.016, -0.875), (0.018, -0.83), (-0.018, -0.83)])
    etek = 0.19 if kadin else 0.14
    d = [0.010 * math.sin(2.1 * t + faz + i) for i in range(4)]
    sag = (bezier((0.016, -0.842), (0.07, -0.845), (0.105, -0.835), (0.112, -0.80), 6)
           + bezier((0.112, -0.80), (0.10, -0.70), (0.065, -0.64), (0.068, -0.56), 6)
           + bezier((0.068, -0.56), (0.09, -0.40), (etek - 0.02, -0.20), (etek, -0.02 + d[0]), 8))
    sol = [(-x, y) for x, y in reversed(sag)]
    k.poligon(sol + sag + [(etek * 0.4, d[1]), (-etek * 0.4, d[2]), (-etek, -0.02 + d[3])])
    pel = 0.04 * math.sin(1.1 * t + faz) + 0.05
    k.poligon([(-0.10, -0.83)] + bezier((-0.13, -0.78), (-0.20 - pel * 0.5, -0.55), (-0.25 - pel, -0.30),
                                         (-0.29 - pel * 1.3, -0.02), 12) + [(-0.13, -0.02), (-0.08, -0.50)])
    if kadin:
        uzuv(k, [(-0.105, -0.80), (-0.13, -0.70), (-0.08, -0.60), (-0.01, -0.58)], 0.02, 0.014)
        uzuv(k, [(0.105, -0.80), (0.13, -0.70), (0.07, -0.61), (0.01, -0.585)], 0.02, 0.014)
        for s in (-1, 1):
            k.poligon([(0.10 * s, -0.66), (0.19 * s, -0.46), (0.11 * s, -0.50)])
    else:
        uzuv(k, [(-0.105, -0.80), (-0.13, -0.70), (-0.125, -0.60), (-0.115, -0.50)], 0.021, 0.015)
        uzuv(k, [(0.105, -0.80), (0.14, -0.72), (0.17, -0.62), (0.205, -0.53)], 0.021, 0.015)
        for i in range(3):
            x = -0.05 - i * 0.018
            k.cizgi([(x, -0.80), (x - 0.03, -0.97)], 0.006)
            k.poligon([(x - 0.03, -0.97), (x - 0.042, -0.94), (x - 0.022, -0.955)])
    if yay and not kadin:
        k.cizgi(bezier((0.212, -0.90), (0.30, -0.78), (0.28, -0.60), (0.215, -0.54), 12)
                + bezier((0.215, -0.54), (0.28, -0.48), (0.30, -0.28), (0.212, -0.16), 12), 0.011)
        k.cizgi([(0.212, -0.90), (0.212, -0.16)], 0.003)
    isle(taban, k, renk, isik, guc)
    if kadin:
        c = pikselle_yerel(u, v, boy, yon, 0.0, -0.955)
        lekele(taban, c[0], c[1], 0.012 * boy, 0.012 * boy, np.array([0.7, 0.9, 1.0]), 1.6 * guc)
        lekele(taban, c[0], c[1], 0.003 * boy, 0.003 * boy, SICAK, 5 * guc)


def insan(taban, u, v, boy, t, renk=AY_ISIGI, isik=(0.5, -0.8), guc=1.0, faz=0.0, tur="adam", yon=1):
    k = Kalem(u, v, boy, yon)
    n = 0.003 * math.sin(1.7 * t + faz)
    if tur == "cocuk":
        k.elips(0, -0.86, 0.075, 0.085)
        sag = (bezier((0.02, -0.77), (0.07, -0.77), (0.095, -0.74), (0.10, -0.68), 5)
               + bezier((0.10, -0.68), (0.09, -0.55), (0.11, -0.42), (0.12, -0.34), 5))
        k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
        for s in (-1, 1):
            uzuv(k, [(0.045 * s, -0.36), (0.05 * s, -0.22), (0.045 * s, -0.10), (0.05 * s, -0.02)], 0.03, 0.026)
            uzuv(k, [(0.09 * s, -0.72), (0.13 * s, -0.60), (0.14 * s, -0.52), (0.15 * s, -0.46)], 0.03, 0.024)
        isle(taban, k, renk, isik, guc)
        return
    if tur == "adam":
        k.elips(0, -0.905, 0.048, 0.056)
        k.poligon([(-0.05, -0.93)] + bezier((-0.052, -0.96), (-0.02, -0.975), (0.03, -0.97), (0.052, -0.93), 8))
        k.poligon(bezier((-0.042, -0.89), (-0.045, -0.84), (-0.02, -0.80), (0.0, -0.795), 6)
                  + bezier((0.0, -0.795), (0.02, -0.80), (0.045, -0.84), (0.042, -0.89), 6))
        sag = (bezier((0.02, -0.845 + n), (0.08, -0.85), (0.13, -0.84), (0.14, -0.79), 6)
               + bezier((0.14, -0.79), (0.13, -0.68), (0.10, -0.60), (0.105, -0.52), 6)
               + bezier((0.105, -0.52), (0.13, -0.47), (0.15, -0.42), (0.155, -0.38), 5))
        k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
        for s in (-1, 1):
            uzuv(k, [(0.05 * s, -0.40), (0.055 * s, -0.25), (0.05 * s, -0.12), (0.055 * s, -0.03)], 0.034, 0.028)
            k.poligon([(0.03 * s, -0.04), (0.10 * s, -0.035), (0.10 * s, 0.0), (0.03 * s, 0.0)])
        pel = 0.025 * math.sin(1.2 * t + faz)
        k.poligon([(-0.12, -0.83)] + bezier((-0.15, -0.75), (-0.19 - pel, -0.60), (-0.21 - pel, -0.45),
                                             (-0.22 - pel, -0.32), 8) + [(-0.12, -0.34)])
        uzuv(k, [(-0.13, -0.80), (-0.16, -0.68), (-0.155, -0.58), (-0.15, -0.48)], 0.027, 0.02)
        uzuv(k, [(0.13, -0.80), (0.17, -0.72), (0.19, -0.66), (0.185, -0.60)], 0.027, 0.02)
        k.cizgi([(0.185, -1.18), (0.185, 0.0)], 0.012)
        k.poligon([(0.185, -1.30), (0.203, -1.20), (0.185, -1.17), (0.167, -1.20)])
    else:
        k.elips(0, -0.905, 0.045, 0.054)
        k.elips(-0.02, -0.955, 0.03, 0.026)
        uzuv(k, [(-0.035, -0.90), (-0.07, -0.82), (-0.075, -0.70), (-0.065, -0.60)], 0.02, 0.012)
        d = 0.01 * math.sin(2 * t + faz)
        sag = (bezier((0.016, -0.845), (0.07, -0.845), (0.11, -0.83), (0.115, -0.79), 6)
               + bezier((0.115, -0.79), (0.10, -0.68), (0.07, -0.60), (0.072, -0.54), 6)
               + bezier((0.072, -0.54), (0.11, -0.36), (0.17, -0.15), (0.19 + d, 0.0), 8))
        k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
        uzuv(k, [(0.11, -0.80), (0.15, -0.68), (0.19, -0.60), (0.24, -0.53)], 0.022, 0.016)
        uzuv(k, [(-0.11, -0.80), (-0.13, -0.68), (-0.12, -0.58), (-0.10, -0.52)], 0.022, 0.016)
    isle(taban, k, renk, isik, guc)


def ainu(taban, u, v, boy, t, renk, guc=1.0, faz=0.0, kollar=0.0, hale=True):
    """Işıktan varlık: ince, uzun cüppeli, parlayan siluet. kollar: 0 aşağı, 1 yukarı açık."""
    k = Kalem(u, v, boy)
    d = [0.012 * math.sin(1.6 * t + faz + i * 1.3) for i in range(6)]
    k.elips(0, -0.91, 0.04, 0.052)
    k.poligon([(-0.042, -0.93)] + bezier((-0.06, -0.86), (-0.075 + d[0], -0.72), (-0.07 + d[0], -0.62),
                                          (-0.05, -0.56), 8)
              + bezier((0.05, -0.56), (0.07 + d[1], -0.62), (0.075 + d[1], -0.72), (0.06, -0.86), 8) + [(0.042, -0.93)])
    sag = (bezier((0.015, -0.85), (0.06, -0.85), (0.085, -0.83), (0.09, -0.79), 6)
           + bezier((0.09, -0.79), (0.08, -0.62), (0.10, -0.35), (0.16 + d[2], 0.0), 10))
    sol = [(-x, y) for x, y in reversed(sag)]
    k.poligon(sol + sag + [(0.05, 0.02 + d[3]), (-0.05, 0.02 + d[4])])
    for s in (-1, 1):
        el = (0.20 * s, -0.56 - 0.38 * kollar)
        dirsek = (0.14 * s, -0.68 - 0.12 * kollar)
        uzuv(k, [(0.085 * s, -0.80), dirsek, dirsek, el], 0.018, 0.012)
        k.poligon([(0.09 * s, -0.79), dirsek, (el[0] + 0.01 * s, el[1] + 0.02), (0.13 * s + d[5] * s, -0.40),
                   (0.09 * s, -0.55)])
    isle(taban, k, renk, guc=guc, parlak=True, parlak_renk=renk)
    if hale:
        lekele(taban, u, v - 0.91 * boy, 0.05 * boy, 0.05 * boy, SICAK, 0.9 * guc)
        lekele(taban, u, v - 0.91 * boy, 0.015 * boy, 0.015 * boy, SICAK, 2.0 * guc)


def detay(taban, kalem, cizgiler, renk, guc, kalinlik=1):
    """Siluetin içine ince ışık çizgileri (sakal, zırh, kıvrım) çizer."""
    if guc <= 0.01:
        return
    m = np.zeros((h, w), np.uint8)
    for pts in cizgiler:
        cv2.polylines(m, [np.int32(kalem.px(pts) * 16)], False, 255, kalinlik, cv2.LINE_AA, 4)
    x0, y0, x1, y1 = kalem.kirp()
    c = m[y0:y1, x0:x1].astype(np.float32) / 255.0
    parilti = cv2.GaussianBlur(c, (0, 0), 0.8) * 0.9 + cv2.GaussianBlur(c, (0, 0), 3) * 0.5
    taban[y0:y1, x0:x1] += parilti[..., None] * np.asarray(renk, np.float32) * guc


def dalga_tepesi(cx, taban_y, boy, yon):
    return (bezier((cx - 0.016, taban_y), (cx - 0.02, taban_y - boy * 0.6), (cx + 0.004 * yon, taban_y - boy),
                   (cx + 0.024 * yon, taban_y - boy * 0.82), 10)
            + bezier((cx + 0.024 * yon, taban_y - boy * 0.82), (cx + 0.012 * yon, taban_y - boy * 0.72),
                     (cx + 0.010, taban_y - boy * 0.4), (cx + 0.016, taban_y), 10))


def ulmo(taban, u, v, boy, t, guc=1.0, boynuz=1.0):
    """Denizden yükselen soylu deniz kralı. v: su çizgisi; yerel su çizgisi y = -0.42."""
    k = Kalem(u, v - (-0.42) * boy, boy)
    su = -0.42
    dl = [0.010 * math.sin(1.3 * t + i * 0.9) for i in range(8)]
    k.elips(0, -0.902, 0.047, 0.060)
    for cx, yuk, yon in ((-0.032, 0.045, -1), (0.0, 0.072, 1), (0.032, 0.045, 1)):
        k.poligon(dalga_tepesi(cx, -0.945, yuk, yon))
    k.poligon([(-0.05, -0.95), (0.05, -0.95), (0.052, -0.935), (-0.052, -0.935)])
    for s_ in (-1, 1):
        k.poligon([(0.042 * s_, -0.935)] + bezier((0.055 * s_, -0.91), (0.066 * s_ + dl[0], -0.87),
                                                   (0.072 * s_ + dl[1], -0.83), (0.068 * s_ + dl[2], -0.795), 8)
                  + bezier((0.068 * s_ + dl[2], -0.795), (0.05 * s_, -0.81), (0.04 * s_, -0.85), (0.036 * s_, -0.88), 6))
    k.poligon([(-0.026, -0.86), (0.026, -0.86), (0.03, -0.80), (-0.03, -0.80)])
    sag = (bezier((0.028, -0.835), (0.09, -0.825), (0.17, -0.80), (0.205, -0.772), 8)
           + bezier((0.205, -0.772), (0.215, -0.72), (0.185, -0.66), (0.17, -0.60), 8)
           + bezier((0.17, -0.60), (0.155, -0.52), (0.14, -0.47), (0.14, su - 0.01), 6))
    k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
    uzuv(k, [(-0.20, -0.765), (-0.26, -0.68), (-0.27, -0.56), (-0.265, su)], 0.045, 0.036)
    b = boynuz
    dirsek = (0.28 + 0.03 * b, -0.66 - 0.23 * b)
    el = (0.26 + 0.07 * b, -0.52 - 0.52 * b)
    uzuv(k, [(0.20, -0.765), dirsek, dirsek, el], 0.045, 0.034)
    boynuz_yol = bezier((el[0] - 0.01, el[1] + 0.04), (el[0] + 0.02, el[1] - 0.06),
                        (el[0] + 0.12, el[1] - 0.12), (el[0] + 0.16, el[1] - 0.05), 24)
    k.konik(boynuz_yol, 0.014, 0.055)
    isle(taban, k, DENIZ_YESIL, isik=(0.35, -0.9), guc=guc, dolgu=(0.008, 0.025, 0.035), kenar_guc=1.3)
    detay(taban, k, [bezier((-0.03, -0.85), (-0.05, -0.76), (-0.02, -0.68), (0.0, -0.64), 10),
                     bezier((0.03, -0.85), (0.05, -0.76), (0.02, -0.68), (0.0, -0.64), 10)]
          + [bezier((x - 0.03, -0.60 + j * 0.05), (x - 0.01, -0.575 + j * 0.05), (x + 0.01, -0.575 + j * 0.05),
                    (x + 0.03, -0.60 + j * 0.05), 6) for j in range(3) for x in (-0.09, -0.03, 0.03, 0.09)],
          DENIZ_YESIL, 0.35 * guc)
    for s_ in (-1, 1):
        g = pikselle_yerel(k.u, k.v, boy, 1, 0.019 * s_, -0.905)
        lekele(taban, g[0], g[1], 0.0055 * boy, 0.003 * boy, DENIZ_YESIL, 5 * guc)
    for cx, yuk in ((-0.032, 0.045), (0.0, 0.072), (0.032, 0.045)):
        c = pikselle_yerel(k.u, k.v, boy, 1, cx + 0.01, -0.945 - yuk * 0.9)
        lekele(taban, c[0], c[1], 0.006 * boy, 0.006 * boy, BEYAZ_KOPUK, (1.2 + 0.4 * math.sin(t * 4 + cx * 50)) * guc)
    agiz = pikselle_yerel(k.u, k.v, boy, 1, *boynuz_yol[-1])
    lekele(taban, agiz[0], agiz[1], 0.05 * boy, 0.05 * boy, SICAK, 0.6 * guc * b)


def manwe(taban, u, v, boy, t, guc=1.0, isaret=0.0, yon=1):
    """Taçlı kral: rüzgârda pelerin, safir küreli asa. isaret: 0-1 kol kaldırma."""
    k = Kalem(u, v, boy, yon)
    k.elips(0, -0.915, 0.045, 0.058)
    tac = [(-0.05, -0.945), (-0.052, -0.99), (-0.036, -0.962), (-0.022, -1.015), (-0.01, -0.968), (0, -1.05),
           (0.01, -0.968), (0.022, -1.015), (0.036, -0.962), (0.052, -0.99), (0.05, -0.945)]
    k.poligon(tac)
    for s in (-1, 1):
        k.poligon([(0.042 * s, -0.93), (0.07 * s, -0.84), (0.075 * s, -0.76), (0.04 * s, -0.80)])
    k.poligon([(-0.03, -0.87), (0.03, -0.87), (0.02, -0.76), (0, -0.72), (-0.02, -0.76)])
    k.poligon([(-0.11, -0.83), (0.11, -0.83), (0.13, -0.78), (0.14, -0.50), (0.21, 0.0), (-0.21, 0.0), (-0.14, -0.50),
               (-0.13, -0.78)])
    r = [0.04 * math.sin(1.4 * t + i * 0.8) for i in range(8)]
    ust = bezier((-0.11, -0.83), (-0.22 + r[0], -0.78), (-0.36 + r[1], -0.62), (-0.44 + r[2], -0.42), 10)
    alt = bezier((-0.44 + r[2], -0.42), (-0.52 + r[3], -0.26), (-0.50 + r[4], -0.12), (-0.58 + r[5], -0.02), 10)
    etek = [(-0.50 + r[6], 0.01), (-0.42 + r[5], -0.03), (-0.34 + r[7], 0.0), (-0.24, -0.02)]
    k.poligon(ust + alt + etek + bezier((-0.20, -0.05), (-0.17, -0.30), (-0.15, -0.55), (-0.12, -0.78), 8))
    k.cizgi([(0.16, -1.05), (0.16, -0.01)], 0.014)
    k.cizgi([(0.11, -0.80), (0.16, -0.68), (0.165, -0.60)], 0.04)
    el = (-0.20 - 0.14 * isaret, -0.55 - 0.22 * isaret)
    k.cizgi([(-0.11, -0.80), (-0.17 - 0.03 * isaret, -0.66 - 0.06 * isaret), el], 0.04)
    k.elips(el[0], el[1], 0.022, 0.018)
    isle(taban, k, GOK_MAVI, isik=(0.4, -0.9), guc=guc, dolgu=(0.01, 0.015, 0.04), kenar_guc=1.3)
    detay(taban, k, [[(-0.125, -0.56), (0.125, -0.56)], [(0.0, -0.76), (0.0, -0.56)],
                     bezier((0.0, -0.56), (0.01, -0.35), (0.03, -0.20), (0.05, 0.0), 8),
                     bezier((-0.05, -0.83), (-0.03, -0.80), (0.03, -0.80), (0.05, -0.83), 6)], GOK_MAVI, 0.45 * guc)
    kure = pikselle_yerel(u, v, boy, yon, 0.16, -1.07)
    lekele(taban, kure[0], kure[1], 0.028 * boy, 0.028 * boy, np.array([0.3, 0.55, 1.0]), 1.6 * guc)
    lekele(taban, kure[0], kure[1], 0.008 * boy, 0.008 * boy, SICAK, 4.0 * guc)
    for x, y in ((0, -1.03), (-0.022, -1.0), (0.022, -1.0)):
        c = pikselle_yerel(u, v, boy, yon, x, y)
        lekele(taban, c[0], c[1], 0.004 * boy, 0.004 * boy, np.array([0.6, 0.8, 1.0]), 3.0 * guc)


def aule(taban, u, v, boy, t, guc=1.0, vurus=0.0, yon=1):
    """Demirci. vurus: 0 = çekiç yukarıda, 1 = örse indi."""
    k = Kalem(u, v, boy, yon)
    k.elips(0, -0.895, 0.052, 0.058)
    for s in (-1, 1):
        uzuv(k, [(0.02 * s, -0.86), (0.035 * s, -0.78), (0.03 * s, -0.70), (0.022 * s, -0.63)], 0.022, 0.01)
    k.poligon(bezier((-0.05, -0.88), (-0.06, -0.80), (-0.03, -0.74), (0.0, -0.72), 6)
              + bezier((0.0, -0.72), (0.03, -0.74), (0.06, -0.80), (0.05, -0.88), 6))
    sag = (bezier((0.03, -0.83), (0.10, -0.84), (0.18, -0.82), (0.20, -0.76), 6)
           + bezier((0.20, -0.76), (0.19, -0.64), (0.14, -0.56), (0.13, -0.48), 6)
           + bezier((0.13, -0.48), (0.15, -0.40), (0.16, -0.30), (0.16, -0.24), 5))
    k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
    for s in (-1, 1):
        uzuv(k, [(0.07 * s, -0.30), (0.10 * s, -0.18), (0.12 * s, -0.08), (0.13 * s, -0.03)], 0.05, 0.042)
        k.poligon([(0.08 * s, -0.045), (0.19 * s, -0.04), (0.20 * s, 0.0), (0.08 * s, 0.0)])
    uzuv(k, [(-0.19, -0.77), (-0.22, -0.64), (-0.14, -0.56), (0.05, -0.50)], 0.045, 0.034)
    f = vurus
    dirsek = (0.30 + 0.02 * f, -0.95 + 0.29 * f)
    el = (0.25 + 0.06 * f, -1.08 + 0.46 * f)
    uzuv(k, [(0.19, -0.77), dirsek, dirsek, el], 0.05, 0.035)
    aci = math.radians(-110 + 190 * f)
    sap_yon = np.array([math.cos(aci), math.sin(aci)])
    el = np.array(el)
    uc = el + 0.24 * sap_yon
    k.cizgi([tuple(el - 0.03 * sap_yon), tuple(uc)], 0.02)
    dik = np.array([-sap_yon[1], sap_yon[0]])
    k.poligon([tuple(p) for p in (uc + 0.08 * dik - 0.03 * sap_yon, uc + 0.08 * dik + 0.05 * sap_yon,
                                  uc - 0.07 * dik + 0.05 * sap_yon, uc - 0.07 * dik - 0.03 * sap_yon)])
    k.cizgi([(0.05, -0.50), (0.34, -0.43)], 0.012)
    k.poligon([(0.24, -0.43), (0.52, -0.43), (0.62, -0.47), (0.68, -0.46), (0.55, -0.38), (0.48, -0.36),
               (0.46, -0.25), (0.50, -0.21), (0.28, -0.21), (0.32, -0.25), (0.30, -0.36)])
    k.poligon([(0.31, -0.21), (0.47, -0.21), (0.50, 0.0), (0.28, 0.0)])
    isle(taban, k, KOR, isik=(0.9, 0.1), guc=guc, dolgu=(0.03, 0.012, 0.005), kenar_guc=1.5)
    detay(taban, k, [[(-0.12, -0.60), (-0.14, -0.22), (0.14, -0.22), (0.12, -0.60)], [(-0.15, -0.50), (0.15, -0.50)],
                     [(-0.02, -0.63), (0.02, -0.63)]], KOR, 0.4 * guc)
    demir = pikselle_yerel(u, v, boy, yon, 0.36, -0.445)
    lekele(taban, demir[0], demir[1], 0.05 * boy, 0.008 * boy, np.array([1.0, 0.55, 0.15]), 3.0 * guc)
    lekele(taban, demir[0], demir[1], 0.14 * boy, 0.06 * boy, KOR, 0.8 * guc)
    for s in (-1, 1):
        g = pikselle_yerel(u, v, boy, yon, 0.02 * s, -0.90)
        lekele(taban, g[0], g[1], 0.004 * boy, 0.0025 * boy, KOR, 3 * guc)


def kartal(taban, u, v, boy, t, renk=GOK_MAVI, guc=1.0, faz=0.0, isik=(0.3, -0.9)):
    """Aşağıdan görülen, kanat çırpan kartal. boy: kanat açıklığı."""
    k = Kalem(u, v, boy)
    cirp = 0.10 * math.sin(3.2 * t + faz)
    k.elips(0, 0, 0.035, 0.11)
    k.elips(0, -0.12, 0.022, 0.028)
    k.poligon([(-0.012, -0.145), (0.012, -0.145), (0, -0.175)])
    k.poligon([(-0.03, 0.08), (0.03, 0.08), (0.065, 0.20), (0.0, 0.215), (-0.065, 0.20)])
    for s in (-1, 1):
        uc = (0.50 * s, -0.08 - cirp)
        on = bezier((0.02 * s, -0.05), (0.18 * s, -0.12 - cirp * 0.5), (0.35 * s, -0.12 - cirp), uc, 10)
        parmak = []
        for i in range(5):
            f = i / 4
            px = uc[0] * (1 - 0.35 * f)
            py = uc[1] + 0.03 + 0.07 * f + cirp * 0.3 * f
            parmak += [(px, py), (px - 0.035 * s, py - 0.035)]
        arka = [(0.12 * s, 0.07 - cirp * 0.2), (0.03 * s, 0.05)]
        k.poligon(on + parmak + arka)
    isle(taban, k, renk, isik=isik, guc=guc, dolgu=(0.01, 0.01, 0.02), kenar_guc=0.9)
    return k


SILUET_SAG = [(0, -0.07), (0.03, 0.01), (0.06, -0.05), (0.09, 0.02), (0.13, -0.03), (0.12, 0.05), (0.125, 0.09),
              (0.10, 0.14), (0.09, 0.16), (0.22, 0.17), (0.30, 0.13), (0.33, 0.19), (0.40, 0.22), (0.43, 0.30),
              (0.44, 0.42), (0.47, 0.60), (0.52, 0.80), (0.58, 0.97), (0.46, 0.93), (0.40, 1.0), (0.28, 0.94),
              (0.18, 1.0), (0.05, 0.95)]


def melkor_dev(taban, u, v, boy, t, guc=1.0, kor_guc=1.0, isik=(0.0, -1.0)):
    """Dikenli taçlı kara lord (1. bölümdeki siluetle aynı). v: ayak hizası."""
    k = Kalem(u, v, boy)
    sag = [(x * 0.405, (y - 1.0) * 1.0) for x, y in SILUET_SAG]
    k.poligon(sag + [(-x, y) for x, y in reversed(sag[1:])])
    isle(taban, k, KIZIL, isik=isik, guc=guc, dolgu=(0.02, 0.0, 0.0), kenar_guc=1.3 * kor_guc)
    for s in (-1, 1):
        g = (u + 0.018 * s * boy, v - 0.915 * boy)
        lekele(taban, g[0], g[1], 0.0045 * boy, 0.003 * boy, KOR, 7 * guc * kor_guc)
    return k


# ------------------------------------------------------------------ manzara

class Yildizlar:
    def __init__(self, adet=900, tohum=21):
        rng = np.random.default_rng(tohum)
        self.x = rng.uniform(0, W, adet)
        self.y = rng.uniform(0, H, adet)
        self.b = rng.uniform(0.15, 1.0, adet) ** 2.2
        self.r = rng.choice([1, 1, 1, 1, 2, 2, 3], adet)
        self.f = rng.uniform(0, 6.28, adet)
        self.renk = np.array([(1.0, 1.0, 1.0), (0.8, 0.88, 1.0), (1.0, 0.9, 0.75)], np.float32)[rng.integers(0, 3, adet)]

    def ciz(self, ayrinti, t, guc, y_sinir=H, kayma=(0.0, 0.0)):
        if guc <= 0.01:
            return
        tw = 0.65 + 0.35 * np.sin(1.7 * t + self.f)
        for x, y, b, r, c, k in zip(self.x, self.y, self.b, self.r, self.renk, tw):
            x = (x + kayma[0]) % W
            y = (y + kayma[1]) % H
            if y > y_sinir:
                continue
            deger = b * k * guc * 255
            if deger < 6:
                continue
            cv2.circle(ayrinti, (int(x * 16), int(y * 16)), int(r), tuple(float(v) for v in c * deger), -1, cv2.LINE_AA, 4)


def gok(taban, ust, alt, ufuk_v=0.1, guc=1.0):
    V = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]
    f = np.clip((V + 0.5) / (ufuk_v + 0.5), 0, 1)[..., None]
    taban += ((np.asarray(ust) * (1 - f) + np.asarray(alt) * f) * guc).astype(np.float32)[:, None, :].reshape(h, 1, 3)


def fbm1(x, tohum, oktav=5):
    rng = np.random.default_rng(tohum)
    y = np.zeros_like(x)
    g, fr = 1.0, 1.0
    for _ in range(oktav):
        faz = rng.uniform(0, 100, 3)
        y += g * (np.sin(x * fr * 2.1 + faz[0]) * 0.5 + np.sin(x * fr * 3.7 + faz[1]) * 0.3 + np.sin(x * fr * 5.3 + faz[2]) * 0.2)
        g *= 0.5
        fr *= 2.03
    return y


def sirt(taban, v_taban, yukseklik, tohum, renk, kenar=None, kenar_guc=0.0, kayma=0.0, sivri=1.0, guc=1.0,
         yukselme=1.0, cokme=0.0, t=0.0):
    """Dağ sırası: fbm tepe çizgisi, altı dolu; isteğe bağlı kenar parıltısı."""
    x = np.linspace(-0.4, 0.4, 160, dtype=np.float32)
    y = fbm1((x + kayma) * 6, tohum)
    y = np.sign(y) * np.abs(y) ** sivri
    tepe = v_taban - yukseklik * yukselme * (0.55 + 0.45 * y)
    if cokme:
        rs = np.random.default_rng(tohum + int(t * 10))
        tepe = tepe + cokme * yukseklik * 0.5 * np.abs(rs.standard_normal(len(x))) * 0.3
    pts = np.stack([w / 2 + x * h, h / 2 + tepe * h], -1)
    poli = np.vstack([pts, [[w + 5, h + 5], [-5, h + 5]]])
    maske = np.zeros((h, w), np.uint8)
    cv2.fillPoly(maske, [np.int32(poli * 16)], 255, cv2.LINE_AA, 4)
    m = maske.astype(np.float32)[..., None] / 255.0
    taban *= (1 - m * guc)
    taban += m * np.asarray(renk, np.float32) * guc
    if kenar is not None and kenar_guc > 0:
        cizgi = np.zeros((h, w), np.uint8)
        cv2.polylines(cizgi, [np.int32(pts * 16)], False, 255, 1, cv2.LINE_AA, 4)
        c = cizgi.astype(np.float32) / 255.0
        parilti = cv2.GaussianBlur(c, (0, 0), 1.0) * 1.2 + cv2.GaussianBlur(c, (0, 0), 6) * 0.8
        taban += parilti[..., None] * np.asarray(kenar, np.float32) * kenar_guc * guc
    return tepe


def deniz(taban, ayrinti, v_ufuk, t, renk=(0.02, 0.06, 0.10), yansima=None, yansima_u=0.0, guc=1.0):
    y0 = int(h / 2 + v_ufuk * h)
    if y0 >= h:
        return
    d = np.linspace(0, 1, h - y0, dtype=np.float32)[:, None, None]
    taban[y0:] = taban[y0:] * (1 - guc) + (np.asarray(renk) * (0.6 + 0.8 * d) * guc)
    rng = np.random.default_rng(4)
    for i in range(160):
        derinlik = rng.uniform(0, 1) ** 2
        yy = H / 2 + (v_ufuk + 0.005 + derinlik * (0.5 - v_ufuk)) * H
        uzun = 8 + 90 * derinlik
        xx = (rng.uniform(0, W) + t * (10 + 30 * derinlik) * (1 if i % 2 else -1)) % (W + 200) - 100
        yakin = abs((xx - W / 2) / H - yansima_u) if yansima is not None else 1
        parlak = (0.10 + 0.35 * derinlik) * (1 + (3.5 * math.exp(-(yakin / 0.06) ** 2) if yansima is not None else 0))
        renk_c = (np.asarray(yansima if yansima is not None else (0.5, 0.7, 0.9)) * parlak * 255 * guc).tolist()
        cv2.line(ayrinti, (int((xx - uzun) * 16), int(yy * 16)), (int((xx + uzun) * 16), int(yy * 16)), renk_c,
                 1 + int(derinlik * 2), cv2.LINE_AA, 4)
    if yansima is not None:
        V = np.linspace(-0.5, 0.5, h, dtype=np.float32)[y0:, None]
        U = (np.arange(w, dtype=np.float32)[None, :] - w / 2) / h
        kolon = np.exp(-((U - yansima_u) / (0.03 + 0.12 * (V - v_ufuk))) ** 2) * np.exp(-(V - v_ufuk) / 0.25)
        taban[y0:] += (kolon[..., None] * np.asarray(yansima) * 0.45 * guc).astype(np.float32)


def isik_sutunu(taban, u, v_ust, v_alt, genislik, renk, guc):
    if guc <= 0.01:
        return
    x0, x1 = int(max(0, w / 2 + (u - 4 * genislik) * h)), int(min(w, w / 2 + (u + 4 * genislik) * h + 1))
    y0, y1 = int(max(0, h / 2 + v_ust * h)), int(min(h, h / 2 + v_alt * h))
    if x1 <= x0 or y1 <= y0:
        return
    U = (np.arange(x0, x1, dtype=np.float32) - w / 2) / h
    gx = np.exp(-0.5 * ((U - u) / genislik) ** 2) + 0.6 * np.exp(-0.5 * ((U - u) / (genislik * 0.25)) ** 2)
    gy = np.linspace(0.4, 1.0, y1 - y0, dtype=np.float32)
    taban[y0:y1, x0:x1] += (gy[:, None] * gx[None, :])[..., None] * np.asarray(renk, np.float32) * guc


def kar_tanesi(taban, ayrinti, u, v, r, t, buyume=1.0, guc=1.0, donus=0.0):
    maske = np.zeros((h, w), np.uint8)
    c = np.array([w / 2 + u * h, h / 2 + v * h])
    R = r * h

    def cz(p, q, k):
        cv2.line(maske, tuple(np.int32(p * 16)), tuple(np.int32(q * 16)), 255, max(1, int(k)), cv2.LINE_AA, 4)

    for i in range(6):
        a = donus + i * math.pi / 3
        d = np.array([math.cos(a), math.sin(a)])
        uc = c + d * R * buyume
        cz(c, uc, R * 0.05)
        for f, uz in ((0.35, 0.34), (0.55, 0.28), (0.75, 0.18), (0.9, 0.08)):
            if f > buyume:
                continue
            p = c + d * R * f
            for s in (-1, 1):
                b = a + s * math.pi / 3
                q = p + np.array([math.cos(b), math.sin(b)]) * R * uz * min(1, (buyume - f) * 3)
                cz(p, q, R * 0.03)
                if uz > 0.2:
                    for g in (0.5,):
                        pp = p + (q - p) * g
                        qq = pp + np.array([math.cos(a), math.sin(a)]) * R * uz * 0.35
                        cz(pp, qq, R * 0.018)
    ic = [c + R * 0.16 * np.array([math.cos(donus + i * math.pi / 3), math.sin(donus + i * math.pi / 3)]) for i in range(6)]
    cv2.polylines(maske, [np.int32(np.array(ic) * 16)], True, 255, max(1, int(R * 0.03)), cv2.LINE_AA, 4)
    m = maske.astype(np.float32) / 255.0
    parilti = cv2.GaussianBlur(m, (0, 0), 2) * 1.3 + cv2.GaussianBlur(m, (0, 0), 9) * 1.0
    taban += (m * 1.4 + parilti)[..., None] * BUZ * guc


def yanardag(taban, u, v, boy, t, guc=1.0, patlama=1.0, tohum=0):
    k = Kalem(u, v, boy)
    k.poligon([(-0.9, 0.0), (-0.35, -0.55), (-0.12, -0.92), (0.10, -0.95), (0.30, -0.60), (0.85, 0.0)])
    isle(taban, k, KOR, isik=(0.0, -1.0), guc=guc, dolgu=(0.02, 0.008, 0.005), kenar_guc=0.8)
    agiz = (u - 0.01 * boy, v - 0.95 * boy)
    lekele(taban, agiz[0], agiz[1], 0.10 * boy, 0.05 * boy, KOR, 1.4 * guc * patlama)
    lekele(taban, agiz[0], agiz[1] - 0.15 * boy, 0.12 * boy, 0.25 * boy, np.array([0.5, 0.08, 0.02]), 0.7 * guc * patlama)
    rng = np.random.default_rng(tohum)
    for i in range(3):
        x0 = rng.uniform(-0.06, 0.06)
        yon_ = 1 if i % 2 else -1
        for j in range(26):
            f = j / 25
            x = x0 + yon_ * f * rng.uniform(0.25, 0.4) + 0.02 * math.sin(j * 1.7 + i)
            y = -0.93 + f * 0.75
            c = (u + x * boy, v + y * boy)
            lekele(taban, c[0], c[1], 0.010 * boy, 0.016 * boy, KOR, (0.55 - 0.35 * f) * guc * patlama)


# ------------------------------------------------------------------ 3. bölüm: tahtlar, at, gezgin, kara lord

def taht(taban, u, v, boy, renk, guc=1.0, isik=(0.0, -1.0), kenar_guc=1.0, dolgu=(0.012, 0.012, 0.025)):
    """Yüksek sırtlı taht (önden). v: taban hizası; boy: sırtın yüksekliği."""
    k = Kalem(u, v, boy)
    k.poligon([(-0.16, -0.62), (-0.13, -0.86), (-0.07, -0.93), (0.0, -1.0), (0.07, -0.93), (0.13, -0.86), (0.16, -0.62),
               (0.16, -0.30), (-0.16, -0.30)])
    for s in (-1, 1):
        k.poligon([(0.16 * s, -0.40), (0.26 * s, -0.42), (0.27 * s, -0.36), (0.25 * s, 0.0), (0.18 * s, 0.0),
                   (0.18 * s, -0.30)])
        k.elips(0.265 * s, -0.43, 0.035, 0.03)
    k.poligon([(-0.25, -0.32), (0.25, -0.32), (0.27, -0.24), (-0.27, -0.24)])
    k.poligon([(-0.22, -0.24), (0.22, -0.24), (0.24, 0.0), (-0.24, 0.0)])
    k.poligon([(-0.32, 0.0), (0.32, 0.0), (0.34, 0.05), (-0.34, 0.05)])
    isle(taban, k, renk, isik=isik, guc=guc, dolgu=dolgu, kenar_guc=kenar_guc)
    detay(taban, k, [[(-0.10, -0.62), (-0.08, -0.84), (0.0, -0.92), (0.08, -0.84), (0.10, -0.62), (-0.10, -0.62)],
                     [(0.0, -0.88), (0.0, -0.66)]], renk, 0.35 * guc * kenar_guc)
    return k


def at(taban, u, v, boy, t, renk=(0.95, 0.97, 1.0), guc=1.0, hiz=1.0, yon=1, binici=True, boru=0.0, parlak=True):
    """Dörtnala koşan at (Nahar) ve isteğe bağlı binici. v: toynak hizası; boy: sağrı yüksekliği ~0.7."""
    k = Kalem(u, v, boy, yon)
    f = t * 2.4 * hiz * 2 * math.pi
    sl = 0.022 * math.sin(f)
    eg = 0.03 * math.sin(f + 0.6)

    def o(x, y):
        return (x, y + sl + eg * x)

    govde = (bezier(o(-0.40, -0.66), o(-0.30, -0.76), o(-0.10, -0.70), o(0.10, -0.71), 8)
             + bezier(o(0.10, -0.71), o(0.22, -0.72), o(0.30, -0.86), o(0.40, -0.98), 8)
             + bezier(o(0.40, -0.98), o(0.44, -1.04), o(0.48, -1.08), o(0.52, -1.06), 5)
             + bezier(o(0.52, -1.06), o(0.60, -1.00), o(0.70, -0.90), o(0.72, -0.86), 6)
             + bezier(o(0.72, -0.86), o(0.70, -0.82), o(0.64, -0.83), o(0.56, -0.88), 5)
             + bezier(o(0.56, -0.88), o(0.48, -0.86), o(0.40, -0.80), o(0.34, -0.66), 6)
             + bezier(o(0.34, -0.66), o(0.32, -0.54), o(0.20, -0.48), o(0.0, -0.47), 8)
             + bezier(o(0.0, -0.47), o(-0.20, -0.46), o(-0.36, -0.48), o(-0.42, -0.58), 8))
    k.poligon(govde)
    k.poligon([o(0.47, -1.06), o(0.48, -1.15), o(0.51, -1.07)])
    for i in range(7):
        x = 0.24 + i * 0.035
        y = -0.78 - i * 0.042
        dal = 0.05 * math.sin(f * 0.5 + i * 0.7)
        k.konik(bezier(o(x, y), o(x - 0.06, y + 0.01), o(x - 0.12, y + 0.03 + dal), o(x - 0.19, y + 0.06 + dal), 10),
                0.022, 0.004)
    kuyruk = bezier(o(-0.40, -0.68), (-0.52, -0.72 + sl), (-0.62, -0.64 + 0.05 * math.sin(f)),
                    (-0.74, -0.56 + 0.07 * math.sin(f + 1)), 16)
    k.konik(kuyruk, 0.04, 0.008)
    # bacaklar: ön ve arka çift, dörtnal döngüsü
    for x0, faz, on in ((0.26, 0.0, True), (0.18, 0.8, True), (-0.28, 2.6, False), (-0.34, 3.3, False)):
        a = math.sin(f + faz)
        c = math.cos(f + faz)
        ust = o(x0, -0.58)
        if on:
            diz = (x0 + 0.08 * a + 0.03, -0.36 - 0.10 * max(0.0, c))
            ayak = (diz[0] + 0.06 * a - 0.02 - 0.10 * max(0.0, c), -0.02 - 0.20 * max(0.0, c))
        else:
            diz = (x0 - 0.06 + 0.07 * a, -0.34 - 0.06 * max(0.0, -c))
            ayak = (diz[0] + 0.02 - 0.08 * max(0.0, -c) + 0.04 * a, -0.02 - 0.14 * max(0.0, -c))
        k.konik([ust, ((ust[0] + diz[0]) / 2, (ust[1] + diz[1]) / 2), diz], 0.055, 0.024)
        k.konik([diz, ((diz[0] + ayak[0]) / 2, (diz[1] + ayak[1]) / 2), ayak], 0.024, 0.018)
        k.elips(ayak[0] + 0.01, ayak[1], 0.026, 0.018)
    if binici:
        k.poligon([o(-0.06, -0.72), o(0.10, -0.72), o(0.08, -1.08), o(0.04, -1.14), o(-0.03, -1.10)])
        k.elips(*o(0.05, -1.21), 0.045, 0.055)
        k.konik([o(0.03, -0.72), o(0.08, -0.60), o(0.06, -0.50)], 0.035, 0.022)
        pel = [0.05 * math.sin(f * 0.5 + i) for i in range(4)]
        k.poligon([o(0.0, -1.12), (-0.20 + pel[0], -1.08 + sl), (-0.36 + pel[1], -0.98 + sl), (-0.46 + pel[2], -0.86 + sl),
                   (-0.30 + pel[3], -0.84 + sl), o(-0.04, -0.86)])
        if boru > 0:
            el = o(0.15, -1.22 - 0.02 * boru)
            k.konik([o(0.08, -1.08), o(0.15, -1.08), el], 0.022, 0.018)
            k.konik(bezier(el, (el[0] + 0.06, el[1] - 0.06), (el[0] + 0.12, el[1] - 0.10), (el[0] + 0.16, el[1] - 0.16),
                           10), 0.012, 0.042)
        else:
            k.konik([o(0.08, -1.08), o(0.18, -0.98), o(0.30, -0.94)], 0.022, 0.016)
    if parlak:
        isle(taban, k, renk, guc=guc, parlak=True, parlak_renk=renk, kenar_guc=0.8)
    else:
        isle(taban, k, renk, isik=(0.3, -0.9), guc=guc, dolgu=(0.02, 0.02, 0.03), kenar_guc=1.2)
    return k


def gezgin(taban, u, v, boy, t, renk=(0.85, 0.88, 0.95), guc=1.0, yon=1, parlak=False, asa_isik=0.0):
    """Sivri, geniş kenarlı şapkalı, asalı, sakallı gezgin (gri büyücü ipucu)."""
    k = Kalem(u, v, boy, yon)
    r = [0.02 * math.sin(1.3 * t + i) for i in range(5)]
    k.poligon([(-0.13, -0.88), (0.13, -0.88), (0.14, -0.86), (-0.14, -0.86)])
    k.poligon([(-0.13, -0.87), (-0.06, -0.90), (-0.04, -0.98), (0.02 + r[0], -1.10), (0.09 + r[0] * 2, -1.19),
               (0.05, -1.04), (0.05, -0.93), (0.09, -0.88)])
    k.elips(0.0, -0.84, 0.042, 0.05)
    k.poligon(bezier((-0.04, -0.84), (-0.05, -0.74), (-0.02, -0.64), (0.0, -0.60), 6)
              + bezier((0.0, -0.60), (0.02, -0.64), (0.05, -0.74), (0.04, -0.84), 6))
    sag = (bezier((0.03, -0.80), (0.08, -0.80), (0.12, -0.78), (0.13, -0.72), 6)
           + bezier((0.13, -0.72), (0.12, -0.55), (0.14, -0.30), (0.19 + r[1], 0.0), 10))
    k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
    k.poligon([(-0.12, -0.74)] + bezier((-0.18, -0.62), (-0.24 + r[2], -0.42), (-0.28 + r[3], -0.20),
                                         (-0.30 + r[4], 0.0), 10) + [(-0.15, 0.0)])
    k.cizgi([(0.22, -1.02), (0.25, 0.0)], 0.018)
    k.poligon([(0.205, -1.02), (0.235, -1.10), (0.26, -1.05), (0.24, -0.99)])
    uzuv(k, [(0.11, -0.74), (0.18, -0.66), (0.21, -0.72), (0.225, -0.78)], 0.03, 0.022)
    if parlak:
        isle(taban, k, renk, guc=guc, parlak=True, parlak_renk=renk)
    else:
        isle(taban, k, renk, isik=(-0.5, -0.8), guc=guc, dolgu=(0.02, 0.02, 0.025), kenar_guc=1.2)
    if asa_isik > 0:
        c = pikselle_yerel(u, v, boy, yon, 0.235, -1.05)
        lekele(taban, c[0], c[1], 0.03 * boy, 0.03 * boy, np.array([0.8, 0.9, 1.0]), 1.4 * asa_isik * guc)
        lekele(taban, c[0], c[1], 0.008 * boy, 0.008 * boy, SICAK, 4.0 * asa_isik * guc)
    return k


def kara_lord(taban, u, v, boy, t, guc=1.0, yon=1, kor_guc=1.0):
    """Dikenli miğferli, zırhlı kara lord (Sauron ipucu); kor gibi yanan gözler ve zırh çatlakları."""
    k = Kalem(u, v, boy, yon)
    r = [0.02 * math.sin(1.1 * t + i) for i in range(6)]
    k.poligon([(-0.055, -0.86), (-0.06, -0.93), (-0.045, -0.99), (-0.03, -1.10), (-0.015, -1.00), (0.0, -1.16),
               (0.015, -1.00), (0.03, -1.10), (0.045, -0.99), (0.06, -0.93), (0.055, -0.86), (0.035, -0.82),
               (-0.035, -0.82)])
    k.poligon([(-0.04, -0.84), (0.04, -0.84), (0.05, -0.78), (-0.05, -0.78)])
    for s in (-1, 1):
        k.poligon([(0.04 * s, -0.80), (0.12 * s, -0.83), (0.20 * s, -0.84), (0.25 * s, -0.80), (0.22 * s, -0.76),
                   (0.26 * s, -0.74), (0.20 * s, -0.68), (0.10 * s, -0.70)])
        k.poligon([(0.18 * s, -0.84), (0.20 * s, -0.93), (0.22 * s, -0.84)])
    sag = (bezier((0.03, -0.78), (0.10, -0.78), (0.16, -0.74), (0.16, -0.66), 6)
           + bezier((0.16, -0.66), (0.14, -0.50), (0.17, -0.25), (0.24 + r[0], 0.0), 10))
    k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
    for s in (-1, 1):
        k.poligon([(0.18 * s, -0.78)] + bezier((0.24 * s, -0.66), (0.30 * s + r[1] * s, -0.40),
                                                (0.34 * s + r[2] * s, -0.20), (0.38 * s + r[3] * s, 0.0), 10)
                  + [(0.20 * s, 0.0)])
    uzuv(k, [(0.20, -0.72), (0.26, -0.58), (0.24, -0.50), (0.20, -0.44)], 0.04, 0.03)
    uzuv(k, [(-0.20, -0.72), (-0.26, -0.58), (-0.24, -0.50), (-0.20, -0.44)], 0.04, 0.03)
    isle(taban, k, KOR, isik=(0.0, -1.0), guc=guc, dolgu=(0.02, 0.006, 0.004), kenar_guc=1.3 * kor_guc)
    detay(taban, k, [[(-0.10, -0.60), (-0.02, -0.52), (-0.06, -0.40)], [(0.08, -0.62), (0.03, -0.50), (0.09, -0.36)],
                     [(-0.04, -0.30), (0.02, -0.22), (-0.01, -0.10)]], KOR, 0.6 * guc * kor_guc)
    for s in (-1, 1):
        g = pikselle_yerel(u, v, boy, yon, 0.02 * s, -0.915)
        lekele(taban, g[0], g[1], 0.0045 * boy, 0.0028 * boy, np.array([1.0, 0.55, 0.12]), 8 * guc * kor_guc)
    return k


def hobbit(taban, u, v, boy, t, renk=AY_ISIGI, guc=1.0, kol=1.0, yon=1, isik=(0.0, -1.0)):
    """Kıvırcık saçlı, pelerinli küçük yolcu; kol: 0 aşağı, 1 yukarı kaldırılmış (şişe tutar)."""
    k = Kalem(u, v, boy, yon)
    k.elips(0, -0.86, 0.085, 0.09)
    for i in range(7):
        a = math.pi * (1.1 + 0.8 * i / 6)
        k.elips(0.075 * math.cos(a), -0.88 + 0.08 * math.sin(a), 0.03, 0.028)
    r = 0.015 * math.sin(1.5 * t)
    sag = (bezier((0.03, -0.77), (0.09, -0.77), (0.13, -0.74), (0.14, -0.68), 5)
           + bezier((0.14, -0.68), (0.14, -0.45), (0.17, -0.25), (0.20 + r, -0.10), 8))
    k.poligon([(-x, y) for x, y in reversed(sag)] + sag)
    for s in (-1, 1):
        uzuv(k, [(0.05 * s, -0.14), (0.055 * s, -0.08), (0.06 * s, -0.04), (0.06 * s, 0.0)], 0.035, 0.03)
        k.elips(0.08 * s, -0.005, 0.05, 0.02)
    el = (0.12 + 0.02 * kol, -0.60 - 0.52 * kol)
    dirsek = (0.16, -0.66 - 0.18 * kol)
    uzuv(k, [(0.10, -0.72), dirsek, dirsek, el], 0.032, 0.024)
    uzuv(k, [(-0.10, -0.72), (-0.15, -0.60), (-0.14, -0.50), (-0.12, -0.44)], 0.032, 0.024)
    isle(taban, k, renk, isik=isik, guc=guc, dolgu=(0.012, 0.012, 0.02), kenar_guc=1.2)
    return pikselle_yerel(u, v, boy, yon, el[0], el[1] - 0.03)


def golge_yaratik(taban, u, v, boy, t, guc=1.0, faz=0.0, yon=1, kacis=0.0):
    """Kambur, pençeli karanlık yaratık; kızıl gözler. kacis: 0-1 kaçarken öne eğilme."""
    k = Kalem(u, v, boy, yon)
    e = 0.15 * kacis
    s_ = 0.02 * math.sin(6 * t + faz) * (1 + kacis)
    k.elips(0.10 + e, -0.62 + e * 0.3, 0.07, 0.06)
    k.poligon([(0.04 + e, -0.68), (0.10 + e, -0.74), (0.12 + e, -0.80), (0.14 + e, -0.72)])
    k.poligon(bezier((0.14 + e, -0.62), (0.02, -0.80), (-0.20, -0.70), (-0.24, -0.40), 10)
              + bezier((-0.24, -0.40), (-0.22, -0.20), (-0.10, -0.12), (0.06, -0.20), 8))
    for i, x in enumerate((-0.16, -0.06, 0.04)):
        a = math.sin(8 * t + faz + i * 2) * (0.3 + kacis)
        uzuv(k, [(x, -0.25), (x + 0.04 * a, -0.14), (x + 0.02 + 0.06 * a, -0.06), (x + 0.03 + 0.08 * a, 0.0)],
             0.03, 0.012)
    uzuv(k, [(0.08 + e, -0.55), (0.18 + e, -0.45 + s_), (0.22 + e, -0.38), (0.28 + e, -0.34)], 0.025, 0.01)
    isle(taban, k, KIZIL, isik=(-0.5, -0.8), guc=guc, dolgu=(0.005, 0.0, 0.0), kenar_guc=0.5)
    for d in (-1, 1):
        g = pikselle_yerel(u, v, boy, yon, 0.12 + e + 0.022 * d, -0.635)
        lekele(taban, g[0], g[1], 0.004 * boy, 0.003 * boy, KIZIL, 6 * guc)


def agac(taban, ayrinti, u, v, boy, t, buyume=1.0, renk=(1.0, 0.85, 0.45), guc=1.0, tohum=3, yaprak=True):
    """Dallanarak büyüyen dev ağaç (parlayan kenarlı siluet); buyume 0 → 1."""
    if guc <= 0.01 or buyume <= 0:
        return
    k = Kalem(u, v, boy)
    rng = np.random.default_rng(tohum)
    uclar = []

    def dal(x, y, aci, uz, kal, derin):
        f = np.clip(buyume * 5 - derin, 0, 1)
        if f <= 0:
            return
        x2 = x + math.cos(aci) * uz * f
        y2 = y + math.sin(aci) * uz * f
        orta = (x + (x2 - x) * 0.5 + 0.02 * math.sin(derin + aci * 3), y + (y2 - y) * 0.5)
        k.konik(bezier((x, y), orta, orta, (x2, y2), 8), kal, kal * 0.7)
        if derin >= 4:
            uclar.append((x2, y2, f))
            return
        for d in (-1, 1):
            dal(x2, y2, aci + d * rng.uniform(0.35, 0.65) + 0.03 * math.sin(t * 0.8 + derin),
                uz * rng.uniform(0.68, 0.8), kal * 0.66, derin + 1)

    for d in (-1, 1):
        k.konik(bezier((0.0, 0.0), (0.05 * d, -0.02), (0.14 * d, 0.0), (0.22 * d, 0.02), 6), 0.05, 0.01)
    dal(0.0, 0.0, -math.pi / 2, 0.30, 0.055, 0)
    isle(taban, k, renk, isik=(0.0, -1.0), guc=guc, dolgu=(0.02, 0.018, 0.01), kenar_guc=1.2)
    if yaprak:
        for x, y, f in uclar:
            c = pikselle_yerel(u, v, boy, 1, x, y)
            lekele(taban, c[0], c[1], 0.035 * boy, 0.03 * boy, np.asarray(renk, np.float32), 0.35 * f * guc)
