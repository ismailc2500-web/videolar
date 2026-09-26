"""Bölüm 1 — Evren Bir Şarkıyla Başladı: sahne çizimleri."""
import math

import cv2
import numpy as np

from motor import *  # noqa: F401,F403
from motor import FPS, H, W


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



def son_islem(c, kare, t):
    z = c.z
    if "uyumsuzluk" in c.gorseller:
        i = c.gorseller.index("uyumsuzluk")
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

    return kare
