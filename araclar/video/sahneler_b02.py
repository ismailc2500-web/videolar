"""Bölüm 2 — Eä: Dünya Var Oluyor: sahne çizimleri (karakterli anlatım)."""
import math

import cv2
import numpy as np

import karakterler as K
from motor import (ALTIN, FPS, GUMUS, H, KIZIL, KOR, MERKEZ, SICAK, W, h, halka, isinlar, lekele, parcaciklar,
                   pencere, puruzsuz, seritler, w)

GORU = (0.0, -0.13)
YESIL = np.array([0.45, 1.0, 0.45], np.float32)
BEYAZ = np.array([0.95, 0.95, 1.0], np.float32)
VALA_RENK = [K.GOK_MAVI, K.DENIZ_YESIL, K.KOR, YESIL, BEYAZ]
V_KOLON = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]


def hazirla(c):
    c.B.yildiz = K.Yildizlar(900, 21)
    c.B.yildiz_uzak = K.Yildizlar(500, 33)
    d = c.B.doku[1]
    sirt = 1 - np.abs(d * 2 - 1)
    c.B.sirt_doku = np.clip((sirt - 0.90) * 10, 0, 1).astype(np.float32)
    c.B.lav_doku = np.clip((sirt - 0.955) * 30, 0, 1).astype(np.float32)
    rng = np.random.default_rng(17)
    uu, vv = rng.uniform(-0.28, 0.28, 900), rng.uniform(-0.5, 0.5, 900)
    r = np.sqrt((uu / 0.2812) ** 2 + (vv / 0.5) ** 2)
    c.B.buz_nokta = np.stack([uu, vv, r, rng.uniform(0, 6.28, 900)], -1)[r > 0.85][:260]


# ------------------------------------------------------------------ yardımcılar

def sis(taban, v, kalinlik, renk, guc):
    if guc > 0.01:
        taban += (np.exp(-((V_KOLON - v) / kalinlik) ** 2)[..., None] * np.asarray(renk, np.float32) * guc)


def zemin(taban, v_ufuk, ust, alt, guc=1.0):
    y0 = int(h / 2 + v_ufuk * h)
    if y0 >= h or guc <= 0.01:
        return
    d = np.linspace(0, 1, h - y0, dtype=np.float32)[:, None, None]
    renk = np.asarray(ust) * (1 - d) + np.asarray(alt) * d
    taban[y0:] = taban[y0:] * (1 - guc) + renk * guc


def pikselle(u, v):
    return W / 2 + u * H, H / 2 + v * H


def kivilcim(ayrinti, u, v, t_bas, t, renk=KOR, adet=36, hiz=0.35, tohum=0, guc=1.0):
    tau = t - t_bas
    if tau < 0 or tau > 1.0 or guc <= 0:
        return
    rng = np.random.default_rng(tohum)
    x0, y0 = pikselle(u, v)
    for _ in range(adet):
        a = rng.uniform(-math.pi * 0.95, -math.pi * 0.05)
        s = rng.uniform(0.3, 1.0) * hiz * H
        x = x0 + math.cos(a) * s * tau
        y = y0 + math.sin(a) * s * tau + 0.6 * H * tau * tau
        px = x - math.cos(a) * s * 0.03
        py = y - (math.sin(a) * s + 1.2 * H * tau) * 0.03
        b = (1 - tau) ** 1.5 * guc
        cv2.line(ayrinti, (int(px * 16), int(py * 16)), (int(x * 16), int(y * 16)),
                 tuple(float(c) for c in (renk * 0.6 + 0.4) * 255 * b), 2, cv2.LINE_AA, 4)


def patlama(ayrinti, u, v, t_bas, t, renk, adet=260, hiz=0.9, omur=3.0, tohum=0, guc=1.0):
    tau = t - t_bas
    if tau < 0 or tau > omur or guc <= 0:
        return
    rng = np.random.default_rng(tohum)
    x0, y0 = pikselle(u, v)
    for _ in range(adet):
        a = rng.uniform(0, 2 * math.pi)
        s = rng.uniform(0.15, 1.0) ** 0.7 * hiz * H
        yol = s * (1 - math.exp(-tau * 2.2)) / 2.2
        x, y = x0 + math.cos(a) * yol, y0 + math.sin(a) * yol
        iz = s * math.exp(-tau * 2.2) * 0.05
        b = (1 - tau / omur) ** 1.2 * guc * rng.uniform(0.4, 1.0)
        renk_c = tuple(float(c) for c in np.clip((np.asarray(renk) * 0.5 + 0.5) * 255 * b, 0, 255))
        cv2.line(ayrinti, (int((x - math.cos(a) * iz) * 16), int((y - math.sin(a) * iz) * 16)),
                 (int(x * 16), int(y * 16)), renk_c, 2, cv2.LINE_AA, 4)


def ruzgar(ayrinti, t, guc, adet=45, tohum=5):
    if guc <= 0.01:
        return
    rng = np.random.default_rng(tohum)
    for _ in range(adet):
        y = rng.uniform(0.05, 0.75) * H
        hiz = rng.uniform(300, 700)
        uz = rng.uniform(40, 160)
        x = (rng.uniform(0, W + 400) + t * hiz) % (W + 400) - 200
        b = rng.uniform(0.15, 0.5) * guc
        cv2.line(ayrinti, (int((x - uz) * 16), int(y * 16)), (int(x * 16), int((y + uz * 0.08) * 16)),
                 (200 * b, 225 * b, 255 * b), 1, cv2.LINE_AA, 4)


def buz(taban, ayrinti, B, t, p):
    if p <= 0.01:
        return
    r = np.sqrt((B.U / 0.2812) ** 2 + (B.V / 0.5) ** 2)
    maske = np.clip((r - (1.30 - 0.38 * p) + 0.10 * (B.doku[0][:h, :w] - 0.5)) * 4, 0, 1)
    maske = cv2.GaussianBlur(maske, (0, 0), 4)
    taban *= (1 - 0.3 * maske)[..., None] * np.array([1 - 0.12 * p, 1 - 0.04 * p, 1.0], np.float32)
    taban += (maske * 0.30)[..., None] * K.BUZ
    esik = 1.30 - 0.38 * p
    for uu, vv, rr, f in B.buz_nokta:
        if rr < esik - 0.05:
            continue
        b = (0.5 + 0.5 * math.sin(t * 3 + f)) * p * 255
        x, y = pikselle(uu, vv)
        cv2.circle(ayrinti, (int(x * 16), int(y * 16)), 2, (b * 0.85, b * 0.95, b), -1, cv2.LINE_AA, 4)


def lav_catlak(taban, B, v_ufuk, guc, yayilma=1.0, merkez_u=0.0):
    if guc <= 0.01:
        return
    y0 = int(h / 2 + v_ufuk * h)
    c = B.lav_doku[y0:h, :w]
    uzak = np.abs(B.U[y0:h] - merkez_u) + np.abs(B.V[y0:h] - v_ufuk) * 2
    maske = c * np.clip((yayilma * 0.6 - uzak) * 6, 0, 1)
    taban[y0:h] += (cv2.GaussianBlur(maske, (0, 0), 0.8) * 2.0 + cv2.GaussianBlur(maske, (0, 0), 4) * 0.8)[..., None] * KOR * guc


def huzme(taban, ayrinti, u0, v0, u1, v1, t, renk, guc, tohum):
    if guc <= 0.01:
        return
    n = 40
    f = np.linspace(0, 1, n)
    du, dv = u1 - u0, v1 - v0
    uz = math.hypot(du, dv)
    nu, nv = -dv / uz, du / uz
    maske = np.zeros((h, w), np.uint8)
    for i in range(2):
        dalga = 0.012 * np.sin(f * 18 + t * 25 + tohum + i * 2.1) * np.sin(f * math.pi)
        uu = u0 + du * f + nu * dalga
        vv = v0 + dv * f + nv * dalga
        yarim = np.stack([w / 2 + uu * h, h / 2 + vv * h], -1)
        cv2.polylines(maske, [np.int32(yarim * 16)], False, 255, 3, cv2.LINE_AA, 4)
        tam = np.stack([W / 2 + uu * H, H / 2 + vv * H], -1)
        cv2.polylines(ayrinti, [np.int32(tam * 16)], False, tuple(float(c) for c in (renk * 0.4 + 0.6) * 255 * guc),
                      2, cv2.LINE_AA, 4)
    m = maske.astype(np.float32) / 255.0
    taban += (cv2.GaussianBlur(m, (0, 0), 3) * 1.6 + cv2.GaussianBlur(m, (0, 0), 10) * 1.2)[..., None] * renk * guc


def goz(taban, u, v, s, guc):
    for d in (-1, 1):
        lekele(taban, u + d * 0.075 * s, v, 0.03 * s, 0.009 * s, KIZIL, 1.8 * guc)
        lekele(taban, u + d * 0.075 * s, v, 0.012 * s, 0.004 * s, KOR, 4.0 * guc)
    lekele(taban, u, v, 0.25 * s, 0.10 * s, np.array([0.5, 0.02, 0.0], np.float32), 0.35 * guc)


def alev(taban, u, v, boy, t, guc):
    if guc <= 0.01:
        return
    tit = 1 + 0.12 * math.sin(t * 17) * math.sin(t * 5.3)
    lekele(taban, u, v - 0.35 * boy, 0.22 * boy, 0.45 * boy * tit, KOR, 0.8 * guc)
    lekele(taban, u, v - 0.25 * boy, 0.14 * boy, 0.32 * boy * tit, ALTIN, 1.4 * guc)
    lekele(taban, u, v - 0.15 * boy, 0.07 * boy, 0.16 * boy * tit, SICAK, 3.5 * guc)
    lekele(taban, u, v - 0.2, 0.9 * boy, 0.9 * boy, SICAK, 0.25 * guc)


def gezegen(taban, B, t, guc, r, merkez=MERKEZ):
    """Okyanus, kıta, buzul, bulut ve atmosferiyle dönen dünya."""
    if guc <= 0.01:
        return
    cu, cv = merkez
    x0, x1 = max(0, int(w / 2 + (cu - r * 1.35) * h)), min(w, int(w / 2 + (cu + r * 1.35) * h))
    y0, y1 = max(0, int(h / 2 + (cv - r * 1.35) * h)), min(h, int(h / 2 + (cv + r * 1.35) * h))
    uu = (B.U[y0:y1, x0:x1] - cu) / r
    vv = (B.V[y0:y1, x0:x1] - cv) / r
    d2 = uu * uu + vv * vv
    ic = (d2 < 1).astype(np.float32)
    zz = np.sqrt(np.clip(1 - d2, 0, 1))
    kosinus = np.sqrt(np.clip(1 - vv * vv, 1e-4, 1))
    boylam = (np.arcsin(np.clip(uu / kosinus, -1, 1)) / np.pi + 0.5) * 0.5 + 0.015 * t
    enlem = np.arcsin(np.clip(vv, -1, 1)) / np.pi + 0.5
    iy = np.clip((enlem * 1399).astype(int), 0, 1399)
    kara = B.doku[0][iy, ((boylam % 1.0) * 1399).astype(int)]
    bulut = B.doku[1][iy, (((boylam * 1.3 + 0.02 * t) % 1.0) * 1399).astype(int)]
    kara_m = np.clip((kara - 0.56) * 14, 0, 1)
    dag = np.clip((kara - 0.66) * 6, 0, 1)
    kutup = np.clip((np.abs(vv) - 0.80) * 12, 0, 1)
    bulut_m = np.clip((bulut - 0.55) * 4, 0, 1) * 0.8
    okyanus = np.array([0.03, 0.10, 0.30]) + np.array([0.02, 0.10, 0.18]) * zz[..., None]
    toprak = np.array([0.16, 0.26, 0.10]) * (1 - dag[..., None]) + np.array([0.36, 0.30, 0.20]) * dag[..., None]
    yuzey = okyanus * (1 - kara_m[..., None]) + toprak * kara_m[..., None]
    yuzey = yuzey * (1 - kutup[..., None]) + 0.85 * kutup[..., None]
    yuzey = yuzey * (1 - bulut_m[..., None]) + 0.95 * bulut_m[..., None]
    isik = np.clip(-0.55 * uu - 0.45 * vv + 0.70 * zz, 0, 1)
    parlama = np.clip(-0.55 * uu - 0.45 * vv + 0.70 * zz, 0, 1) ** 30 * (1 - kara_m) * (1 - bulut_m) * 0.8
    renk = (yuzey * (0.06 + 1.25 * isik[..., None]) + parlama[..., None]) * ic[..., None]
    kenar = np.exp(-((np.sqrt(d2) - 1) / 0.045) ** 2) * (0.35 + 1.0 * np.clip(-uu - vv + 0.3, 0, 1))
    renk += kenar[..., None] * np.array([0.35, 0.60, 1.0])
    renk += (np.clip(1 - d2 / 1.8, 0, 1) ** 2 * 0.12)[..., None] * np.array([0.3, 0.5, 1.0])
    taban[y0:y1, x0:x1] += (renk * guc).astype(np.float32)


def ainur_arkadan(taban, t, guc, parlaklik=1.0, titreme=0.0):
    konumlar = [(-0.25, 0.37, 0.34), (-0.165, 0.35, 0.31), (-0.085, 0.36, 0.30), (0.0, 0.345, 0.29),
                (0.085, 0.36, 0.30), (0.165, 0.35, 0.31), (0.25, 0.37, 0.34)]
    renkler = [K.DENIZ_YESIL, K.GOK_MAVI, ALTIN, BEYAZ, K.KOR, YESIL, GUMUS]
    for i, (u, v, b) in enumerate(konumlar):
        tt = 1.0 + titreme * 0.5 * math.sin(t * (7 + i) + i * 1.7)
        K.ainu(taban, u, v, b, t, renkler[i], guc=guc * 0.55 * parlaklik * tt, faz=i, kollar=0.1, hale=True)


# ------------------------------------------------------------------ sahneler

def s_ruya(taban, ayrinti, B, t, a):
    z = B.z
    ruya = puruzsuz((t - z.c(0, 1)) / 1.2)
    gercek_an = z.c(0, 2) + 1.5
    gercek = puruzsuz((t - gercek_an) / 0.5)
    seritler(taban, ayrinti, B, t, 0.8 * a * (1 - ruya), adet=10, acilis=puruzsuz(t / 1.8))
    parcaciklar(ayrinti, B, t, "sarmal", ALTIN, 0.8 * a * ruya, 500)
    if ruya * (1 - gercek) > 0.01:
        gecici = np.zeros_like(taban)
        gezegen(gecici, B, t, 1.0, 0.16)
        taban += cv2.GaussianBlur(gecici, (0, 0), 5) * (0.55 * ruya * (1 - gercek) * a)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.22, 0.22, np.array([0.5, 0.45, 0.8], np.float32), 0.35 * ruya * (1 - gercek) * a)
    if gercek > 0:
        B.yildiz.ciz(ayrinti, t, gercek * a)
        gezegen(taban, B, t, gercek * a, 0.17)
        halka(taban, B, MERKEZ[0], MERKEZ[1], 0.17 + 0.6 * (t - gercek_an), 0.02, SICAK, 1.2 * max(0, 1 - (t - gercek_an)) * a)


def s_goru(taban, ayrinti, B, t, a, goru_guc=1.0):
    z = B.z
    bas = z.sinir[1]
    r = 0.10 + 0.05 * puruzsuz((t - bas) / 10.0)
    tarih = puruzsuz((t - z.c(1, 2)) / 1.5)
    lekele(taban, GORU[0], GORU[1], r * 1.7, r * 1.7, np.array([0.3, 0.45, 0.9], np.float32), 0.35 * a * goru_guc)
    gezegen(taban, B, t, 0.9 * a * goru_guc, r, merkez=GORU)
    if tarih > 0:
        rng = np.random.default_rng(3)
        for i in range(24):
            aci, rr = rng.uniform(0, 6.28), math.sqrt(rng.uniform(0, 0.8)) * r
            parla = max(0.0, math.sin(t * rng.uniform(2, 5) + i)) * tarih
            lekele(taban, GORU[0] + rr * math.cos(aci), GORU[1] + rr * math.sin(aci) * 0.9, 0.003, 0.003, SICAK,
                   2.5 * parla * a * goru_guc)
        parcaciklar(ayrinti, B, t, "sarmal", SICAK, 0.5 * tarih * a * goru_guc, 300, merkez=GORU)
    ainur_arkadan(taban, t, a)


def s_cocuklar(taban, ayrinti, B, t, a):
    z = B.z
    ufuk = 0.12
    kiskanc = puruzsuz((t - z.c(2, 4)) / 0.8)
    K.gok(taban, (0.004, 0.008, 0.035), (0.04, 0.06, 0.14), ufuk, a * (1 - 0.5 * kiskanc))
    for i in range(7):
        lekele(taban, -0.25 + i * 0.08, -0.42 + i * 0.05, 0.07, 0.03, np.array([0.3, 0.35, 0.6], np.float32), 0.10 * a)
    B.yildiz.ciz(ayrinti, t, a * (1 - 0.4 * kiskanc), y_sinir=H * 0.6)
    if kiskanc > 0:
        lekele(taban, 0, -0.35, 0.45, 0.18, np.array([0.35, 0.02, 0.0], np.float32), 0.6 * kiskanc * a)
        goz(taban, 0.0, -0.27, 1.1, kiskanc * a * (0.85 + 0.15 * math.sin(t * 6)))
    K.sirt(taban, 0.17, 0.11, 3, (0.012, 0.018, 0.045), K.AY_ISIGI, 0.35, guc=a)
    sis(taban, 0.18, 0.03, (0.10, 0.12, 0.22), 0.5 * a)
    K.sirt(taban, 0.25, 0.07, 9, (0.006, 0.009, 0.022), K.AY_ISIGI, 0.25, guc=a)
    sis(taban, 0.26, 0.025, (0.08, 0.09, 0.16), 0.4 * a)
    K.sirt(taban, 0.335, 0.035, 12, (0.002, 0.003, 0.008), K.AY_ISIGI, 0.5, guc=a, sivri=1.5)
    inis = z.c(2, 0) + 2.2
    K.isik_sutunu(taban, 0.0, -0.5, 0.33, 0.05, SICAK, 0.9 * pencere(t, inis - 0.3, z.c(2, 1) + 0.8, 0.5) * a)
    elf_g = puruzsuz((t - z.c(2, 1) + 0.1) / 0.7) * a
    ins_g = puruzsuz((t - z.c(2, 1) - 0.7) / 0.7) * a
    sevgi = pencere(t, z.c(2, 3) - 0.2, z.c(2, 4) + 0.2, 0.6)
    ucuncu = pencere(t, z.c(2, 2), z.c(2, 3) + 0.3, 0.6)
    lekele(taban, 0.0, 0.18, 0.28, 0.12, ALTIN, (0.25 * ucuncu + 0.35 * sevgi) * a)
    if sevgi > 0:
        isinlar(taban, B, t, SICAK, 0.5 * sevgi * a, 0.4, merkez=(0.0, -0.5))
        parcaciklar(ayrinti, B, t, "dus", ALTIN, 0.8 * sevgi * a, 250, t0=z.c(2, 3))
    isik = (-0.5, -0.85)
    renk = K.AY_ISIGI * (1 - kiskanc) + KIZIL * 0.7 * kiskanc
    K.elf(taban, -0.215, 0.315 + 0.02 * (1 - elf_g), 0.37, t, renk=renk, isik=isik, guc=elf_g)
    K.elf(taban, -0.125, 0.32 + 0.02 * (1 - elf_g), 0.345, t, renk=renk, isik=isik, guc=elf_g, kadin=True, faz=1.3)
    K.elf(taban, -0.035, 0.325 + 0.02 * (1 - elf_g), 0.33, t, renk=renk, isik=isik, guc=elf_g, faz=2.1, yon=-1)
    K.insan(taban, 0.085, 0.33 + 0.02 * (1 - ins_g), 0.30, t, renk=renk, isik=isik, guc=ins_g, tur="adam")
    K.insan(taban, 0.175, 0.33 + 0.02 * (1 - ins_g), 0.285, t, renk=renk, isik=isik, guc=ins_g, tur="kadin", faz=0.7)
    K.insan(taban, 0.24, 0.335 + 0.02 * (1 - ins_g), 0.165, t, renk=renk, isik=isik, guc=ins_g, tur="cocuk", faz=1.9)


def _ulmo_denizi(taban, ayrinti, B, t, g, t0, u=0.0, boy=0.62, soguk=0.0, ufuk=0.12):
    ust = np.array([0.0, 0.02, 0.04]) * (1 - soguk) + np.array([0.06, 0.08, 0.12]) * soguk
    alt = np.array([0.02, 0.09, 0.12]) * (1 - soguk) + np.array([0.20, 0.26, 0.34]) * soguk
    K.gok(taban, ust, alt, ufuk, g)
    B.yildiz.ciz(ayrinti, t, g * (1 - 0.6 * soguk), y_sinir=H * (0.5 + ufuk))
    yukselis = 1 - puruzsuz((t - t0) / 2.0)
    boynuz = puruzsuz((t - t0 - 1.2) / 1.0)
    K.ulmo(taban, u, ufuk + 0.30 * yukselis, boy, t, guc=g, boynuz=0.3 + 0.7 * boynuz)
    K.deniz(taban, ayrinti, ufuk, t, renk=(0.01, 0.05, 0.08) if not soguk else (0.06, 0.09, 0.13),
            yansima=K.DENIZ_YESIL, yansima_u=u, guc=g)
    sis(taban, ufuk + 0.005, 0.012, (0.4, 0.8, 0.8), 0.25 * g)
    if yukselis > 0.02:
        x, y = pikselle(u, ufuk)
        rng = np.random.default_rng(int(t * 12))
        for _ in range(40):
            dx = rng.normal(0, 0.12 * boy * H)
            dy = -abs(rng.normal(0, 0.05 * boy * H)) * yukselis
            b = 255 * g * yukselis * rng.uniform(0.3, 1)
            cv2.circle(ayrinti, (int((x + dx) * 16), int((y + dy) * 16)), int(rng.integers(1, 3)), (b, b, b), -1, cv2.LINE_AA, 4)


def s_unsurlar(taban, ayrinti, B, t, a):
    z = B.z
    k = [z.c(3, i) for i in range(7)] + [z.sinir[4] + 0.5]
    g = [pencere(t, k[i] - 0.35, k[i + 1] + 0.05, 0.35) * a for i in range(7)]
    if g[0] > 0:
        for j, (u, renk) in enumerate(((-0.17, K.DENIZ_YESIL), (0.0, K.GOK_MAVI), (0.17, K.KOR))):
            parla = 0.6 + 0.6 * math.exp(-((t - k[0] - 0.6 - j * 0.8) / 0.4) ** 2)
            K.ainu(taban, u, 0.30, 0.34, t, renk, guc=g[0] * 0.7 * parla, faz=j, kollar=0.4)
            lekele(taban, u, 0.0, 0.10, 0.20, renk, 0.25 * g[0])
    if g[1] > 0:
        _ulmo_denizi(taban, ayrinti, B, t, g[1], k[1] - 0.3)
    if g[2] > 0:
        K.gok(taban, (0.02, 0.05, 0.14), (0.30, 0.42, 0.62), 0.25, g[2])
        for i in range(5):
            lekele(taban, -0.3 + i * 0.15 + 0.02 * math.sin(t * 0.3 + i), 0.24 + 0.02 * (i % 2), 0.12, 0.03,
                   np.array([0.6, 0.7, 0.85], np.float32), 0.45 * g[2])
        K.sirt(taban, 0.46, 0.22, 7, (0.02, 0.03, 0.07), K.GOK_MAVI, 0.9, guc=g[2], sivri=0.6, kayma=0.02)
        sis(taban, 0.30, 0.03, (0.5, 0.6, 0.75), 0.35 * g[2])
        K.manwe(taban, 0.02, 0.265, 0.40, t, guc=g[2], isaret=0.35)
        for j in range(3):
            aci = t * 0.6 + j * 2.1
            K.kartal(taban, 0.02 + 0.20 * math.cos(aci), -0.26 + 0.07 * math.sin(aci), 0.10 + 0.03 * math.sin(aci),
                     t, guc=g[2], faz=j)
        ruzgar(ayrinti, t, g[2])
    if g[3] > 0:
        K.gok(taban, (0.025, 0.01, 0.006), (0.07, 0.025, 0.01), 0.3, g[3])
        rng = np.random.default_rng(8)
        for i in range(30):
            renk = [K.GOK_MAVI, YESIL, np.array([1.0, 0.3, 0.5], np.float32), SICAK][i % 4]
            lekele(taban, rng.uniform(-0.28, 0.28), rng.uniform(-0.45, 0.15), 0.004, 0.004, renk,
                   (0.8 + 0.6 * math.sin(t * 3 + i)) * g[3])
        zemin(taban, 0.31, (0.03, 0.012, 0.006), (0.01, 0.004, 0.002), g[3])
        faz = ((t - k[3]) % 1.1) / 1.1
        vurus = puruzsuz((faz - 0.55) / 0.12) * (1 - puruzsuz((faz - 0.75) / 0.25))
        K.aule(taban, -0.13, 0.31, 0.40, t, guc=g[3], vurus=vurus)
        ou, ov = -0.13 + 0.36 * 0.40, 0.31 - 0.445 * 0.40
        for n in range(4):
            vurus_an = k[3] + n * 1.1 + 0.62 * 1.1
            kivilcim(ayrinti, ou, ov, vurus_an, t, tohum=n, guc=g[3])
            lekele(taban, ou, ov, 0.08, 0.05, KOR, 1.5 * math.exp(-max(0, t - vurus_an) / 0.15) * (t >= vurus_an) * g[3])
    soguk = puruzsuz((t - k[4]) / 2.5)
    if g[4] + g[5] + g[6] > 0:
        gs = max(g[4], g[5], g[6])
        if g[4] > 0:
            _ulmo_denizi(taban, ayrinti, B, t, g[4], k[1] - 10, u=0.0, boy=0.62, soguk=soguk)
            goz(taban, 0.0, -0.36, 0.7, 0.6 * g[4] * pencere(t, k[4], k[5], 0.6))
        if g[5] + g[6] > 0:
            g56 = max(g[5], g[6])
            K.gok(taban, (0.02, 0.04, 0.08), (0.10, 0.16, 0.24), 0.18, g56)
            B.yildiz.ciz(ayrinti, t, 0.5 * g56, y_sinir=H * 0.68)
            ulmo_g = g[6]
            K.ulmo(taban, -0.17, 0.20, 0.50, t, guc=ulmo_g, boynuz=0.0)
            K.deniz(taban, ayrinti, 0.18, t, renk=(0.05, 0.09, 0.13), yansima=K.BUZ, yansima_u=0.06, guc=g56)
            buyume = puruzsuz((t - k[5] - 0.3) / 2.8)
            ku = -0.0 * (1 - g[6]) + 0.07 * g[6]
            kv = -0.08 * (1 - g[6]) - 0.16 * g[6]
            K.kar_tanesi(taban, ayrinti, ku, kv, 0.16 - 0.04 * g[6], t, buyume=0.05 + 0.95 * buyume, guc=g56,
                         donus=0.1 * t)
            lekele(taban, ku, kv, 0.2, 0.2, K.BUZ, 0.25 * g56 * buyume)
            if g[6] > 0:
                lekele(taban, -0.17, -0.02, 0.15, 0.2, K.DENIZ_YESIL, 0.25 * g[6])
            parcaciklar(ayrinti, B, t, "dus", BEYAZ, 0.9 * g56, 500, t0=k[5])
        buz(taban, ayrinti, B, t, soguk * gs * (1 - 0.5 * g[6]))


def s_kaybolus(taban, ayrinti, B, t, a):
    z = B.z
    an = z.c(4, 0) + 1.6
    dagil = puruzsuz((t - an) / 0.5)
    s_goru_kure = 1 - dagil
    r = 0.15
    if s_goru_kure > 0:
        lekele(taban, GORU[0], GORU[1], r * 1.7, r * 1.7, np.array([0.3, 0.45, 0.9], np.float32), 0.35 * a * s_goru_kure)
        gezegen(taban, B, t, 0.9 * a * s_goru_kure, r, merkez=GORU)
    patlama(ayrinti, GORU[0], GORU[1], an, t, np.array([0.6, 0.75, 1.0]), adet=320, hiz=0.35, omur=3.5, tohum=4, guc=a)
    huzursuz = puruzsuz((t - z.c(4, 2)) / 1.0)
    ainur_arkadan(taban, t, a, parlaklik=1.0 - 0.45 * dagil, titreme=huzursuz)


def s_ea(taban, ayrinti, B, t, a):
    z = B.z
    ea = z.c(5, 1)
    once = t < ea
    alev_boy = 0.035 + 0.02 * puruzsuz((t - z.c(5, 0)) / 2.5)
    if once:
        alev(taban, MERKEZ[0], MERKEZ[1] + 0.02, alev_boy, t, a * puruzsuz((t - z.c(5, 0) + 0.2) / 1.0))
        return
    tau = t - ea
    evren = puruzsuz(tau / 2.5)
    B.nebula(taban, t, np.array([0.35, 0.15, 0.45], np.float32), 0.25 * evren * a)
    B.nebula(taban, t * 0.7 + 40, np.array([0.10, 0.25, 0.45], np.float32), 0.18 * evren * a)
    B.yildiz.ciz(ayrinti, t, evren * a, kayma=(0.0, -12 * tau))
    B.yildiz_uzak.ciz(ayrinti, t, 0.7 * evren * a, kayma=(0.0, -5 * tau))
    for j, (gu, gv, gs, egim) in enumerate(((-0.17, -0.30, 0.06, 0.45), (0.19, 0.06, 0.045, 0.6), (0.13, -0.40, 0.035, 0.35))):
        for kol in range(2):
            for i in range(26):
                aci = i * 0.24 + kol * math.pi + t * 0.04 + j
                rr = gs * (0.08 + i / 26)
                lekele(taban, gu + rr * math.cos(aci), gv + rr * math.sin(aci) * egim, 0.0035, 0.0035,
                       np.array([0.75, 0.75, 1.0], np.float32), 0.8 * (1 - i / 26) * evren * a)
        lekele(taban, gu, gv, gs * 0.12, gs * 0.12 * egim, SICAK, 1.2 * evren * a)
    halka(taban, B, MERKEZ[0], MERKEZ[1], 0.03 + 1.3 * (1 - math.exp(-tau / 0.6)), 0.02 + 0.03 * tau, SICAK,
          2.5 * math.exp(-tau / 1.0) * a)
    patlama(ayrinti, MERKEZ[0], MERKEZ[1], ea, t, SICAK, adet=380, hiz=1.1, omur=3.5, tohum=7, guc=a)
    kalp = 0.6 + 0.8 * puruzsuz((t - z.c(5, 3)) / 1.5)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.015, 0.015, SICAK, 4 * kalp * a)
    lekele(taban, MERKEZ[0], MERKEZ[1], 0.08, 0.08, ALTIN, 0.5 * kalp * a)
    isinlar(taban, B, t, SICAK, 0.5 * kalp * a, 0.18)


def _arda(taban, ayrinti, B, t, g, ufuk=0.13, kayma=0.0, kizil=0.0):
    K.gok(taban, (0.02, 0.008, 0.01), (0.14 + 0.25 * kizil, 0.05, 0.03), ufuk, g)
    B.yildiz_uzak.ciz(ayrinti, t, 0.35 * g * (1 - kizil), y_sinir=H * 0.45)
    for u, b, tohum in ((-0.22, 0.13, 1), (0.21, 0.10, 2), (0.02, 0.07, 3), (-0.05, 0.05, 4)):
        K.yanardag(taban, u + kayma, ufuk + 0.005, b, t, guc=g, patlama=0.8 + 0.4 * math.sin(t * 2 + tohum), tohum=tohum)
    zemin(taban, ufuk, (0.05, 0.02, 0.015), (0.012, 0.006, 0.005), g)
    lav_catlak(taban, B, ufuk, 0.5 * g, yayilma=1.0)
    sis(taban, ufuk + 0.01, 0.02, (0.35, 0.12, 0.06), 0.35 * g)
    parcaciklar(ayrinti, B, t, "kor", KOR, 0.6 * g, 250)


def s_valar(taban, ayrinti, B, t, a):
    z = B.z
    pan = 0.04 * puruzsuz((t - z.c(6, 3)) / 4.0)
    _arda(taban, ayrinti, B, t, a, kayma=-pan)
    insa = puruzsuz((t - z.c(6, 4) - 0.3) / 3.5)
    if insa > 0:
        K.sirt(taban, 0.14, 0.26, 21, (0.035, 0.018, 0.02), ALTIN, 1.6, guc=a, yukselme=insa, kayma=pan, sivri=0.8)
    bagli = pencere(t, z.c(6, 1), z.sinir[7], 0.6)
    guc_an = puruzsuz((t - z.c(6, 2)) / 0.6)
    for i, (u, zemin_v, vboy) in enumerate(((-0.22, 0.285, 0.24), (-0.11, 0.305, 0.265), (0.0, 0.29, 0.25),
                                             (0.11, 0.305, 0.265), (0.22, 0.285, 0.24))):
        u = u - pan * 0.5
        t_in = z.c(6, 0) + 0.3 + i * 0.45
        ilerleme = puruzsuz((t - t_in) / 0.8)
        if ilerleme <= 0:
            continue
        alt = -0.5 + (zemin_v + 0.5) * ilerleme
        sutun = 1.0 - 0.8 * puruzsuz((t - t_in - 0.9) / 1.0)
        K.isik_sutunu(taban, u, -0.5, alt, 0.018, VALA_RENK[i] * 0.6 + 0.4, 1.1 * sutun * a)
        indi = puruzsuz((t - t_in - 0.7) / 0.6)
        if indi > 0:
            tau = max(0.0, t - t_in - 0.7)
            lekele(taban, u, zemin_v, 0.03 + 0.15 * tau, 0.006 + 0.01 * tau, VALA_RENK[i], 2.0 * math.exp(-tau / 0.4) * a)
            lekele(taban, u, zemin_v, 0.06, 0.008, VALA_RENK[i], (0.6 + 0.6 * bagli) * indi * a)
            K.ainu(taban, u, zemin_v, vboy, t, VALA_RENK[i], guc=0.85 * indi * a * (1 + 0.3 * guc_an), faz=i,
                   kollar=0.1 + 0.9 * max(guc_an * 0.6, insa))
            if insa > 0:
                K.isik_sutunu(taban, u, -0.5, zemin_v - vboy, 0.006, VALA_RENK[i], 0.8 * insa * a)


def s_melkor_gelis(taban, ayrinti, B, t, a):
    z = B.z
    ufuk = 0.14
    gelis = puruzsuz((t - z.c(7, 0) + 0.2) / 2.2)
    ilan = puruzsuz((t - z.c(7, 1)) / 1.0)
    konus = puruzsuz((t - z.c(7, 2)) / 0.8)
    boz = z.c(7, 3)
    savas = puruzsuz((t - z.c(7, 4)) / 0.6)
    _arda(taban, ayrinti, B, t, a, ufuk=ufuk, kizil=0.6 + 0.4 * ilan)
    lekele(taban, -0.10, -0.15, 0.30, 0.35, np.array([0.45, 0.04, 0.0], np.float32), 0.5 * gelis * a)
    K.melkor_dev(taban, -0.10, ufuk + 0.10 + 0.42 * (1 - gelis), 0.66, t, guc=a, kor_guc=0.6 + 0.8 * ilan)
    zemin(taban, ufuk + 0.09, (0.03, 0.01, 0.01), (0.008, 0.004, 0.004), a)
    lav_catlak(taban, B, ufuk + 0.09, a, yayilma=0.2 + 1.2 * ilan, merkez_u=-0.10)
    if t > boz - 0.2:
        for n in range(2):
            t0 = boz + n * 1.2
            yuk = puruzsuz((t - t0) / 0.5)
            cok = puruzsuz((t - t0 - 0.6) / 0.4)
            if yuk > 0 and t < t0 + 1.4:
                K.sirt(taban, 0.30, 0.36, 30 + n, (0.03, 0.02, 0.03), ALTIN, 1.8 * (1 - cok), guc=a,
                       yukselme=yuk * (1 - 0.85 * cok), cokme=cok, t=t, kayma=0.1 * n, sivri=0.8)
                lekele(taban, -0.05 + 0.1 * n, 0.22, 0.25, 0.06, KOR, 1.2 * cok * (1 - cok) * 4 * a)
                kivilcim(ayrinti, -0.05 + 0.1 * n, 0.16, t0 + 0.6, t, renk=np.array([0.6, 0.35, 0.2]), adet=30,
                         hiz=0.25, tohum=20 + n, guc=a)
    lekele(taban, 0.20, 0.10, 0.18, 0.30, K.GOK_MAVI, 0.45 * konus * a)
    K.manwe(taban, 0.17, 0.34, 0.36, t, guc=a * konus, isaret=puruzsuz((t - z.c(7, 2) - 0.4) / 0.8), yon=-1)
    for j in range(2):
        aci = t * 0.7 + j * 3.1
        K.kartal(taban, 0.17 + 0.10 * math.cos(aci), -0.20 + 0.04 * math.sin(aci), 0.09, t, guc=a * konus, faz=j)
    if savas > 0:
        carpisma = (0.02, -0.08)
        huzme(taban, ayrinti, 0.112, -0.045, carpisma[0], carpisma[1], t, K.GOK_MAVI, savas * a, 1)
        huzme(taban, ayrinti, -0.10, -0.25, carpisma[0], carpisma[1], t, KIZIL, savas * a, 2)
        lekele(taban, carpisma[0], carpisma[1], 0.04, 0.04, SICAK, (2.0 + math.sin(t * 25)) * savas * a)
        kivilcim(ayrinti, carpisma[0], carpisma[1], z.c(7, 4), t, renk=SICAK, adet=50, hiz=0.5, tohum=11, guc=a)
        kivilcim(ayrinti, carpisma[0], carpisma[1], z.c(7, 4) + 0.9, t, renk=SICAK, adet=40, hiz=0.4, tohum=12, guc=a)


SAHNELER = {"ruya": s_ruya, "goru": s_goru, "cocuklar": s_cocuklar, "unsurlar": s_unsurlar, "kaybolus": s_kaybolus,
            "ea": s_ea, "valar": s_valar, "melkor_gelis": s_melkor_gelis}
NEBULA = {"ruya": ((0.22, 0.15, 0.32), 0.07), "goru": ((0.14, 0.10, 0.28), 0.08), "cocuklar": ((0.05, 0.08, 0.2), 0.02),
          "unsurlar": ((0.05, 0.10, 0.15), 0.02), "kaybolus": ((0.10, 0.08, 0.20), 0.05), "ea": ((0.2, 0.1, 0.3), 0.0),
          "valar": ((0.30, 0.08, 0.03), 0.03), "melkor_gelis": ((0.40, 0.05, 0.02), 0.04)}


def son_islem(c, kare, t):
    z = c.z
    for an, guc in ((z.c(5, 1), 1.0), (z.c(0, 2) + 1.5, 0.55)):
        if t >= an:
            flas = guc * math.exp(-(t - an) / 0.3)
            if flas > 0.01:
                kare = cv2.addWeighted(kare, 1 - flas, np.full_like(kare, 255), flas, 0)
    titresim = pencere(t, z.c(7, 3), z.c(7, 4) + 1.2, 0.2) * 0.8 + pencere(t, z.c(5, 1), z.c(5, 1) + 0.8, 0.1) * 0.6
    if titresim > 0.02:
        rs = np.random.default_rng(int(t * FPS))
        M = np.float32([[1, 0, rs.uniform(-9, 9) * titresim], [0, 1, rs.uniform(-9, 9) * titresim]])
        kare = cv2.warpAffine(kare, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return kare
