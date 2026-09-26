"""Bölüm 4 — Gandalf ve Sauron Aynı Türden mi?: sahne çizimleri.

Maiar portrelerle tanıtılır: Eönwë, Ossë, Uinen, Melian, Olórin (Gandalf'a
dönüşür) ve Mairon (gözleri kızıla dönerek Sauron olur). Balrog'lar gölge ve
alevden siluetlerdir; bölüm Khazad-dûm Köprüsü'ndeki karşılaşmayla doruğa ulaşır
ve iki Lamba'nın yanmasıyla biter.
"""
import math

import cv2
import numpy as np

import karakterler as K
import maiar as MA
import sahne_araclari as SA
import valar as VL
from motor import FPS, H, KOR, W, bindir, h, isinlar, lekele, pencere, puruzsuz, seritler, w
from sahne_araclari import (KAFA_V, KAFA_Y, S_PORTRE, V_KOLON, _Z, arka_isik, bulut, dikey, dunya, durum, goz_isigi,
                            ilerle, kesit, korlar, parilti, portre, yagmur)
from sahneler_b02 import kivilcim, sis

TEMA = {"gandalf": (0.75, 0.80, 0.95), "olorin": (0.70, 0.75, 0.95), "mairon": (1.0, 0.50, 0.15),
        "melian": (0.70, 0.72, 1.0), "eonwe": (0.45, 0.65, 1.0), "osse": (0.20, 0.85, 0.80),
        "uinen": (0.45, 0.90, 0.95)}
SA.TOHUM.update({"gandalf": 41, "olorin": 43, "mairon": 47, "melian": 53, "eonwe": 59, "osse": 61, "uinen": 67,
                 "aule": 38, "manwe": 3, "nienna": 45})
KIRMIZI_GOZ = (1.0, 0.18, 0.05)


# ------------------------------------------------------------------ olaylar

def olaylar(zd):
    """Görüntü ve müziğin paylaştığı olay zamanları."""
    z = _Z(zd)
    O = {}
    O["ayni"] = z.c(0, 0, 1)
    O["kanca_b"], O["kanca_c"] = z.c(0, 1) - 0.1, z.c(0, 2) - 0.1
    O["muzik"], O["sayi"], O["yardim"] = z.c(1, 1), z.c(1, 2), z.c(1, 3)
    O["yuzler"] = [z.c(2, k) - 0.15 for k in range(4)]
    O["nienna"], O["gelecek"], O["korku"] = z.c(3, 2), z.c(3, 3), z.c(3, 4)
    O["manwe"], O["donusum"], O["gandalf"] = z.c(3, 5), z.c(3, 6, 0), z.c(3, 6, 1)
    O["soru"], O["demirhane"], O["mairon"] = z.c(4, 0), z.c(4, 1), z.c(4, 2)
    O["dusus"], O["saruman"], O["yuzuk"] = z.c(4, 3), z.c(4, 4) - 0.1, z.c(4, 5) - 0.1
    O["ors"] = list(np.arange(z.c(4, 1) + 0.4, z.c(4, 2) - 0.2, 1.2)) + \
        list(np.arange(O["yuzuk"] + 0.5, z.e(4, 5) + 0.3, 1.0))
    O["cekim"], O["cikis"] = z.c(5, 0), z.c(5, 1) - 0.2
    O["kirbac"] = [z.c(5, 3) + 0.3, z.c(5, 3) + 1.7]
    O["yakin"], O["gecemezsin"] = z.c(6, 1) - 0.1, z.c(6, 2) - 0.1
    O["vurus"] = z.e(6, 2) + 0.2
    O["gandalf_dusus"] = O["vurus"] + 1.4
    O["ak"] = z.c(6, 4) - 0.1
    O["lamba"] = [z.c(7, 1) + 0.2, z.c(7, 1) + 1.5]
    return O


def hazirla(c):
    B = c.B
    B.portre = {ad: MA.olustur(ad, S=S_PORTRE) for ad in MA.MAIAR}
    for ad in ("aule", "manwe", "nienna"):
        B.portre[ad] = VL.olustur(ad, S=S_PORTRE)
    B.yildiz = K.Yildizlar(900, 21)
    zd = {"birimler": c.z.birim, "sahneler": c.z.sahne, "sure": c.z.sure}
    B.O, B.zz = olaylar(zd), _Z(zd)
    rng = np.random.default_rng(4)
    B.ruhlar = np.stack([rng.uniform(0, 2 * math.pi, 240), rng.uniform(0.3, 1.0, 240), rng.uniform(0.03, 0.30, 240),
                         rng.uniform(-0.25, 0.25, 240)], -1).astype(np.float32)
    B.adlar = []
    for ad in ("Eönwë", "Ossë", "Uinen", "Melian", "Olórin", "Mairon", "Ilmarë", "Arien", "Tilion", "Curumo"):
        B.adlar.append(ad_yazisi(c.yz, ad))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    B.sol_maske = kesik_maske(xs, ys, 0.62, 0.38)
    B.sag_maske = 1 - B.sol_maske


def ad_yazisi(yz, ad):
    """Küçük, altın renkli, parlayan ad yazısı (önceden çarpılmış RGBA)."""
    from PIL import Image, ImageDraw
    fnt = yz.font("cormorant", 60, 700)
    gen = int(fnt.getlength(ad)) + 40
    tuval = Image.new("L", (gen, 100), 0)
    ImageDraw.Draw(tuval).text((20, 10), ad, font=fnt, fill=255)
    m = np.asarray(tuval, np.float32) / 255.0
    hale = cv2.GaussianBlur(m, (0, 0), 6) * 0.8
    rgba = np.zeros(m.shape + (4,), np.float32)
    renk = np.array((1.0, 0.86, 0.55), np.float32)
    rgba[..., :3] = m[..., None] * renk + hale[..., None] * np.array((1.0, 0.6, 0.2), np.float32) * 0.6
    rgba[..., 3] = np.clip(m + hale * 0.5, 0, 1)
    return rgba


def kesik_maske(xs, ys, ust, alt, yumusak=3.0):
    """Çapraz bölünmüş ekranın sol yarısı (ust/alt: çizginin üst ve alt kenardaki x oranı)."""
    sinir = W * (ust + (alt - ust) * ys / H)
    return np.clip((sinir - xs) / yumusak + 0.5, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ yardımcılar

def dikis(B, t, guc, ust=0.62, alt=0.38, renk=(1.0, 0.9, 0.7)):
    """Bölünmüş ekranın ışıklı çapraz dikişi (ön katman)."""
    if guc <= 0.01:
        return
    kat = B.on_katman()
    p0, p1 = (int(W * ust * 16), 0), (int(W * alt * 16), H * 16)
    for kal, g in ((14, 0.12), (6, 0.3), (2, 1.0)):
        cv2.line(kat, p0, p1, tuple(float(c) * 255 * g * guc for c in renk), kal, cv2.LINE_AA, 4)


def ruh_bulutu(ayrinti, B, t, merkez, yaricap, renk, guc, adet=120, hiz=1.0, cekim=0.0, hedef=None, kizil=0.0):
    """Bir merkez etrafında dönen küçük ışık ruhları; cekim > 0 ise hedefe doğru sarmal çizerek çekilirler."""
    if guc <= 0.01:
        return
    r_ = B.ruhlar[:adet]
    aci = r_[:, 0] + t * hiz * r_[:, 1]
    r = yaricap * (0.3 + r_[:, 2] * 2.3)
    u = merkez[0] + r * np.cos(aci)
    v = merkez[1] + r * 0.55 * np.sin(aci) + r_[:, 3] * 0.2 * yaricap
    if cekim > 0 and hedef is not None:
        f = np.clip(cekim * (0.6 + 0.8 * r_[:, 1]) - r_[:, 2], 0, 1)[:, None]
        f = f * f * (3 - 2 * f)
        u = u * (1 - f[:, 0]) + hedef[0] * f[:, 0]
        v = v * (1 - f[:, 0]) + hedef[1] * f[:, 0]
        karis = f
    else:
        karis = np.zeros((adet, 1), np.float32)
    parla = (0.6 + 0.4 * np.sin(t * 3 + r_[:, 0] * 5)) * guc
    renk = np.asarray(renk, np.float32)
    kizil_renk = np.array((1.0, 0.25, 0.08), np.float32)
    for i in range(adet):
        c = renk * (1 - karis[i] * kizil) + kizil_renk * karis[i] * kizil
        X, Y = W / 2 + u[i] * H, H / 2 + v[i] * H
        b = float(parla[i])
        cv2.circle(ayrinti, (int(X * 16), int(Y * 16)), int((1.5 + 1.5 * r_[i, 1]) * 16),
                   tuple(float(x) * 255 * b for x in c), -1, cv2.LINE_AA, 4)


def kus(kat, X, Y, olcek, t, faz, renk, guc):
    """Kanat çırpan küçük kuş (bülbül) çizgisi."""
    cirp = math.sin(t * 9 + faz)
    L = 14 * olcek
    uc_y = -6 * olcek * cirp
    pts = np.array([[X - L, Y + uc_y], [X - L * 0.45, Y - 3 * olcek], [X, Y], [X + L * 0.45, Y - 3 * olcek],
                    [X + L, Y + uc_y]], np.float32)
    cv2.polylines(kat, [np.int32(pts * 16)], False, tuple(float(c) * 255 * guc for c in renk), 2, cv2.LINE_AA, 4)


def dalgalar(taban, ayrinti, t, v0, renk, kopuk, guc, siddet=1.0, tohum=0):
    """Ufuktan öne doğru yükselen dalga sıraları (deniz yüzeyi)."""
    if guc <= 0.01:
        return
    for j in range(7):
        derin = j / 6
        v = v0 + 0.03 + derin * derin * (0.5 - v0)
        x = np.linspace(-0.4, 0.4, 90)
        faz = t * (0.8 + derin) * (1 + siddet) + j * 1.7 + tohum
        y = v - (0.006 + 0.03 * derin) * siddet * (0.5 + 0.5 * np.sin(x * (18 - 8 * derin) + faz)) ** 2
        k = K.Kalem(0.0, 0.0, 1.0)
        k.poligon(list(zip(x, y)) + [(0.4, 0.6), (-0.4, 0.6)])
        K.isle(taban, k, kopuk, isik=(0.0, -1.0), guc=guc * (0.5 + 0.5 * derin),
               dolgu=tuple(np.asarray(renk) * (0.6 + 0.6 * derin)), kenar_guc=0.5 + 0.8 * siddet * derin)


def mairon_durumu(t, O, **ek):
    """Mairon → Sauron dönüşümü: gözler kızıla döner, gülümseme soğur, kaşlar çatılır."""
    f = puruzsuz((t - O["dusus"] - 0.8) / 2.2)
    goz = np.array(MA.MAIAR["mairon"]["goz"]) * (1 - f) + np.array(KIRMIZI_GOZ) * f
    d = durum("mairon", t, goz_renk=tuple(goz), goz_isima=0.25 + 0.9 * f, kas_catik=0.55 * f,
              gulus=ek.pop("gulus", 0.35 * (1 - f) + 0.25 * f), **ek)
    return d, f


# ------------------------------------------------------------------ 0 · kanca

def s_kanca(taban, ayrinti, B, t, a):
    O = B.O
    gA = a * kesit(t, -1, O["kanca_b"], 0.06)
    if gA > 0.01:
        ayni = puruzsuz((t - O["ayni"]) / 0.9)
        U = (np.arange(w, dtype=np.float32)[None, :] - w / 2) / h
        sinir = (0.62 - 0.24 * (V_KOLON + 0.5)) - 0.5
        sol = np.clip((sinir * w / h - U) * 40 + 0.5, 0, 1)[..., None]
        soguk = np.array((0.10, 0.12, 0.22), np.float32)
        sicak = np.array((0.30, 0.06, 0.02), np.float32)
        orta = (soguk + sicak) * 0.5
        taban += (sol * soguk + (1 - sol) * sicak) * (1 - ayni) * gA + orta * ayni * gA * 0.6
        lekele(taban, -0.10, -0.12, 0.2, 0.25, np.array((0.5, 0.6, 1.0), np.float32), 0.35 * gA * (1 - ayni))
        lekele(taban, 0.10, -0.12, 0.2, 0.25, KOR, 0.45 * gA * (1 - ayni))
        korlar(ayrinti, t, 0.5 * gA, adet=30, tohum=4, alan=(W * 0.5, W), yukselis=80)
        yuz_a = gA * (1 - ayni)
        dg = durum("gandalf", t, kas_catik=0.2, bakis=(0.4, 0.0))
        portre(B, "gandalf", t, dg, yuz_a, x=0.33 * W, y=0.42 * H, olcek=0.82, ton=(0.9, 0.95, 1.1),
               maske=B.sol_maske, alt=(0.60 * H, 1.2 * H), soldur=(0.74 * H, 0.92 * H))
        dm, _ = mairon_durumu(t, {"dusus": -10}, bakis=(-0.4, 0.0))
        portre(B, "mairon", t, dm, yuz_a, x=0.67 * W, y=0.42 * H, olcek=0.82, ton=(1.0, 0.6, 0.5),
               maske=B.sag_maske, alt=(0.60 * H, 1.2 * H), soldur=(0.74 * H, 0.92 * H))
        goz_isigi(B, "mairon", dm, 0.6 * yuz_a, KIRMIZI_GOZ, x=0.67 * W, y=0.42 * H, olcek=0.82)
        dikis(B, t, yuz_a)
        if ayni > 0.01:
            for i, x in enumerate((-0.09, 0.09)):
                xx = x * (1 - 0.35 * ayni)
                lekele(taban, xx, -0.10, 0.012 + 0.018 * ayni, 0.012 + 0.018 * ayni,
                       np.array((1.0, 0.85, 0.6), np.float32), 1.1 * ayni * gA)
                lekele(taban, xx, -0.10, 0.005, 0.005, np.array((1, 1, 1), np.float32), 3.5 * ayni * gA)
    gB = a * kesit(t, O["kanca_b"], O["kanca_c"], 0.06)
    if gB > 0.01:
        kopru_sahnesi(taban, ayrinti, B, t, gB, yakin=0.3)
    gC = a * kesit(t, O["kanca_c"], 999, 0.06)
    if gC > 0.01:
        yerel = t - O["kanca_c"]
        don = puruzsuz((yerel - 0.7) / 0.8)
        dikey(taban, [(-0.5, (0.02, 0.02, 0.05)), (0.0, (0.07, 0.05, 0.12)), (0.5, (0.01, 0.01, 0.02))], gC)
        B.yildiz.ciz(ayrinti, t, 0.6 * gC)
        K.gezgin(taban, -0.18, 0.18, 0.34, t, guc=gC * (1 - don), asa_isik=0.8)
        K.kara_lord(taban, 0.0, 0.18, 0.36, t, guc=gC * (1 - don))
        K.balrog(taban, ayrinti, 0.19, 0.18, 0.36, t, guc=gC * (1 - don), yon=-1, kanat=0.0)
        for u, ren in ((-0.18, (0.8, 0.85, 1.0)), (0.0, (1.0, 0.6, 0.3)), (0.19, (1.0, 0.5, 0.2))):
            r = np.asarray(ren, np.float32) * (1 - don) + np.array((1.0, 0.92, 0.75), np.float32) * don
            lekele(taban, u, 0.0 - 0.08 * don, 0.03 + 0.03 * don, 0.03 + 0.03 * don, r, 2.5 * don * gC)
            lekele(taban, u, 0.0 - 0.08 * don, 0.006, 0.006, np.array((1, 1, 1), np.float32), 5 * don * gC)


# ------------------------------------------------------------------ 1 · Maiar nedir?

def s_maiar(taban, ayrinti, B, t, a):
    O = B.O
    p = ilerle(B, 1, t)
    dikey(taban, [(-0.5, (0.02, 0.03, 0.10)), (0.0, (0.06, 0.06, 0.18)), (0.5, (0.02, 0.02, 0.05))], a)
    B.nebula(taban, t, np.array((0.25, 0.20, 0.45), np.float32), 0.25 * a)
    B.yildiz.ciz(ayrinti, t, 0.6 * a)
    yardim = puruzsuz((t - O["yardim"]) / 0.6)
    g1 = a * (1 - yardim)
    inis = 0.06 * (1 - puruzsuz(p * 3))
    for i, (u, v, boy, ren) in enumerate(((-0.17, 0.22, 0.55, (0.5, 0.7, 1.0)), (0.0, 0.26, 0.70, (1.0, 0.85, 0.5)),
                                          (0.17, 0.22, 0.55, (0.4, 1.0, 0.85)))):
        if g1 > 0.01:
            K.ainu(taban, u, v - inis, boy, t, np.array(ren, np.float32), guc=0.5 * g1, faz=i, kollar=0.3)
            ruh_bulutu(ayrinti, B, t + i * 3, (u, v - inis - boy * 0.6), boy * 0.35, ren, g1, adet=60, hiz=0.8)
    muz = pencere(t, O["muzik"] - 0.2, O["sayi"] + 0.3, 0.5)
    if muz > 0.01:
        seritler(taban, ayrinti, B, t, 0.8 * muz * a, adet=10, acilis=1.0)
    sayi = pencere(t, O["sayi"] - 0.2, O["yardim"] + 0.2, 0.4)
    if sayi > 0.01:
        ruh_bulutu(ayrinti, B, t, (0.0, -0.05), 0.28, (1.0, 0.9, 0.65), sayi * a, adet=240, hiz=0.5)
        rng = np.random.default_rng(14)
        for i, img in enumerate(B.adlar):
            gec = O["sayi"] + 0.25 + i * 0.28
            g = sayi * a * puruzsuz((t - gec) / 0.3) * (0.75 + 0.25 * math.sin(t * 2 + i))
            X = W * rng.uniform(0.18, 0.82)
            Y = H * (0.20 + 0.055 * i) + 12 * math.sin(t * 0.8 + i)
            if g > 0.01:
                B.on_plan.append(lambda kare, img=img, X=X, Y=Y, g=g: bindir(kare, img, Y, g, 1.0, X))
                parilti(B.on_katman(), X - img.shape[1] / 2 - 8, Y + 4, 5, (1.0, 0.9, 0.6), g)
    if yardim > 0.01:
        d = durum("manwe", t, gulus=0.3, bakis=(0.0, 0.2))
        portre(B, "manwe", t, d, a * yardim, olcek=0.9)
        arka_isik(taban, 0.0, KAFA_V, (0.4, 0.6, 1.0), 0.5 * a * yardim, 0.28)
        kat = B.on_katman()
        ruh_bulutu(kat, B, t, (0.0, KAFA_V + 0.05), 0.12, (1.0, 0.9, 0.6), a * yardim, adet=70, hiz=0.6)


# ------------------------------------------------------------------ 2 · Tanıdık Maiar

def s_tanidik(taban, ayrinti, B, t, a):
    O = B.O
    y = O["yuzler"] + [B.z.sinir[3] + 1]
    for k, ad in enumerate(("eonwe", "osse", "uinen", "melian")):
        g = a * kesit(t, y[k] if k else -1, y[k + 1] if k < 3 else 999, 0.08)
        if g <= 0.01:
            continue
        yerel = t - y[k]
        ol = 1.02 + 0.05 * puruzsuz(yerel / 4)
        if ad == "eonwe":
            dikey(taban, [(-0.5, (0.08, 0.18, 0.45)), (0.0, (0.30, 0.45, 0.75)), (0.5, (0.15, 0.20, 0.35))], g)
            lekele(taban, 0.2, -0.4, 0.2, 0.2, np.array((1.0, 0.95, 0.8), np.float32), 0.7 * g)
            bulut(taban, B, t, 0.1, 0.5, (0.9, 0.95, 1.0), 0.6 * g, hiz=18)
            for s in (-1, 1):
                sancak = [(0.20 * s, -0.45)] + [(0.20 * s + 0.12 * s * f, -0.45 + 0.02 * math.sin(t * 5 + f * 6) + 0.01 * f)
                                                for f in np.linspace(0, 1, 12)]
                sancak += [(0.20 * s + 0.12 * s * f, -0.30 + 0.02 * math.sin(t * 5 + f * 6) + 0.04 * f)
                           for f in np.linspace(1, 0, 12)]
                dunya(taban, sancak, (0.9, 0.95, 1.0), isik=(0.0, -1.0), dolgu=(0.25, 0.40, 0.85), guc=g, kenar_guc=0.8)
                dunya(taban, [(0.198 * s, -0.47), (0.204 * s, -0.47), (0.204 * s, 0.5), (0.198 * s, 0.5)],
                      (1.0, 0.9, 0.6), dolgu=(0.5, 0.45, 0.3), guc=g)
            d = durum(ad, t, gulus=0.3, ruzgar=1.2, kas_kalk=0.15)
            ton = (1.0, 1.0, 1.05)
        elif ad == "osse":
            simsek = max(0.0, math.sin(yerel * 7.0)) ** 30 * (yerel > 0.5)
            dikey(taban, [(-0.5, (0.03, 0.05, 0.08)), (0.0, (0.06, 0.10, 0.14)), (0.5, (0.01, 0.03, 0.04))],
                  g * (1 + 2.5 * simsek))
            bulut(taban, B, t, -0.5, -0.1, (0.25, 0.30, 0.35), 0.6 * g, hiz=40, k=1)
            dalgalar(taban, ayrinti, t, 0.05, (0.02, 0.08, 0.10), (0.7, 0.95, 1.0), g, siddet=1.4)
            yagmur(ayrinti, t, 0.5 * g, adet=90, tohum=6, egim=0.5)
            d = durum(ad, t, gulus=0.7, agiz=0.25, kas_catik=0.45, ruzgar=2.2)
            ton = (0.9 + 0.3 * simsek, 1.0 + 0.3 * simsek, 1.0 + 0.3 * simsek)
        elif ad == "uinen":
            dikey(taban, [(-0.5, (0.10, 0.10, 0.25)), (-0.05, (0.45, 0.35, 0.40)), (0.05, (0.20, 0.30, 0.40)),
                          (0.5, (0.02, 0.08, 0.12))], g)
            lekele(taban, 0.0, -0.02, 0.06, 0.02, np.array((1.0, 0.85, 0.6), np.float32), 1.2 * g)
            dalgalar(taban, ayrinti, t, 0.0, (0.04, 0.12, 0.16), (0.8, 0.95, 1.0), g, siddet=0.25)
            d = durum(ad, t, gulus=0.45, ruzgar=0.9, kapak=0.1)
            ton = (1.0, 1.0, 1.02)
        else:
            dikey(taban, [(-0.5, (0.02, 0.02, 0.06)), (0.0, (0.06, 0.06, 0.14)), (0.5, (0.01, 0.02, 0.03))], g)
            B.yildiz.ciz(ayrinti, t, 0.5 * g, y_sinir=0.4 * H)
            for i, u in enumerate((-0.27, -0.22, 0.22, 0.27)):
                dunya(taban, [(u - 0.03, 0.5), (u - 0.02 + 0.004 * math.sin(t + i), -0.5), (u + 0.02, -0.5), (u + 0.03, 0.5)],
                      (0.7, 0.75, 1.0), isik=(-1.0, 0.0) if u > 0 else (1.0, 0.0), dolgu=(0.03, 0.03, 0.06), guc=g,
                      kenar_guc=0.1)
            arka_isik(taban, 0.0, KAFA_V, (0.6, 0.65, 1.0), 0.45 * g, 0.3)
            sarki = 0.5 + 0.5 * math.sin(yerel * 2 * math.pi * 1.6)
            d = durum(ad, t, gulus=0.35, agiz=0.12 + 0.28 * sarki, kapak=0.12, ruzgar=0.7, bakis=(0.2, -0.3))
            ton = (0.98, 0.98, 1.08)
        portre(B, ad, t, d, g, olcek=ol, ton=ton)
        kat = B.on_katman()
        if ad == "osse":
            yagmur(kat, t, 0.6 * g, adet=90, tohum=8, egim=0.5)
        if ad == "melian":
            for i in range(7):
                aci = t * 0.6 + i * 2 * math.pi / 7
                X = W / 2 + math.cos(aci) * 0.33 * W
                Y = KAFA_Y - 60 + math.sin(aci) * 0.10 * H + 30 * math.sin(t * 2 + i)
                kus(kat, X, Y, 1.6 + 0.4 * math.sin(aci), t, i, (0.85, 0.88, 1.0), g * 0.9)
            for i in range(14):
                ilerle_ = ((t * 0.25 + i / 14) % 1.0)
                X = W / 2 + (i % 2 * 2 - 1) * (40 + ilerle_ * 380)
                Y = KAFA_Y + 110 - ilerle_ * 420 + 20 * math.sin(t * 3 + i)
                parilti(kat, X, Y, 3, (0.8, 0.85, 1.0), g * (1 - ilerle_) * 0.9)


# ------------------------------------------------------------------ 3 · Olórin

def lorien(taban, ayrinti, B, t, g, karanlik=0.0):
    dikey(taban, [(-0.5, (0.06, 0.07, 0.18)), (0.0, (0.14, 0.14, 0.30)), (0.5, (0.03, 0.05, 0.08))],
          g * (1 - 0.6 * karanlik))
    bulut(taban, B, t, -0.2, 0.3, (0.45, 0.45, 0.75), 0.4 * g * (1 - karanlik), hiz=4, k=1, olcek=0.6)
    for i, (u, boy) in enumerate(((-0.25, 0.55), (0.25, 0.60), (-0.19, 0.40), (0.20, 0.42))):
        k = K.Kalem(u, 0.40, boy)
        k.konik([(0.0, 0.0), (0.01, -0.4), (0.0, -0.75)], 0.03, 0.02)
        for j in range(9):
            x0 = (j - 4) * 0.045
            k.konik(K.bezier((x0 * 0.3, -0.75), (x0, -0.85), (x0 * 1.8, -0.60),
                             (x0 * 2.2 + 0.01 * math.sin(t * 0.8 + j + i), -0.30), 12), 0.008, 0.003)
        K.isle(taban, k, (0.7, 0.75, 1.0), isik=(0.0, -1.0), guc=g, dolgu=(0.02, 0.03, 0.06), kenar_guc=0.4)
    for s in (-1, 1):
        cx = 0.21 * s
        for j in range(18):
            f = ((t * 0.5 + j / 18) % 1.0)
            u = cx + s * 0.04 * f
            v = 0.26 - 0.12 * f + 0.25 * f * f
            lekele(taban, u, v, 0.004, 0.004, np.array((0.75, 0.85, 1.0), np.float32), 1.5 * g * (1 - f))
    korlar(ayrinti, t, 0.5 * g, adet=40, tohum=19, renk=np.array((0.7, 0.9, 1.0), np.float32), yukselis=20)


def s_olorin(taban, ayrinti, B, t, a):
    O = B.O
    p = ilerle(B, 3, t)
    dus = pencere(t, O["korku"] - 0.2, O["manwe"] + 0.3, 0.5)
    lorien(taban, ayrinti, B, t, a, karanlik=0.7 * dus)
    nienna = pencere(t, O["nienna"] - 0.2, O["gelecek"] + 0.1, 0.5)
    manwe = pencere(t, O["manwe"] - 0.1, O["donusum"] + 0.3, 0.5)
    don = puruzsuz((t - O["donusum"]) / (O["gandalf"] - O["donusum"] + 0.05))
    gandalf = puruzsuz((t - O["gandalf"] + 0.05) / 0.25)
    x = W * (0.5 + 0.12 * nienna - 0.10 * manwe)
    if nienna > 0.01:
        yagmur(ayrinti, t, 0.35 * nienna * a, adet=70, tohum=12)
        dn = durum("nienna", t, uzgun=0.5, yas=0.6, gulus=0.3, bakis=(0.5, 0.1))
        portre(B, "nienna", t, dn, a * nienna, x=0.24 * W, y=0.40 * H, olcek=0.66, aydinlik=0.9,
               alt=(0.55 * H, 1.1 * H), soldur=(0.62 * H, 0.80 * H))
    if manwe > 0.01:
        arka_isik(taban, 0.25, KAFA_V - 0.02, (0.4, 0.6, 1.0), 0.6 * manwe * a, 0.25)
        dm = durum("manwe", t, gulus=0.25, bakis=(-0.5, 0.2))
        portre(B, "manwe", t, dm, a * manwe, x=0.77 * W, y=0.38 * H, olcek=0.64, aydinlik=0.95,
               alt=(0.55 * H, 1.1 * H), soldur=(0.60 * H, 0.78 * H))
    gelecek = pencere(t, O["gelecek"] - 0.2, O["korku"] + 0.2, 0.6)
    if gelecek > 0.01:
        K.gezgin(taban, 0.20, 0.42, 0.78, t, renk=(0.75, 0.80, 1.0), guc=0.7 * gelecek * a, asa_isik=0.9)
    if dus > 0.01:
        K.kara_lord(taban, 0.20, 0.08, 0.20, t, guc=0.6 * dus * a)
        lekele(taban, 0.20, -0.05, 0.08, 0.08, KOR, 0.4 * dus * a)
    bakis = (0.0, 0.0)
    if nienna > 0.5:
        bakis = (-0.6, 0.1)
    elif dus > 0.5:
        bakis = (0.1, 0.6)
    elif manwe > 0.5:
        bakis = (0.6, -0.3)
    d = durum("olorin", t, uzgun=0.7 * dus, gulus=0.25 * (1 - dus) + 0.2 * manwe, bakis=bakis, ruzgar=0.8,
              kapak=0.15 * nienna)
    isik = pencere(t, O["donusum"] - 0.1, O["gandalf"] + 0.6, 0.3)
    if isik > 0.01:
        arka_isik(taban, 0.0, KAFA_V, (1.0, 1.0, 1.0), 1.2 * isik * a, 0.3)
        isinlar(taban, B, t, np.array((0.9, 0.95, 1.0), np.float32), 1.0 * isik * a, 0.5, merkez=(0.0, KAFA_V))
    ol = 1.0 + 0.05 * p
    portre(B, "olorin", t, d, a * (1 - gandalf), x=x, olcek=ol, ton=(1.0 + 0.5 * don,) * 3)
    if gandalf > 0.01:
        dg = durum("gandalf", t, gulus=0.3, kas_kalk=0.2, bakis=(0.0, 0.0))
        portre(B, "gandalf", t, dg, a * gandalf, olcek=ol, ton=(1.0 + 0.4 * (1 - gandalf),) * 3)


# ------------------------------------------------------------------ 4 · Sauron (Mairon)

def demirhane(taban, ayrinti, B, t, g, vurus, kizil=0.0):
    ren = np.array((0.30, 0.10, 0.03), np.float32) * (1 - kizil) + np.array((0.35, 0.02, 0.01), np.float32) * kizil
    dikey(taban, [(-0.5, (0.02, 0.01, 0.01)), (0.1, tuple(ren * 0.35)), (0.5, tuple(ren))], g)
    kemer = [(0.21 * math.cos(x), -0.18 - 0.26 * math.sin(x)) for x in np.linspace(0, math.pi, 30)]
    dunya(taban, [(0.32, 0.5), (0.32, -0.5), (-0.32, -0.5), (-0.32, 0.5), (-0.21, 0.5)] + kemer[::-1] + [(0.21, 0.5)],
          KOR, isik=(0.3, 1.0), dolgu=(0.025, 0.012, 0.007), guc=g, kenar_guc=0.35)
    lekele(taban, 0.18, 0.42, 0.25, 0.14, KOR, (0.9 + 0.2 * math.sin(t * 9) + 1.5 * vurus) * g)
    korlar(ayrinti, t, 0.8 * g, adet=50, tohum=6, yukselis=120)


def s_sauron(taban, ayrinti, B, t, a):
    O = B.O
    vurus = max([math.exp(-(t - v) / 0.25) for v in O["ors"] if t >= v] + [0.0])
    soru = a * kesit(t, -1, O["demirhane"] - 0.1, 0.1)
    ana = a * kesit(t, O["demirhane"] - 0.1, O["saruman"], 0.1)
    sar = a * kesit(t, O["saruman"], O["yuzuk"], 0.1)
    yuz = a * kesit(t, O["yuzuk"], 999, 0.1)
    if soru > 0.01:
        dikey(taban, [(-0.5, (0.01, 0.0, 0.0)), (0.5, (0.06, 0.01, 0.0))], soru)
        acil = puruzsuz((t - O["soru"]) / 1.0)
        lekele(taban, 0.0, -0.12, 0.10 * acil, 0.02 + 0.03 * acil, KOR, 1.5 * soru * acil)
        lekele(taban, 0.0, -0.12, 0.015 * acil, 0.02 * acil, np.array((1.0, 0.8, 0.3), np.float32), 4 * soru * acil)
        korlar(ayrinti, t, 0.5 * soru, adet=30, tohum=9)
    if ana > 0.01:
        d, f = mairon_durumu(t, O)
        demirhane(taban, ayrinti, B, t, ana, vurus, kizil=f)
        melkor = puruzsuz((t - O["dusus"]) / 1.2)
        if melkor > 0.01:
            K.melkor_dev(taban, 0.0, 0.58, 1.05, t, guc=0.75 * melkor * ana, kor_guc=0.6 + 0.8 * f)
        aule = ana * pencere(t, O["demirhane"], O["mairon"] + 0.2, 0.4)
        if aule > 0.01:
            da = durum("aule", t, gulus=0.3, bakis=(0.6, 0.2))
            portre(B, "aule", t, da, aule, x=0.26 * W, y=0.37 * H, olcek=0.68, aydinlik=0.85,
                   ton=(1.05, 0.92, 0.80), alt=(0.55 * H, 1.1 * H), soldur=(0.62 * H, 0.80 * H))
        x = W * (0.5 + 0.12 * aule / max(ana, 1e-3))
        if O["mairon"] < t < O["dusus"]:
            d["gulus"] = 0.55
        if aule > 0.3:
            d["bakis"] = (-0.5, 0.0)
        ton = (1.05 + 0.2 * vurus, 0.92 + 0.1 * vurus - 0.35 * f, 0.82 - 0.4 * f)
        portre(B, "mairon", t, d, ana, x=x, olcek=1.0 + 0.06 * ilerle(B, 4, t), ton=ton, aydinlik=1.0 - 0.3 * f)
        goz_isigi(B, "mairon", d, 0.9 * f * ana, KIRMIZI_GOZ, x=x, olcek=1.0 + 0.06 * ilerle(B, 4, t))
        kat = B.on_katman()
        for j, v in enumerate(O["ors"]):
            kivilcim(kat, 0.20, 0.40, v, t, adet=35, hiz=0.5, tohum=j, guc=ana)
    if sar > 0.01:
        dikey(taban, [(-0.5, (0.02, 0.02, 0.03)), (0.0, (0.06, 0.06, 0.08)), (0.5, (0.02, 0.02, 0.02))], sar)
        dunya(taban, [(0.12, 0.30), (0.13, -0.28), (0.11, -0.36), (0.14, -0.33), (0.155, -0.40), (0.17, -0.33),
                      (0.20, -0.36), (0.18, -0.28), (0.19, 0.30)], (0.6, 0.65, 0.8), dolgu=(0.01, 0.01, 0.015), guc=sar,
              kenar_guc=0.5)
        K.gezgin(taban, -0.06, 0.30, 0.46, t, renk=(0.95, 0.95, 1.0), guc=sar, parlak=True, asa_isik=0.9)
        sis(taban, 0.30, 0.06, (0.2, 0.2, 0.25), 0.5 * sar)
    if yuz > 0.01:
        demirhane(taban, ayrinti, B, t, yuz, vurus, kizil=1.0)
        d, f = mairon_durumu(t, O)
        portre(B, "mairon", t, d, 0.45 * yuz, olcek=1.06, ton=(1.0, 0.55, 0.45), aydinlik=0.7,
               alt=(0.45 * H, 0.75 * H), soldur=(0.50 * H, 0.66 * H))
        goz_isigi(B, "mairon", d, 0.8 * yuz, KIRMIZI_GOZ, olcek=1.06)
        kat = B.on_katman()
        parla = 0.75 + 0.35 * vurus + 0.1 * math.sin(t * 5)
        X, Y = W / 2, 0.595 * H
        parilti(kat, X, Y, 150, (1.0, 0.55, 0.15), 0.30 * yuz * parla)
        for kal, g in ((44, 0.10), (22, 0.25), (11, 0.8), (4, 1.0)):
            renk = (1.0, 0.78, 0.35) if kal > 4 else (1.0, 0.95, 0.8)
            cv2.ellipse(kat, (int(X * 16), int(Y * 16)), (int(150 * 16), int(52 * 16)), 0, 0, 360,
                        tuple(float(c) * 255 * g * parla * yuz for c in renk), kal, cv2.LINE_AA, 4)
        dunya(taban, [(-0.20, 0.125), (0.20, 0.125), (0.23, 0.16), (0.23, 0.6), (-0.23, 0.6), (-0.23, 0.16)],
              KOR, isik=(0.0, -1.0), dolgu=(0.03, 0.02, 0.02), guc=yuz, kenar_guc=0.8)
        for j, v in enumerate(O["ors"]):
            kivilcim(kat, 0.0, 0.10, v, t, adet=45, hiz=0.6, tohum=j + 7, renk=np.array((1.0, 0.7, 0.3), np.float32),
                     guc=yuz)


# ------------------------------------------------------------------ 5 · Balroglar

def magara(taban, ayrinti, B, t, g):
    dikey(taban, [(-0.5, (0.01, 0.005, 0.005)), (0.1, (0.06, 0.015, 0.008)), (0.5, (0.20, 0.05, 0.02))], g)
    for s in (-1, 1):
        x = np.linspace(0, 1, 30)
        kenar = [(s * (0.20 + 0.06 * math.sin(i * 1.3) + 0.03 * math.sin(i * 3.7)), -0.5 + f) for i, f in enumerate(x)]
        dunya(taban, kenar + [(s * 0.5, 0.5), (s * 0.5, -0.5)], KOR, isik=(-0.8 * s, 0.4), dolgu=(0.015, 0.006, 0.004),
              guc=g, kenar_guc=0.5)
    korlar(ayrinti, t, 0.7 * g, adet=60, tohum=21, yukselis=140)


def s_balrog(taban, ayrinti, B, t, a):
    O = B.O
    cek = a * kesit(t, -1, O["cikis"], 0.15)
    bal = a * kesit(t, O["cikis"], 999, 0.15)
    if cek > 0.01:
        dikey(taban, [(-0.5, (0.01, 0.01, 0.03)), (0.0, (0.05, 0.02, 0.05)), (0.5, (0.02, 0.0, 0.0))], cek)
        B.yildiz.ciz(ayrinti, t, 0.4 * cek)
        K.melkor_dev(taban, 0.0, 0.42, 0.85, t, guc=cek, kor_guc=1.2)
        f = puruzsuz((t - O["cekim"]) / 3.0)
        ruh_bulutu(ayrinti, B, t, (0.0, -0.05), 0.30, (1.0, 0.9, 0.6), cek, adet=200, hiz=0.4, cekim=f * 1.6,
                   hedef=(0.0, -0.33), kizil=1.0)
    if bal > 0.01:
        yerel = t - O["cikis"]
        magara(taban, ayrinti, B, t, bal)
        cik = puruzsuz(yerel / 2.0)
        saklama = max([pencere(t, k, k + 0.7, 0.2) for k in O["kirbac"]] + [0.0])
        kanat = 0.4 + 0.6 * puruzsuz((t - O["kirbac"][0] + 0.5) / 1.0)
        lekele(taban, 0.0, 0.1, 0.35, 0.4, KOR, (0.25 + 0.4 * saklama) * bal)
        K.balrog(taban, ayrinti, 0.0, 0.47 + 0.10 * (1 - cik), 0.78, t, guc=bal * (0.3 + 0.7 * cik),
                 saklama=saklama, kanat=kanat)
        korlar(B.on_katman(), t + 3, 0.8 * bal, adet=30, tohum=22, yukselis=220)


# ------------------------------------------------------------------ 6 · Köprü

def kopru_sahnesi(taban, ayrinti, B, t, g, yakin=0.0, vurus_t=None, gandalf_t=None):
    """Khazad-dûm: sütunlu salon, uçurum, ince köprü; Gandalf (beyaz) ve Balrog karşı karşıya.
    vurus_t: asanın köprüye vurduğu an (köprü kırılır, Balrog düşer); gandalf_t: kırbacın Gandalf'ı çektiği an."""
    dikey(taban, [(-0.5, (0.01, 0.005, 0.005)), (0.1, (0.05, 0.015, 0.01)), (0.5, (0.35, 0.08, 0.02))], g)
    for i in range(7):
        u = -0.36 + i * 0.12
        dunya(taban, [(u - 0.025, 0.08), (u - 0.02, -0.5), (u + 0.02, -0.5), (u + 0.025, 0.08)], KOR,
              isik=(0.0, 1.0), dolgu=(0.015, 0.008, 0.006), guc=g * 0.8, kenar_guc=0.25)
    korlar(ayrinti, t, 0.8 * g, adet=60, tohum=25, yukselis=170)
    vurus = 0.0 if vurus_t is None else puruzsuz((t - vurus_t) / 0.3)
    dusus = 0.0 if vurus_t is None else max(0.0, t - vurus_t - 0.2)
    g_dusus = 0.0 if gandalf_t is None else max(0.0, t - gandalf_t)
    alev = 1.0 + (0.0 if vurus_t is None else 2.0 * pencere(t, vurus_t + 0.6, vurus_t + 4.0, 0.8))
    lekele(taban, 0.0, 0.40, 0.4, 0.15, KOR, 1.0 * g * alev)
    v = 0.14 - 0.03 * yakin
    ol = 1.0 + 0.5 * yakin
    K.kopru(taban, v, t, guc=g, kirik=vurus, kirik_u=0.02, dusus=dusus)
    gx = -0.13 * ol + 0.05 * min(1.0, g_dusus)
    K.gezgin(taban, gx, v - 0.004 + g_dusus * g_dusus * 0.45, 0.19 * ol, t, renk=(0.9, 0.95, 1.0),
             guc=g * max(0.0, 1 - 0.6 * g_dusus), parlak=True,
             asa_isik=1.0 + 3.0 * pencere(t, (vurus_t or -9) - 0.2, (vurus_t or -9) + 0.4, 0.15))
    saklama = 0.5 + 0.5 * math.sin(t * 1.3)
    if gandalf_t is not None:
        saklama = max(saklama, pencere(t, gandalf_t - 0.5, gandalf_t + 0.3, 0.2))
    if dusus < 2.5:
        K.balrog(taban, ayrinti, 0.14 * ol, v - 0.004 + dusus * dusus * 0.5, 0.52 * ol, t,
                 guc=g * max(0.0, 1 - 0.4 * dusus), yon=-1, saklama=saklama)


def s_kopru(taban, ayrinti, B, t, a):
    O, z = B.O, B.zz
    p = ilerle(B, 6, t)
    genis = a * (1 - kesit(t, O["yakin"], O["vurus"] - 0.05, 0.08)) * (1 - kesit(t, O["ak"], 999, 0.1))
    if genis > 0.01:
        vurus_t = O["vurus"] if t > O["yakin"] else None
        gandalf_t = O["gandalf_dusus"] if t > O["yakin"] else None
        kopru_sahnesi(taban, ayrinti, B, t, genis, yakin=0.25 * p if t < O["yakin"] else 0.35, vurus_t=vurus_t,
                      gandalf_t=gandalf_t)
        ruhlar = pencere(t, z.c(6, 0, 1), O["yakin"], 0.5)
        if ruhlar > 0.01:
            lekele(taban, -0.13, 0.03, 0.06, 0.12, np.array((0.8, 0.9, 1.0), np.float32), 0.6 * ruhlar * genis)
            lekele(taban, 0.14, -0.10, 0.12, 0.2, KOR, 0.5 * ruhlar * genis)
            B.yildiz.ciz(ayrinti, t, 0.5 * ruhlar * genis, y_sinir=0.35 * H)
    bol = a * kesit(t, O["yakin"], O["gecemezsin"], 0.08)
    if bol > 0.01:
        U = (np.arange(w, dtype=np.float32)[None, :] - w / 2) / h
        sinir = (0.62 - 0.24 * (V_KOLON + 0.5)) - 0.5
        sol = np.clip((sinir * w / h - U) * 40 + 0.5, 0, 1)[..., None]
        taban += (sol * np.array((0.10, 0.12, 0.20), np.float32)
                  + (1 - sol) * np.array((0.25, 0.05, 0.01), np.float32)) * bol
        K.balrog(taban, ayrinti, 0.17, 0.60, 0.92, t, guc=bol, yon=-1, kanat=0.6)
        dg = durum("gandalf", t, kas_catik=0.6, bakis=(0.5, 0.0))
        portre(B, "gandalf", t, dg, bol, x=0.33 * W, y=0.42 * H, olcek=0.82, ton=(0.95, 1.0, 1.15),
               maske=B.sol_maske, alt=(0.60 * H, 1.2 * H), soldur=(0.74 * H, 0.92 * H))
        dikis(B, t, bol, renk=(1.0, 0.95, 0.85))
    gec = a * kesit(t, O["gecemezsin"], O["vurus"] - 0.05, 0.06)
    if gec > 0.01:
        yerel = t - O["gecemezsin"]
        dikey(taban, [(-0.5, (0.05, 0.05, 0.08)), (0.5, (0.02, 0.02, 0.03))], gec)
        patla = puruzsuz(yerel / 0.35)
        arka_isik(taban, 0.0, KAFA_V, (0.9, 0.95, 1.0), (0.4 + 0.6 * patla) * gec, 0.30)
        isinlar(taban, B, t, np.array((0.8, 0.88, 1.0), np.float32), 0.9 * patla * gec, 0.6, merkez=(0.0, KAFA_V))
        dg = durum("gandalf", t, kas_catik=1.0, agiz=0.85 * pencere(yerel, 0.05, 1.2, 0.1), ruzgar=2.5,
                   bakis=(0.0, 0.0))
        dg["egim"] += 2.0 * math.sin(yerel * 20) * math.exp(-yerel * 3)
        portre(B, "gandalf", t, dg, gec, olcek=1.08 + 0.05 * patla, ton=(0.98, 1.0, 1.08))
    ak = a * kesit(t, O["ak"], 999, 0.1)
    if ak > 0.01:
        yerel = t - O["ak"]
        dikey(taban, [(-0.5, (0.35, 0.36, 0.42)), (0.0, (0.20, 0.21, 0.26)), (0.5, (0.05, 0.05, 0.07))], ak)
        arka_isik(taban, 0.0, KAFA_V, (1.0, 1.0, 1.0), 1.4 * ak, 0.36)
        isinlar(taban, B, t, np.array((1.0, 1.0, 1.0), np.float32), 0.9 * ak, 0.7, merkez=(0.0, KAFA_V))
        korlar(ayrinti, t, 0.6 * ak, adet=40, tohum=31, renk=np.array((0.9, 0.95, 1.0), np.float32), yukselis=60)
        da = durum("gandalf_ak", t, gulus=0.45, ruzgar=1.2, bakis=(0.0, 0.0))
        ol = 1.0 + 0.05 * puruzsuz(yerel / 3.0)
        portre(B, "gandalf_ak", t, da, ak, olcek=ol, ton=(1.08, 1.08, 1.12))
        goz_isigi(B, "gandalf_ak", da, 0.35 * ak, (0.8, 0.9, 1.0), olcek=ol)


# ------------------------------------------------------------------ 7 · Kapanış: iki Lamba

def s_kapanis(taban, ayrinti, B, t, a):
    O = B.O
    l1 = puruzsuz((t - O["lamba"][0]) / 0.8)
    l2 = puruzsuz((t - O["lamba"][1]) / 0.8)
    isik = 0.5 * (l1 + l2)
    dikey(taban, [(-0.5, (0.01, 0.01, 0.03)), (0.05, (0.03, 0.03, 0.06)), (0.5, (0.01, 0.01, 0.01))], a)
    B.yildiz.ciz(ayrinti, t, 0.6 * a * (1 - 0.6 * isik), y_sinir=0.55 * H)
    lekele(taban, -0.15, 0.05, 0.4, 0.2, np.array((0.5, 0.6, 0.9), np.float32), 0.22 * l1 * a)
    lekele(taban, 0.15, 0.05, 0.4, 0.2, np.array((1.0, 0.75, 0.35), np.float32), 0.22 * l2 * a)
    K.sirt(taban, 0.16, 0.08, 51, (0.02 + 0.10 * isik, 0.02 + 0.09 * isik, 0.03 + 0.07 * isik),
           kenar=(0.8, 0.8, 0.9), kenar_guc=0.3 + 0.7 * isik, guc=a)
    K.lamba(taban, -0.19, 0.20, 0.47, (0.75, 0.85, 1.0), guc=a, yanma=l1, t=t, tohum=1)
    K.lamba(taban, 0.19, 0.20, 0.47, (1.0, 0.80, 0.40), guc=a, yanma=l2, t=t, tohum=2)
    for u, l, ren in ((-0.19, l1, (0.75, 0.85, 1.0)), (0.19, l2, (1.0, 0.80, 0.40))):
        if l > 0.01:
            isinlar(taban, B, t, np.array(ren, np.float32), 0.3 * l * a, 0.3, merkez=(u, 0.20 - 0.48))
    zemin = K.Kalem(0, 0, 1)
    zemin.poligon([(-0.4, 0.19), (0.4, 0.19), (0.4, 0.5), (-0.4, 0.5)])
    K.isle(taban, zemin, (0.8, 0.8, 0.8), isik=(0.0, -1.0), guc=a,
           dolgu=(0.02 + 0.05 * isik, 0.02 + 0.05 * isik, 0.02 + 0.04 * isik), kenar_guc=0.1 * isik)


SAHNELER = {"kanca": s_kanca, "maiar": s_maiar, "tanidik": s_tanidik, "olorin": s_olorin, "sauron": s_sauron,
            "balrog": s_balrog, "kopru": s_kopru, "kapanis": s_kapanis}
NEBULA = {ad: ((0.0, 0.0, 0.0), 0.0) for ad in SAHNELER}


def son_islem(c, kare, t):
    O = c.B.O
    flaslar = [(O["ayni"], 0.45), (O["kanca_b"], 0.35), (O["kanca_c"], 0.35), (O["gandalf"], 0.7), (O["dusus"], 0.2),
               (O["saruman"], 0.25), (O["yuzuk"], 0.25), (O["cikis"], 0.3), (O["yakin"], 0.3), (O["gecemezsin"], 0.5),
               (O["vurus"], 0.9), (O["ak"], 0.8)] + [(x, 0.3) for x in O["yuzler"][1:]] + [(x, 0.10) for x in O["ors"]] + \
        [(x, 0.25) for x in O["kirbac"]] + [(x, 0.3) for x in O["lamba"]]
    flas = sum(g * math.exp(-(t - x) / 0.22) for x, g in flaslar if t >= x)
    if flas > 0.01:
        flas = min(flas, 0.85)
        kare = cv2.addWeighted(kare, 1 - flas, np.full_like(kare, 255), flas, 0)
    sars = (sum(math.exp(-(t - x) / 0.35) for x in O["kirbac"] + [O["vurus"], O["gecemezsin"], O["gandalf_dusus"]]
                if t >= x) * 0.9
            + sum(math.exp(-(t - x) / 0.15) for x in O["ors"] if t >= x) * 0.3)
    if sars > 0.02:
        rs = np.random.default_rng(int(t * FPS))
        M = np.float32([[1, 0, rs.uniform(-10, 10) * sars], [0, 1, rs.uniform(-10, 10) * sars]])
        kare = cv2.warpAffine(kare, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return kare
