"""Bölüm 3 — Valar: sahne çizimleri (yüzü görünen, animasyonlu portreler).

Her Vala, portre.py düzeneğiyle çizilen, göz kırpan, nefes alan, bakışını
kaydıran ve anlatıma göre ifade değiştiren bir portreyle tanıtılır. Portreler
motorun ön plan katmanına (B.on_plan) bindirilir; arkalarında sahneye özgü,
hareketli bir dünya (Taniquetil, yıldızlar, deniz dibi, demirhane, orman,
Mandos'un salonları, yağmur, gece ormanı) çizilir. Ön plandaki parçacıklar
(kıvılcım, yağmur, kabarcık, çiçek) B.on_katman() üzerine çizilir.
"""
import math

import cv2
import numpy as np

import karakterler as K
import portre as P
import valar as VL
from motor import FPS, H, KIZIL, KOR, W, h, halka, isinlar, lekele, pencere, puruzsuz, w
from sahneler_b02 import kivilcim, patlama, ruzgar, sis

S_PORTRE = 400
KAFA_Y = 0.36 * H
ALT = (0.62 * H, 1.30 * H)
SOLDUR = (0.80 * H, 1.0 * H)
V_KOLON = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]
KAFA_V = KAFA_Y / H - 0.5

TEMA = {"manwe": (0.30, 0.50, 1.0), "varda": (0.55, 0.60, 1.0), "ulmo": (0.15, 0.80, 0.75), "aule": (1.0, 0.45, 0.12),
        "yavanna": (0.45, 0.90, 0.30), "mandos": (0.40, 0.45, 0.70), "nienna": (0.65, 0.68, 0.78),
        "orome": (0.55, 0.85, 0.40), "tulkas": (1.0, 0.75, 0.30)}
TOHUM = {ad: i * 7 + 3 for i, ad in enumerate(TEMA)}
MONTAJ = [("manwe", {}), ("varda", {"gulus": 0.3}), ("ulmo", {"ruzgar": 1.2}), ("mandos", {"kas_catik": 0.6}),
          ("tulkas", {"gulus": 1.0, "agiz": 0.6, "kapak": 0.35})]


# ------------------------------------------------------------------ zaman ve olaylar

class _Z:
    """zaman.json sözlüğünden cümle/parça zamanları (müzik de aynı olayları kullanır)."""

    def __init__(self, zd):
        self.birim, self.sahne = zd["birimler"], zd["sahneler"]
        self.sinir = [0.0] + [(self.sahne[i - 1]["bit"] + self.sahne[i]["bas"]) / 2
                              for i in range(1, len(self.sahne))] + [zd["sure"]]

    def c(self, s, k=0, p=None):
        return next(b["bas"] for b in self.birim if b["sahne"] == s and b["cumle"] == k and (p is None or b["parca"] == p))

    def e(self, s, k=0, p=None):
        return [b["bit"] for b in self.birim if b["sahne"] == s and b["cumle"] == k and (p is None or b["parca"] == p)][-1]


def olaylar(zd):
    """Görüntü ve müziğin paylaştığı olay zamanları."""
    z = _Z(zd)
    O = {}
    T1 = z.c(0, 1) - 0.1
    O["yuzler"] = [i * T1 / len(MONTAJ) for i in range(len(MONTAJ))]
    O["kanca_b"], O["kanca_c"] = T1, z.c(0, 3) - 0.1
    O["hayir"] = z.c(0, 1, 1)
    O["insanlar"] = z.c(0, 2)
    O["melkor_cokus"] = z.c(1, 1) + 0.75
    O["yakinlas"] = z.c(1, 2)
    O["manwe_gecis"] = z.sinir[2]
    O["kartal"] = z.c(2, 3) - 0.3
    O["yildizlar"] = z.c(3, 0)
    O["sise"] = (z.c(3, 2) - 0.12, z.c(3, 3) - 0.12)
    O["ulmo_ses"] = z.c(4, 1) + 0.6
    bas, bit = z.c(5, 0) + 0.35, z.c(5, 1) - 0.1
    O["ors"] = list(np.arange(bas, bit, 1.25))
    O["paylas"], O["sahiplen"] = z.c(5, 2, 0), z.c(5, 2, 1)
    O["agac"] = z.c(6, 2) - 0.2
    O["mandos_goz"] = z.c(7, 2) + 0.5
    O["nienna_isik"] = z.c(8, 2) - 0.3
    O["boru"] = [z.sahne[9]["bas"] - 0.8, z.c(9, 2) + 0.25]
    O["at"] = (z.c(9, 1) - 0.12, z.c(9, 2) - 0.12)
    O["yumruk"] = [z.c(10, 1) + 0.6, z.c(10, 1) + 1.35]
    O["kahkaha"] = z.c(10, 1, 1)
    O["kuyruklu"] = z.c(10, 2) - 0.2
    O["ruhlar"] = z.c(11, 1) - 0.3
    O["iki_isim"] = z.c(11, 2) - 0.1
    O["gandalf"] = z.e(11, 2, 0) - 0.55
    O["sauron"] = z.c(11, 2, 1) + 0.15
    return O


def hazirla(c):
    B = c.B
    B.portre = {ad: VL.olustur(ad, S=S_PORTRE) for ad in VL.VALAR}
    B.kucuk = {ad: VL.olustur(ad, S=230) for ad in ("varda", "ulmo", "aule", "yavanna")}
    B.yildiz = K.Yildizlar(900, 21)
    B.O = olaylar({"birimler": c.z.birim, "sahneler": c.z.sahne, "sure": c.z.sure})
    B.zz = _Z({"birimler": c.z.birim, "sahneler": c.z.sahne, "sure": c.z.sure})
    rng = np.random.default_rng(8)
    # ağ iplikleri (Varda sahnesindeki mağara)
    B.ag = []
    for kose in ((-0.3, -0.5), (0.3, -0.5), (0.3, -0.1), (-0.3, 0.0)):
        for _ in range(7):
            a = rng.uniform(0, 2 * math.pi)
            L = rng.uniform(0.15, 0.4)
            B.ag.append(((kose[0], kose[1]), (kose[0] + L * math.cos(a), kose[1] + L * math.sin(a))))
    # orman ışık hüzmeleri (sabit maske)
    m = np.zeros((h, w), np.float32)
    for x0, gen in ((-0.20, 0.05), (-0.08, 0.03), (0.06, 0.045), (0.17, 0.025)):
        pts = np.float32([[w / 2 + (x0 - gen) * h, 0], [w / 2 + (x0 + gen) * h, 0],
                          [w / 2 + (x0 + gen + 0.25) * h, h], [w / 2 + (x0 - gen + 0.25) * h, h]])
        cv2.fillPoly(m, [np.int32(pts)], 1.0)
    B.huzme = cv2.GaussianBlur(m, (0, 0), 9) * np.clip(1.2 - (V_KOLON + 0.5), 0, 1)
    # gece ormanı ağaçları
    B.agaclar = [(rng.uniform(-0.32, 0.32), rng.uniform(0.02, 0.10), rng.uniform(0.25, 0.45), rng.uniform(0, 6))
                 for _ in range(16)]
    B.gozler = [(rng.uniform(-0.27, 0.27), rng.uniform(-0.05, 0.25), rng.uniform(0.6, 1.2), rng.uniform(0, 6))
                for _ in range(14)]
    B.gozler = [g for g in B.gozler if abs(g[0]) > 0.11]


# ------------------------------------------------------------------ yardımcılar

def ilerle(B, i, t):
    bas, bit = B.z.sinir[i], B.z.sinir[i + 1]
    return float(np.clip((t - bas) / (bit - bas), 0, 1))


def dikey(taban, duraklar, guc=1.0):
    vs = [v for v, _ in duraklar]
    cs = np.array([c for _, c in duraklar], np.float32)
    col = np.stack([np.interp(V_KOLON[:, 0], vs, cs[:, k]) for k in range(3)], -1)
    taban += col[:, None, :] * guc


def bulut(taban, B, t, v0, v1, renk, guc, hiz=10.0, olcek=0.8, k=0, esik=0.42, kayma=0.0):
    if guc <= 0.01:
        return
    M = np.float32([[olcek * 0.45, 0, 300 + hiz * t + kayma], [0, olcek, 500 + k * 200]])
    n = cv2.warpAffine(B.doku[k], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    n = np.clip((n - esik) * 2.2, 0, 1)
    bant = (puruzsuz((V_KOLON - v0) / 0.06) * (1 - puruzsuz((V_KOLON - v1) / 0.10)))
    taban += (n * bant)[..., None] * np.asarray(renk, np.float32) * guc


def kostik(taban, B, t, renk, guc):
    top = None
    for k, (hx, hy, s) in enumerate(((14, 6, 0.55), (-10, 8, 0.45))):
        M = np.float32([[s, 0, 300 + hx * t], [0, s, 420 + hy * t]])
        n = cv2.warpAffine(B.doku[k], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)
        r = np.clip((1 - np.abs(n * 2 - 1) - 0.86) * 7, 0, 1)
        top = r if top is None else top + r
    derin = np.clip(0.9 - (V_KOLON + 0.5) * 1.1, 0, 1)
    taban += (top * derin)[..., None] * np.asarray(renk, np.float32) * guc


def dunya(taban, pts, renk, isik=(0.0, -1.0), dolgu=(0.01, 0.01, 0.02), guc=1.0, kenar_guc=1.0):
    """Dünya (U, V) koordinatlarında siluet poligonu."""
    k = K.Kalem(0.0, 0.0, 1.0)
    k.poligon(pts)
    K.isle(taban, k, renk, isik=isik, guc=guc, dolgu=dolgu, kenar_guc=kenar_guc)
    return k


def dag(u0, v_tepe, gen, v_taban, tohum, sivri=1.3, puruz=0.022, n=120):
    """Pürüzlü dağ silueti noktaları (dünya koordinatı)."""
    x = np.linspace(-gen, gen, n)
    temel = (1 - np.abs(x) / gen) ** sivri
    gurultu = K.fbm1(x * 6 / gen, tohum, oktav=2) * puruz * (0.35 + temel)
    y = v_taban - (v_taban - v_tepe) * temel + gurultu - gurultu[n // 2] * temel
    return list(zip(u0 + x, y)) + [(u0 + gen, 0.7), (u0 - gen, 0.7)]


def sirt_cizgileri(u0, v_tepe, gen, v_taban, tohum, adet=5):
    rng = np.random.default_rng(tohum)
    cizgiler = []
    for i in range(adet):
        yon = -1 if i % 2 else 1
        x0 = rng.uniform(-0.25, 0.25) * gen
        pts = [(u0 + x0, v_tepe + abs(x0) / gen * (v_taban - v_tepe) * 0.9 + 0.01)]
        for k in range(6):
            x, y = pts[-1]
            pts.append((x + yon * rng.uniform(0.01, 0.035), y + rng.uniform(0.03, 0.06)))
        cizgiler.append(pts)
    return cizgiler


def parilti(kat, X, Y, r, renk, guc):
    if guc <= 0.01 or r <= 0.5:
        return
    R = int(r * 3) + 1
    x0, y0, x1, y1 = int(X) - R, int(Y) - R, int(X) + R + 1, int(Y) + R + 1
    a0, a1, b0, b1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
    if a1 <= a0 or b1 <= b0:
        return
    yy, xx = np.mgrid[a0:a1, b0:b1].astype(np.float32)
    g = np.exp(-((xx - X) ** 2 + (yy - Y) ** 2) / (2 * r * r))
    ek = g[..., None] * np.asarray(renk, np.float32) * 255 * guc
    kat[a0:a1, b0:b1] = np.clip(kat[a0:a1, b0:b1] + ek, 0, 255).astype(np.uint8)


def portre(B, ad, t, d, a, x=W / 2, y=KAFA_Y, olcek=1.0, ton=(1.0, 1.0, 1.0), aydinlik=1.0, alt=ALT, soldur=SOLDUR,
           pr=None, ekran=False):
    if a <= 0.003:
        return
    pr = pr or B.portre[ad]

    def ciz(kare):
        P.yerlestir(kare, pr.kare(t, d), pr, x, y, olcek, a, ton, aydinlik, alt, soldur, ekran)

    B.on_plan.append(ciz)


def goz_isigi(B, ad, d, guc, renk, x=W / 2, y=KAFA_Y, olcek=1.0):
    if guc <= 0.01:
        return
    pr = B.portre[ad]
    gx = d.get("bakis", (0, 0))[0]
    for s in (-1, 1):
        px, py = pr.nokta(0.145 * s + 0.028 * gx + 0.03 * d.get("donus", 0.0), -0.03, d)
        X, Y = P.kareye(pr, px, py, x, y, olcek)
        kat = B.on_katman()
        parilti(kat, X, Y, 6 * olcek, (1, 1, 1), guc * 0.8)
        parilti(kat, X, Y, 16 * olcek, renk, guc * 0.6)
        parilti(kat, X, Y, 44 * olcek, renk, guc * 0.18)


def bas_noktasi(B, ad, d, x, y, olcek, bx, by):
    pr = B.portre[ad]
    px, py = pr.nokta(bx, by, d)
    return P.kareye(pr, px, py, x, y, olcek)


def on_siluet(B, ciz, alfa=1.0):
    """Karakterler kütüphanesindeki bir siluet çizimini portrelerin önüne bindirir."""
    tb = np.zeros((h, w, 3), np.float32)
    k = ciz(tb)
    x0, y0, x1, y1 = k.kirp(pay=30)
    if x1 <= x0 or y1 <= y0:
        return
    renk = 1 - np.exp(-np.maximum(tb[y0:y1, x0:x1], 0) * 1.1)
    m = k.m[y0:y1, x0:x1].astype(np.float32) / 255.0
    boyut = ((x1 - x0) * 2, (y1 - y0) * 2)
    renk = cv2.resize(renk, boyut, interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m, boyut, interpolation=cv2.INTER_LINEAR)[..., None]

    def bindir(kare):
        bolge = kare[y0 * 2:y1 * 2, x0 * 2:x1 * 2].astype(np.float32)
        kare[y0 * 2:y1 * 2, x0 * 2:x1 * 2] = np.clip(bolge * (1 - m * alfa) + renk * 255 * alfa, 0, 255).astype(np.uint8)

    B.on_plan.append(bindir)


def yagmur(kat, t, guc, adet=240, tohum=9, egim=0.16):
    if guc <= 0.01:
        return
    rng = np.random.default_rng(tohum)
    for _ in range(adet):
        x0, y0, s = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.5, 1.0)
        y = (y0 + t * 1700 * s) % (H + 80) - 40
        x = (x0 + y * egim) % W
        uz = 40 * s
        b = 110 * s * guc
        cv2.line(kat, (int(x * 16), int(y * 16)), (int((x - egim * uz) * 16), int((y - uz) * 16)),
                 (b * 0.85, b * 0.9, b), 1, cv2.LINE_AA, 4)


def kabarcik(kat, t, guc, adet=40, tohum=4):
    if guc <= 0.01:
        return
    rng = np.random.default_rng(tohum)
    for _ in range(adet):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H + 200)
        hiz, r, faz = rng.uniform(50, 150), rng.uniform(3, 12), rng.uniform(0, 6.28)
        y = H + 60 - ((y0 + t * hiz) % (H + 200))
        x = x0 + 14 * math.sin(t * 1.7 + faz)
        b = 150 * guc * min(1.0, r / 8)
        cv2.circle(kat, (int(x * 16), int(y * 16)), int(r * 16), (b * 0.6, b, b), 1, cv2.LINE_AA, 4)
        cv2.circle(kat, (int((x - r * 0.35) * 16), int((y - r * 0.35) * 16)), int(max(1, r * 0.25) * 16),
                   (b, b, b), -1, cv2.LINE_AA, 4)


def korlar(kat, t, guc, adet=60, tohum=6, renk=KOR, y_alt=H, yukselis=160, alan=(0, W)):
    if guc <= 0.01:
        return
    rng = np.random.default_rng(tohum)
    for _ in range(adet):
        x0, y0 = rng.uniform(*alan), rng.uniform(0, H)
        hiz, faz, r = rng.uniform(0.5, 1.4) * yukselis, rng.uniform(0, 6.28), rng.uniform(1.2, 3.2)
        y = y_alt - ((y0 + t * hiz) % y_alt)
        x = x0 + 22 * math.sin(t * 0.9 + faz) + 8 * math.sin(t * 3.1 + faz * 2)
        omur = y / y_alt
        b = guc * (0.4 + 0.6 * abs(math.sin(t * 5 + faz))) * min(1.0, omur * 2.5)
        cv2.circle(kat, (int(x * 16), int(y * 16)), int(r * 16), tuple(float(c) for c in (renk * 0.7 + 0.3) * 255 * b),
                   -1, cv2.LINE_AA, 4)


def cicek(kat, X, Y, r, buyume, renk, t, faz):
    if buyume <= 0.02:
        return
    rr = r * puruzsuz(buyume)
    renk = np.asarray(renk, np.float32)
    cv2.line(kat, (int(X * 16), int(Y * 16)), (int((X + 6 * math.sin(faz)) * 16), int((Y + r * 3.2) * 16)),
             (40, 120, 40), max(1, int(r * 0.18)), cv2.LINE_AA, 4)
    for i in range(5):
        a = i * 2 * math.pi / 5 + faz + 0.2 * math.sin(t * 0.8 + faz)
        px, py = X + rr * 0.62 * math.cos(a), Y + rr * 0.62 * math.sin(a)
        cv2.ellipse(kat, (int(px * 16), int(py * 16)), (int(rr * 0.55 * 16), int(rr * 0.32 * 16)), math.degrees(a),
                    0, 360, tuple(float(c) for c in renk * 200), -1, cv2.LINE_AA, 4)
    cv2.circle(kat, (int(X * 16), int(Y * 16)), int(rr * 0.28 * 16), (255, 210, 90), -1, cv2.LINE_AA, 4)


def goz_cifti(ayrinti, u, v, olcek, guc, renk=(255, 40, 20)):
    if guc <= 0.02:
        return
    X, Y = W / 2 + u * H, H / 2 + v * H
    for s in (-1, 1):
        cv2.ellipse(ayrinti, (int((X + s * 9 * olcek) * 16), int(Y * 16)), (int(5 * olcek * 16), int(2.5 * olcek * 16)),
                    s * 12, 0, 360, tuple(float(c) * guc for c in renk), -1, cv2.LINE_AA, 4)


def zoom(u, v, F, Z, hedef, g):
    """Kamera yakınlaşması: F noktasına Z kat yaklaşır, F ekranda hedef'e kayar (g: 0-1)."""
    return (F[0] + (u - F[0]) * Z + (hedef[0] - F[0]) * g, F[1] + (v - F[1]) * Z + (hedef[1] - F[1]) * g)


def ifade(t, anahtarlar):
    """[(zaman, deger)] listesinde yumuşak geçişli değer."""
    if t <= anahtarlar[0][0]:
        return anahtarlar[0][1]
    for (t0, a0), (t1, a1) in zip(anahtarlar, anahtarlar[1:]):
        if t < t1:
            f = puruzsuz((t - t0) / max(1e-3, t1 - t0))
            if isinstance(a0, tuple):
                return tuple(x + (y - x) * f for x, y in zip(a0, a1))
            return a0 + (a1 - a0) * f
    return anahtarlar[-1][1]


def kahkaha(t, t0, guc=1.0):
    """Gülme animasyonu: ağız açılıp kapanır, gözler kısılır, baş sallanır."""
    if t < t0 or guc <= 0:
        return {}
    k = puruzsuz((t - t0) / 0.3) * guc
    titre = 0.5 + 0.5 * math.sin((t - t0) * 2 * math.pi * 4.2)
    return {"gulus": 1.0 * k, "agiz": (0.45 + 0.55 * titre) * k, "kapak": 0.45 * k,
            "egim_ek": 2.5 * math.sin((t - t0) * 2 * math.pi * 1.3) * k, "nefes_ek": 0.6 * titre * k}


def durum(ad, t, **ek):
    egim_ek, nefes_ek = ek.pop("egim_ek", 0.0), ek.pop("nefes_ek", 0.0)
    d = P.bosta(t, TOHUM[ad], **ek)
    d["egim"] += egim_ek
    d["nefes"] += nefes_ek
    return d


def arka_isik(taban, u, v, renk, guc, r=0.22):
    lekele(taban, u, v, r, r * 1.1, np.asarray(renk, np.float32), guc)


def kesit(t, bas, bit, g=0.12):
    return pencere(t, bas, bit, g)


# ------------------------------------------------------------------ 0 · kanca

def s_kanca(taban, ayrinti, B, t, a):
    O = B.O
    # A · hızlı yüz montajı
    gA = a * kesit(t, -1, O["kanca_b"], 0.06)
    if gA > 0.01:
        k = min(len(MONTAJ) - 1, int(np.searchsorted(O["yuzler"], t, side="right") - 1))
        ad, ek = MONTAJ[max(0, k)]
        t0 = O["yuzler"][max(0, k)]
        yerel = t - t0
        renk = np.asarray(TEMA[ad], np.float32)
        dikey(taban, [(-0.5, renk * 0.05), (0.0, renk * 0.12), (0.5, renk * 0.02)], gA)
        arka_isik(taban, 0.0, 0.43 - 0.5 - 0.02, renk, 0.55 * gA, 0.30)
        isinlar(taban, B, t, renk, 0.35 * gA, 0.4, merkez=(0.0, -0.10))
        B.yildiz.ciz(ayrinti, t, 0.5 * gA)
        d = durum(ad, t + 3.0, **ek)
        d["bakis"] = (0.0, 0.0)
        portre(B, ad, t, d, gA, y=0.43 * H, olcek=1.12 - 0.05 * puruzsuz(yerel / 0.6), alt=(0.66 * H, 1.3 * H),
               soldur=(0.84 * H, 1.0 * H))
        if ad == "mandos":
            goz_isigi(B, ad, d, 0.5 * gA, TEMA[ad], y=0.43 * H, olcek=1.12 - 0.05 * puruzsuz(yerel / 0.6))
    # B · gökteki yüzler ve onlara tanrı diyen insanlar
    gB = a * kesit(t, O["kanca_b"], O["kanca_c"], 0.06)
    if gB > 0.01:
        hayir = pencere(t, O["hayir"], O["insanlar"] + 0.2, 0.3)
        dikey(taban, [(-0.5, (0.02, 0.03, 0.10)), (0.05, (0.10, 0.12, 0.28)), (0.30, (0.35, 0.22, 0.25)),
                      (0.5, (0.05, 0.03, 0.05))], gB * (1 - 0.35 * hayir))
        B.yildiz.ciz(ayrinti, t, 0.5 * gB, y_sinir=0.45 * H)
        bulut(taban, B, t, 0.00, 0.30, (0.55, 0.60, 0.85), 0.35 * gB, hiz=8)
        for u, v, r in ((0.0, -0.22, 0.25), (-0.19, -0.18, 0.14), (0.19, -0.18, 0.14)):
            arka_isik(taban, u, v, (0.4, 0.55, 1.0), 0.35 * gB * (1 - 0.5 * hayir), r)
        ton = (0.85, 0.95, 1.15)
        yuz_a = gB * (0.95 - 0.45 * hayir)
        for ad, x, y, ol in (("varda", 0.18 * W, 0.30 * H, 0.60), ("ulmo", 0.82 * W, 0.30 * H, 0.60)):
            portre(B, ad, t, durum(ad, t, bakis=(0.0, 0.3)), yuz_a * 0.8, x=x, y=y, olcek=ol, ton=ton, alt=None,
                   soldur=(y + 0.06 * H, y + 0.15 * H), ekran=True)
        portre(B, "manwe", t, durum("manwe", t, bakis=(0.0, 0.4)), yuz_a, y=0.25 * H, olcek=0.92, ton=ton, alt=None,
               soldur=(0.33 * H, 0.45 * H), ekran=True)
        gI = gB * puruzsuz((t - O["insanlar"]) / 0.5)
        dunya(taban, [(-0.4, 0.5), (-0.4, 0.36), (-0.2, 0.33), (0.0, 0.315), (0.2, 0.33), (0.4, 0.35), (0.4, 0.5)],
              (0.6, 0.7, 1.0), guc=gB, dolgu=(0.01, 0.01, 0.02))
        if gI > 0.01:
            for i, (u, tur) in enumerate(((-0.13, "adam"), (-0.07, "kadin"), (-0.02, "cocuk"), (0.04, "adam"),
                                          (0.10, "kadin"))):
                boy = 0.085 if tur != "cocuk" else 0.055
                K.insan(taban, u, 0.325 + abs(u) * 0.12, boy, t, renk=(0.75, 0.82, 1.0), isik=(0.0, -1.0), guc=gI,
                        faz=i, tur=tur, yon=1 if u < 0 else -1)
            for u in (-0.10, 0.0, 0.08):
                K.isik_sutunu(taban, u * 0.6, -0.15, 0.32, 0.03, np.array([0.5, 0.6, 1.0], np.float32), 0.12 * gI)
    # C · Valar dizilişi
    gC = a * kesit(t, O["kanca_c"], 999, 0.06)
    if gC > 0.01:
        yerel = t - O["kanca_c"]
        dikey(taban, [(-0.5, (0.02, 0.02, 0.06)), (0.0, (0.08, 0.06, 0.16)), (0.5, (0.01, 0.01, 0.03))], gC)
        B.nebula(taban, t, np.array((0.25, 0.18, 0.40), np.float32), 0.25 * gC)
        B.yildiz.ciz(ayrinti, t, 0.8 * gC)
        arka_isik(taban, 0.0, -0.05, (1.0, 0.72, 0.35), 0.5 * gC, 0.35)
        isinlar(taban, B, t, np.array((1.0, 0.8, 0.5), np.float32), 0.45 * gC, 0.45, merkez=(0.0, -0.08))
        z = 1.0 + 0.05 * puruzsuz(yerel / 4.0)
        karart, sol = (0.56 * H, 1.1 * H), (0.68 * H, 0.84 * H)
        dizi = [("aule", 0.06, 0.425, 0.70, 0.95), ("yavanna", 0.94, 0.425, 0.70, 0.95),
                ("varda", 0.27, 0.45, 0.88, 1.0), ("ulmo", 0.73, 0.45, 0.88, 1.0)]
        for i, (ad, x, y, ay, ol) in enumerate(dizi):
            gecikme = 0.12 * (3 - i)
            gi = gC * puruzsuz((yerel - gecikme) / 0.35)
            X = W / 2 + (x * W - W / 2) * z
            portre(B, ad, t, durum(ad, t, gulus=0.2), gi, x=X, y=y * H, olcek=ol * z, aydinlik=ay, alt=karart,
                   soldur=sol, pr=B.kucuk[ad])
        portre(B, "manwe", t, durum("manwe", t, gulus=0.15), gC, y=0.47 * H, olcek=0.72 * z, alt=karart, soldur=sol)


# ------------------------------------------------------------------ 1 · Máhanaxar: on dört taht

def s_tanitim(taban, ayrinti, B, t, a):
    O = B.O
    g = a
    yak = puruzsuz((t - O["yakinlas"]) / (B.z.sinir[2] - O["yakinlas"] + 0.3))
    Z = 1.0 + 3.6 * yak ** 2
    cu, cv, ru, rv = 0.0, 0.10, 0.22, 0.075
    F = (cu, cv - rv - 0.09)
    hedef = (0.0, KAFA_V)
    dikey(taban, [(-0.5, (0.01, 0.02, 0.07)), (-0.1, (0.05, 0.07, 0.20)), (0.07, (0.30, 0.20, 0.25)),
                  (0.12, (0.06, 0.05, 0.08)), (0.5, (0.01, 0.01, 0.02))], g)
    B.yildiz.ciz(ayrinti, t, 0.7 * g * (1 - yak), y_sinir=0.55 * H)
    K.sirt(taban, 0.07 + 0.1 * yak, 0.20, 41, (0.035, 0.04, 0.08), kenar=(0.8, 0.85, 1.0), kenar_guc=0.7, guc=g,
           kayma=0.0, sivri=1.2)
    K.sirt(taban, 0.08 + 0.2 * yak, 0.10, 43, (0.02, 0.022, 0.04), kenar=(0.6, 0.6, 0.8), kenar_guc=0.4, guc=g)
    zeminp = [zoom(u, v, F, Z, hedef, yak) for u, v in ((-0.6, 0.06), (0.6, 0.06), (0.6, 0.6), (-0.6, 0.6))]
    dunya(taban, zeminp, (0.4, 0.4, 0.6), dolgu=(0.03, 0.03, 0.05), guc=g, kenar_guc=0.4)
    lekele(taban, *zoom(cu, cv, F, Z, hedef, yak), ru * Z * 1.2, rv * Z * 1.6, np.array((1.0, 0.8, 0.5), np.float32),
           0.25 * g)
    tahtlar = []
    for i in range(14):
        th = math.pi / 2 + 2 * math.pi * i / 14
        d = (math.sin(th) + 1) / 2
        u, v = cu + ru * math.cos(th), cv + rv * math.sin(th)
        boy = 0.075 * (0.7 + 0.6 * d)
        kralice = i % 2 == 1
        renk = np.array((1.0, 0.72, 0.55) if kralice else (0.55, 0.75, 1.0), np.float32)
        if i == 7:
            renk = np.array((0.5, 0.75, 1.0), np.float32)
        tahtlar.append((v, u, boy, renk, i))
    for v, u, boy, renk, i in sorted(tahtlar):
        su, sv = zoom(u, v, F, Z, hedef, yak)
        sb = boy * Z
        if abs(su) > 0.5 or sv > 0.8 or sv - sb > 0.6:
            continue
        gecikme = 0.05 * i
        gi = g * puruzsuz((t - B.z.sinir[1] - gecikme + 0.3) / 0.4)
        K.taht(taban, su, sv, sb, renk * 0.7, guc=gi, kenar_guc=0.8)
        K.ainu(taban, su, sv - 0.02 * sb, sb * 0.82, t, renk, guc=gi * (1.2 if i == 7 else 0.8), faz=i,
               hale=i == 7)
    # Melkor'un tahtı: parçalanır
    cok = puruzsuz((t - O["melkor_cokus"]) / 0.9)
    mg = g * (1 - cok) * (1 - yak)
    if mg > 0.01:
        mu, mv = zoom(0.0, 0.42, F, Z, hedef, yak)
        K.taht(taban, mu, mv, 0.17 * Z, KIZIL, guc=mg, dolgu=(0.02, 0.0, 0.0), kenar_guc=1.2)
        K.melkor_dev(taban, mu, mv - 0.01, 0.145 * Z, t, guc=mg, kor_guc=1.0 + 2 * pencere(t, O["melkor_cokus"] - 0.3,
                                                                                      O["melkor_cokus"] + 0.2, 0.2))
    patlama(ayrinti, 0.0, 0.33, O["melkor_cokus"], t, KOR, adet=180, hiz=0.5, omur=2.5, tohum=3, guc=g * (1 - yak))


# ------------------------------------------------------------------ 2 · Manwë

def s_manwe(taban, ayrinti, B, t, a):
    O, z = B.O, B.zz
    p = ilerle(B, 2, t)
    dikey(taban, [(-0.5, (0.04, 0.10, 0.32)), (-0.1, (0.16, 0.30, 0.60)), (0.2, (0.45, 0.55, 0.78)),
                  (0.5, (0.30, 0.36, 0.55))], a)
    lekele(taban, -0.22, -0.40, 0.18, 0.18, np.array((1.0, 0.85, 0.6), np.float32), 0.8 * a)
    lekele(taban, -0.22, -0.40, 0.03, 0.03, np.array((1.0, 0.95, 0.8), np.float32), 2.0 * a)
    kay = 0.02 * p
    dunya(taban, dag(-0.24 + kay * 0.5, -0.12, 0.30, 0.25, 5), (0.9, 0.95, 1.0), isik=(-0.7, -0.7),
          dolgu=(0.20, 0.25, 0.42), guc=a, kenar_guc=0.3)
    k = dunya(taban, dag(0.08 + kay, -0.36, 0.42, 0.30, 2), (1.0, 1.0, 1.0), isik=(-0.7, -0.7),
              dolgu=(0.34, 0.42, 0.64), guc=a, kenar_guc=0.4)
    K.detay(taban, k, sirt_cizgileri(0.08 + kay, -0.36, 0.42, 0.30, 4, 6), (1.0, 1.0, 1.0), 0.25 * a)
    lekele(taban, -0.02 + kay, -0.22, 0.10, 0.12, np.array((0.9, 0.95, 1.0), np.float32), 0.25 * a)
    bulut(taban, B, t, 0.05, 0.50, (0.85, 0.90, 1.0), 0.75 * a, hiz=14, k=0)
    bulut(taban, B, t, 0.18, 0.50, (0.95, 0.97, 1.0), 0.55 * a, hiz=26, k=1, olcek=0.6)
    sis(taban, 0.35, 0.18, (0.55, 0.62, 0.85), 0.5 * a)
    for i, (r, hiz, boy) in enumerate(((0.10, 0.45, 0.05), (0.14, -0.33, 0.04))):
        aci = t * hiz + i * 2
        K.kartal(taban, 0.02 + r * math.cos(aci), -0.36 + 0.3 * r * math.sin(aci), boy, t, guc=0.9 * a, faz=i)
    ruzgar(ayrinti, t, 0.8 * a, adet=40)
    ilk = z.c(2, 1)
    d = durum("manwe", t, ruzgar=1.4, gulus=0.2,
              kapak=ifade(t, [(ilk + 0.3, 0.0), (ilk + 0.7, 0.85), (z.e(2, 1) - 0.3, 0.85), (z.e(2, 1), 0.0)]),
              bakis=ifade(t, [(O["kartal"] - 0.2, (0.0, 0.0)), (O["kartal"] + 0.2, (-0.6, -0.6)),
                              (O["kartal"] + 1.6, (-0.6, -0.6)), (O["kartal"] + 2.0, (0.1, -0.1))]))
    if ilk + 0.3 < t < z.e(2, 1):
        d["bakis"] = (0.0, -0.5)
    tanri_isigi = pencere(t, ilk + 0.2, z.e(2, 1) + 0.3, 0.5)
    lekele(taban, 0.0, -0.5, 0.10, 0.25, np.array((1.0, 0.9, 0.6), np.float32), 0.8 * tanri_isigi * a)
    ol = 1.0 + 0.06 * p
    portre(B, "manwe", t, d, a, olcek=ol, ton=(1.0 + 0.15 * tanri_isigi, 1.0 + 0.1 * tanri_isigi, 1.05))
    kat = B.on_katman()
    ruzgar(kat, t + 3, 0.5 * a, adet=14, tohum=9)
    uc = t - O["kartal"]
    if 0 < uc < 2.2:
        f = uc / 2.2
        on_siluet(B, lambda tb: K.kartal(tb, 0.50 - 1.0 * f, -0.31 + 0.06 * f - 0.05 * math.sin(f * 3), 0.42, t * 1.3,
                                         guc=1.0, isik=(-0.5, -0.8)), alfa=a)


# ------------------------------------------------------------------ 3 · Varda

def s_varda(taban, ayrinti, B, t, a):
    O, z = B.O, B.zz
    p = ilerle(B, 3, t)
    s0, s1 = O["sise"]
    yuz = a * (1 - kesit(t, s0, s1, 0.12))
    sise = a * kesit(t, s0, s1, 0.12)
    if yuz > 0.01:
        korku = puruzsuz((t - z.c(3, 3)) / 0.6)
        dikey(taban, [(-0.5, (0.01, 0.01, 0.05)), (0.0, (0.04, 0.03, 0.12)), (0.5, (0.01, 0.01, 0.03))], yuz)
        B.nebula(taban, t, np.array((0.30, 0.22, 0.55), np.float32), 0.45 * yuz)
        B.yildiz.ciz(ayrinti, t, yuz * (1.0 + 0.6 * korku))
        arka_isik(taban, 0.0, KAFA_V - 0.05, (0.55, 0.60, 1.0), (0.35 + 0.3 * korku) * yuz, 0.26)
        for i in range(14):
            th = math.pi * (1.05 + 0.9 * i / 13)
            r = 0.23 + 0.05 * math.sin(i * 2.3)
            u, v = r * math.cos(th), KAFA_V + 0.02 + r * 0.95 * math.sin(th)
            yak = puruzsuz((t - O["yildizlar"] - 0.18 * i) / 0.25)
            if yak <= 0:
                continue
            flas = math.exp(-max(0.0, t - O["yildizlar"] - 0.18 * i) / 0.4)
            b = yak * (0.55 + 0.25 * math.sin(t * 3 + i) + 1.2 * flas + 0.8 * korku) * yuz
            lekele(taban, u, v, 0.006, 0.006, np.array((0.8, 0.85, 1.0), np.float32), 3.0 * b)
            X, Y = W / 2 + u * H, H / 2 + v * H
            L = (10 + 22 * flas + 14 * korku) * yak
            c = tuple(float(x) for x in np.array((200, 215, 255)) * min(1.0, b * 0.8))
            cv2.line(ayrinti, (int((X - L) * 16), int(Y * 16)), (int((X + L) * 16), int(Y * 16)), c, 1, cv2.LINE_AA, 4)
            cv2.line(ayrinti, (int(X * 16), int((Y - L) * 16)), (int(X * 16), int((Y + L) * 16)), c, 1, cv2.LINE_AA, 4)
        if korku > 0.01:
            geri = puruzsuz((t - z.c(3, 3) - 0.8) / 2.0)
            K.melkor_dev(taban, -0.17 - 0.08 * geri, 0.22 + 0.04 * geri, 0.55 - 0.1 * geri, t,
                         guc=0.55 * korku * (1 - 0.8 * geri) * yuz, kor_guc=0.8)
        d = durum("varda", t, gulus=ifade(t, [(z.c(3, 1), 0.25), (z.c(3, 1) + 0.4, 0.55), (z.c(3, 3), 0.55),
                                               (z.c(3, 3) + 0.4, 0.0)]),
                  kas_catik=0.45 * korku, ruzgar=0.8, bakis=(0.0, 0.0) if korku > 0.5 else None)
        if d["bakis"] is None:
            d["bakis"] = P.bosta(t, TOHUM["varda"])["bakis"]
        ol = 1.0 + 0.06 * p
        portre(B, "varda", t, d, yuz, olcek=ol, ton=(0.95, 0.97, 1.08))
        yx, yy = bas_noktasi(B, "varda", d, W / 2, KAFA_Y, ol, 0.0, -0.415)
        kat = B.on_katman()
        parilti(kat, yx, yy, 8 * ol, (1, 1, 1), (0.8 + 0.2 * math.sin(t * 4)) * yuz)
        parilti(kat, yx, yy, 30 * ol, (0.7, 0.8, 1.0), (0.35 + 0.3 * korku) * yuz)
        goz_isigi(B, "varda", d, 0.35 * korku * yuz, (0.6, 0.75, 1.0), olcek=ol)
        rng = np.random.default_rng(12)
        for _ in range(30):
            x, y0, hz_ = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(15, 40)
            y = (y0 - t * hz_) % H
            parilti(kat, x, y, 1.6, (0.8, 0.85, 1.0), (0.5 + 0.5 * math.sin(t * 3 + x)) * 0.8 * yuz)
    if sise > 0.01:
        yerel = t - s0
        isik = puruzsuz(yerel / 1.2)
        dikey(taban, [(-0.5, (0.0, 0.0, 0.01)), (0.5, (0.01, 0.01, 0.015))], sise)
        for (u0, v0), (u1, v1) in B.ag:
            p0, p1 = (W / 2 + u0 * H, H / 2 + v0 * H), (W / 2 + u1 * H, H / 2 + v1 * H)
            b = 40 * sise * (0.5 + isik)
            cv2.line(ayrinti, (int(p0[0] * 16), int(p0[1] * 16)), (int(p1[0] * 16), int(p1[1] * 16)), (b, b, b * 1.1), 1,
                     cv2.LINE_AA, 4)
        K.hobbit(taban, -0.10, 0.36, 0.19, t, renk=(0.8, 0.88, 1.0), guc=sise, kol=0.0, isik=(0.5, -0.8))
        el = K.hobbit(taban, 0.03, 0.34, 0.22, t, renk=(0.85, 0.92, 1.0), guc=sise, kol=puruzsuz(yerel / 0.6),
                      isik=(0.0, -1.0))
        lekele(taban, el[0], el[1], 0.03 + 0.12 * isik, 0.03 + 0.12 * isik, np.array((0.75, 0.85, 1.0), np.float32),
               1.3 * sise)
        lekele(taban, el[0], el[1], 0.012, 0.012, np.array((1.0, 1.0, 1.0), np.float32), 6 * sise)
        isinlar(taban, B, t, np.array((0.7, 0.8, 1.0), np.float32), 1.2 * sise * isik, 0.25 + 0.2 * isik, merkez=el)
        geri = puruzsuz((yerel - 0.4) / 1.2)
        for j, (du, dv, r) in enumerate(((0, 0, 1.3), (0.02, 0, 1.3), (-0.012, 0.012, 0.8), (0.032, 0.012, 0.8),
                                         (-0.02, -0.012, 0.6), (0.04, -0.012, 0.6), (0.006, 0.02, 0.5),
                                         (0.014, 0.02, 0.5))):
            u, v = 0.13 + du + 0.12 * geri, -0.24 + dv - 0.10 * geri
            X, Y = W / 2 + u * H, H / 2 + v * H
            b = sise * (1 - geri) * (0.7 + 0.3 * math.sin(t * 7 + j))
            cv2.circle(ayrinti, (int(X * 16), int(Y * 16)), int(5 * r * 16), (255 * b, 30 * b, 20 * b), -1, cv2.LINE_AA,
                       4)


# ------------------------------------------------------------------ 4 · Ulmo

def s_ulmo(taban, ayrinti, B, t, a):
    O, z = B.O, B.zz
    p = ilerle(B, 4, t)
    dikey(taban, [(-0.5, (0.05, 0.28, 0.32)), (-0.1, (0.02, 0.14, 0.18)), (0.5, (0.0, 0.02, 0.04))], a)
    kostik(taban, B, t, (0.35, 0.95, 0.9), 0.35 * a)
    for i, u in enumerate((-0.20, -0.09, 0.04, 0.16, 0.25)):
        uu = u + 0.02 * math.sin(t * 0.4 + i)
        K.isik_sutunu(taban, uu, -0.5, 0.2, 0.012 + 0.006 * (i % 2), np.array((0.4, 0.95, 0.9), np.float32),
                      (0.12 + 0.06 * math.sin(t * 0.7 + i * 1.7)) * a)
    for i, u in enumerate((-0.26, -0.21, 0.22, 0.27)):
        pts = [(u + 0.02 * math.sin(t * 0.8 + i + y * 8) * (0.5 - y), y) for y in np.linspace(0.5, 0.05 + 0.05 * i, 12)]
        k = K.Kalem(0, 0, 1)
        k.konik(pts, 0.012, 0.004)
        K.isle(taban, k, (0.3, 0.95, 0.8), isik=(0.0, -1.0), guc=a, dolgu=(0.0, 0.02, 0.02), kenar_guc=0.6)
    ses = t - O["ulmo_ses"]
    for j in range(4):
        r = (ses - j * 0.7) * 0.18
        if 0 < r < 0.6:
            halka(taban, B, 0.0, KAFA_V, r, 0.006 + r * 0.02, np.array((0.4, 1.0, 0.9), np.float32),
                  0.5 * a * (1 - r / 0.6))
    korlar(ayrinti, t, 0.35 * a, adet=50, tohum=2, renk=np.array((0.5, 1.0, 0.9), np.float32), yukselis=40)
    d = durum("ulmo", t, ruzgar=1.8, gulus=ifade(t, [(z.c(4, 2) + 1.0, 0.0), (z.c(4, 2) + 1.8, 0.35)]),
              bakis=ifade(t, [(z.c(4, 2) + 1.0, (0.0, 0.0)), (z.c(4, 2) + 1.4, (0.3, 0.55))])
              if t > z.c(4, 2) + 1.0 else None)
    if d["bakis"] is None:
        d["bakis"] = P.bosta(t, TOHUM["ulmo"])["bakis"]
    ol = 1.0 + 0.06 * p
    portre(B, "ulmo", t, d, a, olcek=ol, ton=(0.88, 1.0, 1.02))
    goz_isigi(B, "ulmo", d, (0.35 + 0.35 * pencere(t, O["ulmo_ses"] - 0.3, O["ulmo_ses"] + 2.5, 0.5)) * a,
              (0.3, 1.0, 0.9), olcek=ol)
    kabarcik(B.on_katman(), t, a)


# ------------------------------------------------------------------ 5 · Aulë

def s_aule(taban, ayrinti, B, t, a):
    O, z = B.O, B.zz
    p = ilerle(B, 5, t)
    vurus = max([math.exp(-(t - v) / 0.25) for v in O["ors"] if t >= v] + [0.0])
    dikey(taban, [(-0.5, (0.02, 0.01, 0.01)), (0.1, (0.10, 0.04, 0.02)), (0.5, (0.30, 0.10, 0.03))], a)
    kemer = [(0.21 * math.cos(x), -0.18 - 0.26 * math.sin(x)) for x in np.linspace(0, math.pi, 30)]
    dunya(taban, [(0.32, 0.5), (0.32, -0.5), (-0.32, -0.5), (-0.32, 0.5), (-0.21, 0.5)] + kemer[::-1] + [(0.21, 0.5)],
          KOR, isik=(0.3, 1.0), dolgu=(0.025, 0.012, 0.007), guc=a, kenar_guc=0.35)
    lekele(taban, 0.0, -0.30, 0.22, 0.10, np.array((0.6, 0.25, 0.08), np.float32), 0.25 * a)
    lekele(taban, 0.18, 0.42, 0.25, 0.14, KOR, (0.9 + 0.2 * math.sin(t * 9) + 1.5 * vurus) * a)
    lekele(taban, 0.18, 0.40, 0.06, 0.03, np.array((1.0, 0.8, 0.4), np.float32), (1.0 + 3 * vurus) * a)
    korlar(ayrinti, t, 0.8 * a, adet=50, tohum=6, yukselis=120)
    melkor = puruzsuz((t - z.c(5, 1) - 0.3) / 0.8)
    sahip = pencere(t, O["sahiplen"], O["sahiplen"] + 2.4, 0.3)
    if melkor > 0.01:
        K.melkor_dev(taban, -0.215, 0.33, 0.62, t, guc=(0.55 + 0.45 * sahip) * melkor * a, kor_guc=0.8 + 1.5 * sahip)
    d = durum("aule", t, kas_catik=ifade(t, [(O["paylas"], 0.25), (O["paylas"] + 0.3, 0.0), (O["sahiplen"], 0.0),
                                            (O["sahiplen"] + 0.3, 0.55)]),
              gulus=ifade(t, [(O["paylas"], 0.1), (O["paylas"] + 0.3, 0.7), (O["sahiplen"], 0.7),
                              (O["sahiplen"] + 0.3, 0.0)]),
              bakis=ifade(t, [(z.c(5, 1), (0.0, 0.1)), (z.c(5, 1) + 0.4, (-0.7, 0.0)), (O["paylas"], (-0.7, 0.0)),
                              (O["paylas"] + 0.3, (0.0, 0.0)), (O["sahiplen"], (0.0, 0.0)),
                              (O["sahiplen"] + 0.3, (-0.7, 0.05))]) if t > z.c(5, 1) else (0.35, 0.45),
              kapak=0.15 * vurus)
    d["egim"] += 1.5 * vurus
    ol = 1.0 + 0.06 * p
    ton = (1.05 + 0.25 * vurus, 0.93 + 0.12 * vurus, 0.82)
    portre(B, "aule", t, d, a, olcek=ol, ton=ton)
    kat = B.on_katman()
    korlar(kat, t + 5, 0.9 * a, adet=26, tohum=11, yukselis=200)
    for j, v in enumerate(O["ors"]):
        kivilcim(kat, 0.20, 0.40, v, t, adet=40, hiz=0.5, tohum=j, guc=a)


# ------------------------------------------------------------------ 6 · Yavanna

def s_yavanna(taban, ayrinti, B, t, a):
    O = B.O
    p = ilerle(B, 6, t)
    dikey(taban, [(-0.5, (0.18, 0.26, 0.10)), (0.0, (0.10, 0.20, 0.06)), (0.5, (0.02, 0.06, 0.02))], a)
    lekele(taban, -0.20, -0.45, 0.20, 0.20, np.array((1.0, 0.9, 0.5), np.float32), 0.9 * a)
    taban += (B.huzme[..., None] * np.array((1.0, 0.9, 0.55), np.float32) * (0.30 + 0.06 * math.sin(t * 0.7)) * a)
    buyu = puruzsuz((t - O["agac"]) / 3.0)
    if buyu > 0:
        K.agac(taban, ayrinti, 0.0, 0.40, 0.95, t, buyume=buyu, renk=(1.0, 0.85, 0.45), guc=a)
        lekele(taban, 0.0, -0.30, 0.3, 0.2, np.array((1.0, 0.85, 0.45), np.float32), 0.5 * buyu * a)
    for i, (u, gen) in enumerate(((-0.26, 0.05), (-0.19, 0.03), (0.21, 0.04), (0.28, 0.06))):
        sal = 0.006 * math.sin(t * 0.6 + i)
        dunya(taban, [(u - gen, 0.5), (u - gen * 0.7 + sal, -0.5), (u + gen * 0.7 + sal, -0.5), (u + gen, 0.5)],
              (1.0, 0.9, 0.5), isik=(-0.8, -0.3), dolgu=(0.03, 0.04, 0.015), guc=a, kenar_guc=0.25)
    korlar(ayrinti, t, 0.55 * a, adet=60, tohum=8, renk=np.array((1.0, 0.9, 0.4), np.float32), yukselis=25)
    d = durum("yavanna", t, gulus=0.45 + 0.2 * buyu, ruzgar=0.9,
              bakis=ifade(t, [(O["agac"], (0.3, 0.4)), (O["agac"] + 0.4, (0.0, -0.6))]) if t > O["agac"] - 0.5
              else P.bosta(t, TOHUM["yavanna"])["bakis"])
    ol = 1.0 + 0.06 * p
    portre(B, "yavanna", t, d, a, olcek=ol, ton=(1.02, 1.02, 0.95))
    kat = B.on_katman()
    rng = np.random.default_rng(21)
    renkler = [(1.0, 0.95, 0.9), (1.0, 0.55, 0.70), (1.0, 0.85, 0.35), (0.80, 0.65, 1.0), (1.0, 0.6, 0.35)]
    for i in range(16):
        X, Y = rng.uniform(0.02, 0.98) * W, rng.uniform(0.84, 0.99) * H
        cicek(kat, X, Y, rng.uniform(16, 30), (t - B.z.sinir[6] + 0.3 - i * 0.18) / 0.8, renkler[i % 5], t,
              rng.uniform(0, 6))
    korlar(kat, t + 2, 0.7 * a, adet=22, tohum=13, renk=np.array((1.0, 0.9, 0.4), np.float32), yukselis=30)


# ------------------------------------------------------------------ 7 · Mandos

def s_mandos(taban, ayrinti, B, t, a):
    O = B.O
    p = ilerle(B, 7, t)
    dikey(taban, [(-0.5, (0.01, 0.01, 0.02)), (-0.12, (0.05, 0.06, 0.10)), (0.5, (0.01, 0.01, 0.02))], a)
    kv = (0.0, -0.13)
    lekele(taban, kv[0], kv[1], 0.08, 0.10, np.array((0.5, 0.6, 0.9), np.float32), 0.6 * a)
    kay = (t * 0.03) % 1.0
    for j in range(9, -1, -1):
        dz = 1.0 + (j + 1 - kay) * 0.55
        for s in (-1, 1):
            x = s * 0.30 / dz
            gen = 0.05 / dz
            ust, alt = kv[1] - 0.55 / dz, kv[1] + 0.60 / dz
            dunya(taban, [(x - gen, alt), (x - gen, ust), (x + gen, ust), (x + gen, alt)], (0.55, 0.65, 1.0),
                  isik=(-s * 1.0, 0.0), dolgu=(0.03, 0.032, 0.048), guc=a * min(1.0, 3.0 / dz), kenar_guc=0.22)
    sis(taban, 0.22, 0.08, (0.25, 0.28, 0.40), 0.5 * a)
    for i in range(8):
        u = 0.25 * math.sin(t * 0.15 + i * 1.9)
        v = -0.05 + 0.2 * math.sin(t * 0.11 + i * 2.7)
        lekele(taban, u, v, 0.012, 0.02, np.array((0.6, 0.75, 1.0), np.float32),
               0.5 * a * (0.5 + 0.5 * math.sin(t * 0.8 + i)))
    goz = pencere(t, O["mandos_goz"], 99, 0.8)
    d = durum("mandos", t, kirpma_araligi=6.0, kas_catik=0.35, bakis=(0.0, 0.0), ruzgar=0.3)
    d["egim"] *= 0.4
    ol = 1.0 + 0.10 * p
    portre(B, "mandos", t, d, a, olcek=ol, ton=(0.88, 0.92, 1.05))
    goz_isigi(B, "mandos", d, (0.25 + 0.6 * goz) * a, (0.55, 0.7, 1.0), olcek=ol)


# ------------------------------------------------------------------ 8 · Nienna

def s_nienna(taban, ayrinti, B, t, a):
    O = B.O
    p = ilerle(B, 8, t)
    umut = puruzsuz((t - O["nienna_isik"]) / 1.5)
    dikey(taban, [(-0.5, (0.10, 0.11, 0.14)), (0.0, (0.16, 0.17, 0.21)), (0.5, (0.04, 0.045, 0.06))], a)
    bulut(taban, B, t, -0.5, -0.1, (0.25, 0.26, 0.30), 0.6 * a, hiz=5, k=1)
    for i, (v, kay) in enumerate(((0.10, 0.0), (0.16, 3.0))):
        x = np.linspace(-0.4, 0.4, 40)
        yy = v - 0.05 * (0.5 + 0.5 * np.sin(x * 9 + kay)) - 0.03 * np.sin(x * 23 + kay)
        dunya(taban, list(zip(x, yy)) + [(0.4, 0.5), (-0.4, 0.5)], (0.6, 0.65, 0.8), dolgu=(0.06 + 0.02 * i,) * 3,
              guc=a, kenar_guc=0.06)
    lekele(taban, 0.05, -0.45, 0.12 + 0.2 * umut, 0.30, np.array((1.0, 0.85, 0.6), np.float32), 1.2 * umut * a)
    yagmur(ayrinti, t, 0.8 * a, adet=160, tohum=3)
    d = durum("nienna", t, uzgun=1.0 - 0.6 * umut, yas=1.0 - 0.7 * umut, gulus=0.35 * umut,
              bakis=(0.0, 0.5 - 0.9 * umut) if t > O["nienna_isik"] else (0.0, 0.5), ruzgar=0.4)
    ol = 1.0 + 0.06 * p
    portre(B, "nienna", t, d, a, olcek=ol, ton=(0.92 + 0.12 * umut, 0.94 + 0.08 * umut, 1.0))
    yagmur(B.on_katman(), t, (0.9 - 0.5 * umut) * a, adet=150, tohum=5)


# ------------------------------------------------------------------ 9 · Oromë

def _orman(taban, t, g, kayma=0.0):
    for i, (u, v, boy, faz) in enumerate(B_AGAC):
        uu = ((u + kayma * (0.5 + boy) + 0.4) % 0.8) - 0.4
        k = K.Kalem(uu, v + 0.3, boy)
        for j in range(5):
            y = -0.25 - j * 0.16
            gen = 0.16 - j * 0.028
            k.poligon([(-gen, y + 0.12), (0.0, y - 0.14), (gen, y + 0.12)])
        k.poligon([(-0.015, 0.0), (0.015, 0.0), (0.015, -0.3), (-0.015, -0.3)])
        K.isle(taban, k, (0.6, 0.75, 1.0), isik=(0.6, -0.8), guc=g, dolgu=(0.008, 0.014, 0.018), kenar_guc=0.3)


B_AGAC = [(-0.30, -0.02, 0.62, 0), (-0.21, 0.02, 0.50, 1), (-0.13, -0.03, 0.70, 2), (0.14, -0.02, 0.66, 3),
          (0.23, 0.03, 0.52, 4), (0.31, -0.01, 0.60, 5), (0.00, -0.06, 0.40, 6)]


def s_orome(taban, ayrinti, B, t, a):
    O = B.O
    p = ilerle(B, 9, t)
    a0, a1 = O["at"]
    at_g = a * kesit(t, a0, a1, 0.12)
    yuz = a * (1 - kesit(t, a0, a1, 0.12))
    dikey(taban, [(-0.5, (0.01, 0.02, 0.05)), (0.0, (0.03, 0.06, 0.10)), (0.5, (0.0, 0.01, 0.01))], a)
    lekele(taban, 0.17, -0.36, 0.035, 0.035, np.array((0.9, 0.95, 1.0), np.float32), 2.5 * a)
    lekele(taban, 0.17, -0.36, 0.15, 0.15, np.array((0.4, 0.5, 0.8), np.float32), 0.5 * a)
    B.yildiz.ciz(ayrinti, t, 0.4 * a, y_sinir=0.4 * H)
    patlamalar = [t - b for b in O["boru"] if t >= b]
    sars = sum(math.exp(-x / 0.5) for x in patlamalar)
    if yuz > 0.01:
        _orman(taban, t, yuz * (1 - 0.3 * sars))
        for j, b in enumerate(O["boru"]):
            x = t - b
            for k in range(3):
                r = (x - k * 0.18) * 0.55
                if 0 < r < 0.7:
                    halka(taban, B, 0.0, KAFA_V + 0.08, r, 0.01 + 0.03 * r, np.array((0.8, 1.0, 0.6), np.float32),
                          1.0 * yuz * (1 - r / 0.7))
        kac = puruzsuz((t - O["boru"][1] - 0.2) / 1.2)
        for i, (u, v, ol, faz) in enumerate(B.gozler):
            kirp = 1.0 if (math.sin(t * 1.3 + faz * 3) > -0.95) else 0.0
            gor = puruzsuz((t - B.z.sinir[9] - 0.6 - 0.15 * i) / 0.4) * (1 - kac) * kirp
            goz_cifti(ayrinti, u + math.copysign(0.2 * kac, u), v, ol, gor * yuz)
        d = durum("orome", t, gulus=0.5, ruzgar=1.0 + 2 * sars,
                  agiz=0.8 * sum(pencere(x, 0.0, 0.9, 0.12) for x in patlamalar),
                  kas_kalk=0.4 * sum(pencere(x, 0.0, 0.9, 0.12) for x in patlamalar))
        d["egim"] -= 3.0 * sum(pencere(x, 0.0, 0.9, 0.2) for x in patlamalar)
        ol = 1.0 + 0.06 * p
        portre(B, "orome", t, d, yuz, olcek=ol, ton=(0.85, 0.92, 1.05))
    if at_g > 0.01:
        f = (t - a0) / (a1 - a0)
        zemin_v = 0.30
        _orman(taban, t, at_g, kayma=-0.35 * f)
        dunya(taban, [(-0.4, zemin_v), (0.4, zemin_v - 0.01), (0.4, 0.5), (-0.4, 0.5)], (0.6, 0.7, 1.0),
              dolgu=(0.01, 0.015, 0.02), guc=at_g, kenar_guc=0.5)
        u = -0.42 + 0.84 * f
        K.at(taban, u, zemin_v, 0.26, t, guc=at_g, hiz=1.1)
        lekele(taban, u, zemin_v - 0.15, 0.12, 0.08, np.array((0.8, 0.9, 1.0), np.float32), 0.35 * at_g)
        ruzgar(ayrinti, t, 0.4 * at_g, adet=20)


# ------------------------------------------------------------------ 10 · Tulkas

def s_tulkas(taban, ayrinti, B, t, a):
    O, z = B.O, B.zz
    p = ilerle(B, 10, t)
    darbe = sum(math.exp(-(t - v) / 0.22) for v in O["yumruk"] if t >= v)
    dikey(taban, [(-0.5, (0.20, 0.10, 0.03)), (0.0, (0.35, 0.20, 0.06)), (0.5, (0.08, 0.04, 0.01))], a)
    lekele(taban, 0.0, KAFA_V - 0.05, 0.3, 0.3, np.array((1.0, 0.75, 0.35), np.float32), (0.5 + 0.8 * darbe) * a)
    isinlar(taban, B, t, np.array((1.0, 0.8, 0.4), np.float32), (0.35 + 0.6 * darbe) * a, 0.5, merkez=(0.0, KAFA_V))
    for j, v in enumerate(O["yumruk"]):
        x = t - v
        if 0 < x < 0.8:
            halka(taban, B, 0.0, KAFA_V + 0.15, x * 0.8, 0.01 + x * 0.04, np.array((1.0, 0.85, 0.5), np.float32),
                  2.0 * a * (1 - x / 0.8))
    kuy = t - O["kuyruklu"]
    if 0 < kuy < 3.5:
        f = puruzsuz(kuy / 1.4)
        u, v = 0.30 - 0.36 * f, -0.50 + 0.55 * f
        lekele(taban, u, v, 0.02, 0.02, np.array((1.0, 0.9, 0.6), np.float32), 4 * a * (1 - kuy / 3.5))
        for i in range(12):
            ff = max(0.0, f - i * 0.03)
            lekele(taban, 0.30 - 0.36 * ff, -0.50 + 0.55 * ff, 0.012 + 0.003 * i, 0.012 + 0.003 * i,
                   np.array((1.0, 0.75, 0.35), np.float32), 1.2 * a * (1 - i / 12) * (1 - kuy / 3.5))
        kacis = puruzsuz((kuy - 1.2) / 1.5)
        K.melkor_dev(taban, 0.20 + 0.15 * kacis, 0.30, 0.45, t, guc=0.5 * a * (1 - kacis) * puruzsuz(kuy / 0.6))
    korlar(ayrinti, t, 0.6 * a, adet=60, tohum=14, renk=np.array((1.0, 0.8, 0.4), np.float32), yukselis=35)
    k = kahkaha(t, O["kahkaha"], 1.0) if t < z.c(10, 2) + 1.5 else kahkaha(t, O["kahkaha"],
                                                                               1 - puruzsuz((t - z.c(10, 2) - 1.5) / 0.5))
    if not k:
        k = {"gulus": 0.6}
    d = durum("tulkas", t, ruzgar=0.8, kas_kalk=0.2, **k)
    d["egim"] += 2.0 * darbe
    ol = 1.0 + 0.06 * p
    portre(B, "tulkas", t, d, a, olcek=ol * (1 + 0.02 * darbe), ton=(1.05 + 0.2 * darbe, 1.0 + 0.1 * darbe, 0.92))
    korlar(B.on_katman(), t + 4, 0.7 * a, adet=24, tohum=15, renk=np.array((1.0, 0.8, 0.4), np.float32),
           yukselis=40)


# ------------------------------------------------------------------ 11 · Maiar, Gandalf ve Sauron

def s_maiar(taban, ayrinti, B, t, a):
    O = B.O
    iki = puruzsuz((t - O["iki_isim"]) / 0.6)
    g1 = a * (1 - iki)
    g2 = a * iki
    if g1 > 0.01:
        dikey(taban, [(-0.5, (0.02, 0.03, 0.10)), (0.0, (0.12, 0.10, 0.25)), (0.12, (0.35, 0.22, 0.20)),
                      (0.5, (0.03, 0.02, 0.04))], g1)
        B.yildiz.ciz(ayrinti, t, 0.7 * g1, y_sinir=0.5 * H)
        k = dunya(taban, dag(0.0, -0.12, 0.45, 0.30, 2), (0.9, 0.9, 1.0), isik=(-0.6, -0.8), dolgu=(0.06, 0.06, 0.12),
                  guc=g1, kenar_guc=0.45)
        K.detay(taban, k, sirt_cizgileri(0.0, -0.12, 0.45, 0.30, 4, 6), (0.8, 0.8, 1.0), 0.15 * g1)
        for i, ren in enumerate(TEMA.values()):
            u = -0.04 + 0.01 * i
            lekele(taban, u, -0.125 + 0.004 * abs(i - 4), 0.004, 0.004, np.array(ren, np.float32), 4 * g1)
        lekele(taban, 0.0, -0.13, 0.06, 0.05, np.array((1.0, 0.85, 0.6), np.float32), 0.6 * g1)
        ruh = t - O["ruhlar"]
        if ruh > 0:
            rng = np.random.default_rng(31)
            for i in range(160):
                faz, hiz, r0 = rng.uniform(0, 6.28), rng.uniform(0.4, 1.0), rng.uniform(0.0, 0.25)
                yas = (ruh * hiz * 0.35 + rng.uniform(0, 0.6)) % 1.2
                if yas > ruh * 0.6:
                    continue
                r = 0.02 + r0 + yas * 0.18
                aci = faz + yas * 4 * (1 if i % 2 else -1)
                u, v = r * math.cos(aci), -0.13 - yas * 0.30 + 0.3 * r * math.sin(aci)
                X, Y = W / 2 + u * H, H / 2 + v * H
                ren = list(TEMA.values())[i % 9]
                b = g1 * (1 - yas / 1.2) * (0.6 + 0.4 * math.sin(t * 5 + faz))
                cv2.circle(ayrinti, (int(X * 16), int(Y * 16)), int(2.2 * 16),
                           tuple(float(c) * 255 * b for c in ren), -1, cv2.LINE_AA, 4)
    if g2 > 0.01:
        dikey(taban, [(-0.5, (0.01, 0.01, 0.02)), (0.1, (0.04, 0.03, 0.05)), (0.5, (0.01, 0.01, 0.01))], g2)
        sis(taban, 0.14, 0.05, (0.20, 0.18, 0.22), 0.6 * g2)
        gg = g2 * (0.22 + 0.78 * puruzsuz((t - O["gandalf"]) / 0.5))
        gs = g2 * (0.22 + 0.78 * puruzsuz((t - O["sauron"]) / 0.5))
        lekele(taban, -0.13, -0.05, 0.12, 0.25, np.array((0.7, 0.75, 0.9), np.float32), 0.35 * gg)
        lekele(taban, 0.13, -0.05, 0.12, 0.25, KOR, 0.35 * gs)
        K.gezgin(taban, -0.13, 0.14, 0.40, t, guc=gg, asa_isik=0.8 + 0.2 * math.sin(t * 3))
        K.kara_lord(taban, 0.14, 0.14, 0.44, t, guc=gs, yon=-1, kor_guc=1.0 + 0.2 * math.sin(t * 2.3))
        korlar(ayrinti, t, 0.5 * gs, adet=30, tohum=17, alan=(W * 0.5, W), yukselis=60)


SAHNELER = {"kanca": s_kanca, "tanitim": s_tanitim, "manwe": s_manwe, "varda": s_varda, "ulmo": s_ulmo,
            "aule": s_aule, "yavanna": s_yavanna, "mandos": s_mandos, "nienna": s_nienna, "orome": s_orome,
            "tulkas": s_tulkas, "maiar": s_maiar}
NEBULA = {ad: ((0.0, 0.0, 0.0), 0.0) for ad in SAHNELER}


def son_islem(c, kare, t):
    O = c.B.O
    flaslar = [(x, 0.30) for x in O["yuzler"][1:]] + [(O["kanca_b"], 0.45), (O["kanca_c"], 0.45),
                                                        (O["melkor_cokus"], 0.35), (O["manwe_gecis"], 0.55),
                                                        (O["sise"][0], 0.3), (O["sise"][1], 0.25), (O["at"][0], 0.25),
                                                        (O["at"][1], 0.25), (O["iki_isim"], 0.2),
                                                        (O["gandalf"], 0.25), (O["sauron"], 0.3)]
    flaslar += [(x, 0.35) for x in O["boru"]] + [(x, 0.3) for x in O["yumruk"]] + [(x, 0.10) for x in O["ors"]]
    flas = sum(g * math.exp(-(t - x) / 0.22) for x, g in flaslar if t >= x)
    if flas > 0.01:
        flas = min(flas, 0.8)
        kare = cv2.addWeighted(kare, 1 - flas, np.full_like(kare, 255), flas, 0)
    sars = (sum(math.exp(-(t - x) / 0.35) for x in O["boru"] + O["yumruk"] if t >= x) * 0.9
            + sum(math.exp(-(t - x) / 0.15) for x in O["ors"] if t >= x) * 0.35
            + pencere(t, O["melkor_cokus"] - 0.1, O["melkor_cokus"] + 0.6, 0.15) * 0.6)
    if sars > 0.02:
        rs = np.random.default_rng(int(t * FPS))
        M = np.float32([[1, 0, rs.uniform(-9, 9) * sars], [0, 1, rs.uniform(-9, 9) * sars]])
        kare = cv2.warpAffine(kare, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return kare
