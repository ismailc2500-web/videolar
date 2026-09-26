"""Portreli bölümlerin (3. bölüm ve sonrası) ortak sahne araçları.

Zaman yardımcısı, gökyüzü/bulut/dağ/deniz dibi gibi arka plan çizimleri, portreyi
ön plana bindirme, göz ışığı, ön plan parçacıkları (yağmur, kabarcık, kor, çiçek)
ve ifade anahtarlama (ifade, kahkaha, durum) burada toplanır.
"""
import math

import cv2
import numpy as np

import karakterler as K
import portre as P
from motor import H, KOR, W, h, lekele, pencere, puruzsuz, w

S_PORTRE = 400
KAFA_Y = 0.36 * H
ALT = (0.62 * H, 1.30 * H)
SOLDUR = (0.80 * H, 1.0 * H)
V_KOLON = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]
KAFA_V = KAFA_Y / H - 0.5

TOHUM = {}
"""Karakter adı → canlılık tohumu (göz kırpma/bakış düzeni); bölümler kendi değerlerini ekler."""


def tohum(ad):
    return TOHUM.get(ad, sum(ord(c) * (i + 1) for i, c in enumerate(ad)) % 997)


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
           pr=None, ekran=False, maske=None):
    """Portreyi ön plana bindirir. maske: (H, W) 0-1 dizisi verilirse yalnızca o bölgeye çizer (bölünmüş ekran)."""
    if a <= 0.003:
        return
    pr = pr or B.portre[ad]

    def ciz(kare):
        if maske is None:
            P.yerlestir(kare, pr.kare(t, d), pr, x, y, olcek, a, ton, aydinlik, alt, soldur, ekran)
            return
        once = kare.copy()
        P.yerlestir(kare, pr.kare(t, d), pr, x, y, olcek, a, ton, aydinlik, alt, soldur, ekran)
        m = maske[..., None]
        kare[:] = (once * (1 - m) + kare * m).astype(np.uint8)

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
    """Karakterin bu karedeki hâli: kendiliğinden canlılık (bosta) + verilen ifadeler."""
    egim_ek, nefes_ek = ek.pop("egim_ek", 0.0), ek.pop("nefes_ek", 0.0)
    d = P.bosta(t, tohum(ad), **ek)
    d["egim"] += egim_ek
    d["nefes"] += nefes_ek
    return d


def arka_isik(taban, u, v, renk, guc, r=0.22):
    lekele(taban, u, v, r, r * 1.1, np.asarray(renk, np.float32), guc)


def kesit(t, bas, bit, g=0.12):
    return pencere(t, bas, bit, g)
