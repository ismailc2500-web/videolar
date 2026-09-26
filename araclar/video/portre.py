"""Yüzü görünen, animasyonlu portre karakterleri (kesme animasyon düzeneği).

Her karakter katmanlardan oluşur: arka saç, gövde/kıyafet, boyun ve yüz, gözler,
kaşlar, ağız, sakal, bıyık, ön saç ve taç. Durağan katmanlar bir kez boyanıp
kırpılarak önbelleğe alınır; gözler, kaşlar ve ağız her karede küçük bir tuvalde
yeniden çizilir. Baş grubu boyun ekseni etrafında döner, saçlar ve sakal
dalgalanır, gövde nefes alır.

Birimler: baş yüksekliği = 1; başın merkezi (0, 0); y aşağı doğru artar.
Gölgelendirme: maskenin bulanık hâli yükseklik alanı kabul edilir ("yastık"),
normalinden ışık ve kenar ışığı hesaplanır; üstüne düzlemsel ışık eğimi, cel
tarzı ton basamakları ve ince kontur çizgisi eklenir.
"""
import math

import cv2
import numpy as np


def _p(x):
    return np.asarray(x, np.float32)


def puruz(x, a=0.0, b=1.0):
    x = np.clip((x - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def catmull(pts, n=8, kapali=False):
    """Noktalardan geçen yumuşak eğri (Catmull-Rom)."""
    p = _p(pts)
    p = np.vstack([p[-1:], p, p[:2]]) if kapali else np.vstack([p[:1], p, p[-1:]])
    cikti = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            cikti.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not kapali:
        cikti.append(p[-2])
    return np.array(cikti, np.float32)


def ayna(yarim):
    """x >= 0 yarım çevreyi (üstten alta) simetrik kapalı çevreye çevirir."""
    y = _p(yarim)
    sol = y[::-1].copy()
    sol[:, 0] *= -1
    return np.vstack([y, sol[1:-1]])


class Katman:
    """Kırpılmış, önceden çarpılmış RGBA katman (tuval koordinatlarında konumlu)."""

    def __init__(self, rgb, a, x0, y0):
        self.rgb, self.a, self.x0, self.y0 = rgb, a, x0, y0


class Tuval:
    """Yerel birimlerle çizilen, önceden çarpılmış (premultiplied) RGBA tuval."""

    def __init__(self, S, x0, x1, y0, y1):
        self.S = S
        self.x0, self.y0 = x0, y0
        self.g = int(round((x1 - x0) * S))
        self.y = int(round((y1 - y0) * S))
        self.rgb = np.zeros((self.y, self.g, 3), np.float32)
        self.a = np.zeros((self.y, self.g), np.float32)

    def px(self, pts):
        a = _p(pts).reshape(-1, 2)
        return np.stack([(a[:, 0] - self.x0) * self.S, (a[:, 1] - self.y0) * self.S], -1)

    def bos_maske(self):
        return np.zeros((self.y, self.g), np.uint8)

    def poli(self, m, pts, deger=255):
        cv2.fillPoly(m, [np.int32(self.px(pts) * 16)], deger, cv2.LINE_AA, 4)
        return m

    def cizgi(self, m, pts, kalinlik, deger=255, kapali=False):
        k = max(1, int(round(kalinlik * self.S)))
        cv2.polylines(m, [np.int32(self.px(pts) * 16)], kapali, deger, k, cv2.LINE_AA, 4)
        return m

    def elips(self, m, x, y, rx, ry, aci=0.0, deger=255):
        c = self.px([(x, y)])[0]
        cv2.ellipse(m, (int(c[0] * 16), int(c[1] * 16)), (max(1, int(rx * self.S * 16)), max(1, int(ry * self.S * 16))),
                    aci, 0, 360, deger, -1, cv2.LINE_AA, 4)
        return m

    def konik(self, m, pts, r0, r1, deger=255):
        n = len(pts)
        for i, (x, y) in enumerate(pts):
            r = r0 + (r1 - r0) * i / max(1, n - 1)
            self.elips(m, x, y, r, r, deger=deger)
        return m

    def ekle(self, rgb, a, y0=0, x0=0):
        hh, ww = a.shape
        y1, x1 = min(self.y, y0 + hh), min(self.g, x0 + ww)
        ky0, kx0 = max(0, y0), max(0, x0)
        if y1 <= ky0 or x1 <= kx0:
            return
        s_rgb = rgb[ky0 - y0:y1 - y0, kx0 - x0:x1 - x0]
        s_a = a[ky0 - y0:y1 - y0, kx0 - x0:x1 - x0]
        self.rgb[ky0:y1, kx0:x1] = self.rgb[ky0:y1, kx0:x1] * (1 - s_a[..., None]) + s_rgb
        self.a[ky0:y1, kx0:x1] = self.a[ky0:y1, kx0:x1] * (1 - s_a) + s_a

    def katman_ekle(self, k):
        self.ekle(k.rgb, k.a, k.y0, k.x0)

    def _kutu(self, m, pay):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return None
        return (max(0, xs.min() - pay), min(self.g, xs.max() + pay + 1), max(0, ys.min() - pay),
                min(self.y, ys.max() + pay + 1))

    def boya(self, m, renk, golge=None, isik=(-0.55, -0.6, 0.58), sigma=0.06, yumusaklik=(0.40, 0.56),
             parlama=0.0, kenar=None, kenar_yon=(0.75, -0.25), doku=None, opak=1.0, isima=None, duzlem=0.0,
             hat=0.0, hat_renk=None, sicak_gecis=True, sekil="yastik", sekil_agirlik=0.75):
        """Maskeyi yastık gölgelendirmesiyle boyar ve tuvale ekler."""
        sp = max(1.0, sigma * self.S)
        kutu = self._kutu(m, int(3 * sp) + 3)
        if kutu is None:
            return
        x0, x1, y0, y1 = kutu
        mm = m[y0:y1, x0:x1].astype(np.float32) / 255.0
        yuk = cv2.GaussianBlur(mm, (0, 0), sp)
        gx = cv2.Sobel(yuk, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(yuk, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        k = sp * 2.2
        nx, ny = -gx * k, -gy * k
        n = np.sqrt(nx * nx + ny * ny + 1)
        nx, ny, nz = nx / n, ny / n, 1 / n
        L = _p(isik) / np.linalg.norm(isik)
        ys_, xs_ = np.nonzero(mm > 0.5)
        if sekil != "yastik" and len(xs_):
            cx, cy = (xs_.min() + xs_.max()) / 2, (ys_.min() + ys_.max()) / 2
            rx, ry = max(1.0, (xs_.max() - xs_.min()) / 2 * 1.08), max(1.0, (ys_.max() - ys_.min()) / 2 * 1.08)
            XX = (np.arange(x1 - x0, dtype=np.float32)[None, :] - cx) / rx
            YY = (np.arange(y1 - y0, dtype=np.float32)[:, None] - cy) / ry
            if sekil == "silindir":
                YY = np.zeros_like(YY)
            ex, ey = np.clip(XX * np.ones_like(YY), -0.98, 0.98), np.clip(YY * np.ones_like(XX), -0.98, 0.98)
            ez = np.sqrt(np.clip(1 - ex * ex - ey * ey, 0.02, 1))
            w_ = sekil_agirlik
            nx, ny, nz = nx * (1 - w_) + ex * w_, ny * (1 - w_) + ey * w_, nz * (1 - w_) + ez * w_
            n = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / n, ny / n, nz / n
        lam = nx * L[0] + ny * L[1] + nz * L[2]
        if duzlem and len(xs_):
            cx, cy = xs_.mean(), ys_.mean()
            rx, ry = max(1.0, xs_.std() * 1.7), max(1.0, ys_.std() * 1.7)
            XX = (np.arange(x1 - x0, dtype=np.float32)[None, :] - cx) / rx
            YY = (np.arange(y1 - y0, dtype=np.float32)[:, None] - cy) / ry
            lam = lam + duzlem * (XX * L[0] + YY * L[1])
        lam = np.clip(lam, 0, 1)
        ton = puruz(lam, *yumusaklik)
        renk = _p(renk)
        golge = _p(golge) if golge is not None else renk * _p([0.52, 0.42, 0.58])
        col = golge + (renk - golge) * ton[..., None]
        if sicak_gecis:
            gecis = puruz(lam, yumusaklik[0] - 0.08, yumusaklik[0]) * (1 - puruz(lam, yumusaklik[1], yumusaklik[1] + 0.08))
            col += gecis[..., None] * _p([0.035, 0.008, -0.005])
        if doku is not None:
            col = col * (1 + doku[y0:y1, x0:x1, None])
        if parlama:
            col += (puruz(lam, 0.90, 0.99) * parlama)[..., None]
        if kenar is not None:
            r = np.clip(nx * kenar_yon[0] + ny * kenar_yon[1], 0, 1) ** 2 * (1 - nz) * 2.4
            col += r[..., None] * _p(kenar)
        if isima is not None:
            col += _p(isima)
        a = mm * opak
        if hat:
            kal = max(1, int(round(self.S * 0.006)))
            ic = cv2.erode(mm, np.ones((2 * kal + 1, 2 * kal + 1), np.uint8))
            kenar_bandi = np.clip(mm - ic, 0, 1) * hat
            hr = _p(hat_renk) if hat_renk is not None else golge * 0.35
            col = col * (1 - kenar_bandi[..., None]) + hr * kenar_bandi[..., None]
        self.ekle(np.clip(col, 0, 4) * a[..., None], a, y0, x0)

    def duz(self, m, renk, opak=1.0, bulanik=0.0, carp=False):
        """Gölgesiz düz renk (çizgi, gölge lekesi, parıltı). carp=True: alttakini koyulaştırır."""
        kutu = self._kutu(m, int(3 * bulanik * self.S) + 2)
        if kutu is None:
            return
        x0, x1, y0, y1 = kutu
        mm = m[y0:y1, x0:x1].astype(np.float32) / 255.0
        if bulanik:
            mm = cv2.GaussianBlur(mm, (0, 0), bulanik * self.S)
        a = mm * opak
        if carp:
            self.rgb[y0:y1, x0:x1] *= (1 - a[..., None] * (1 - _p(renk)))
            return
        self.ekle(np.ones_like(a)[..., None] * _p(renk) * a[..., None], a, y0, x0)

    def kirp(self):
        ys, xs = np.nonzero(self.a > 0.002)
        if len(xs) == 0:
            return Katman(np.zeros((1, 1, 3), np.float32), np.zeros((1, 1), np.float32), 0, 0)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return Katman(self.rgb[y0:y1, x0:x1].copy(), self.a[y0:y1, x0:x1].copy(), x0, y0)


# ------------------------------------------------------------------ yüz geometrisi

YUZ_ERKEK = [(0, -0.50), (0.17, -0.48), (0.30, -0.40), (0.352, -0.25), (0.362, -0.08), (0.352, 0.08), (0.322, 0.22),
             (0.265, 0.34), (0.18, 0.44), (0.085, 0.50), (0, 0.515)]
YUZ_KADIN = [(0, -0.50), (0.17, -0.48), (0.29, -0.40), (0.342, -0.25), (0.350, -0.08), (0.336, 0.08), (0.300, 0.21),
             (0.235, 0.33), (0.15, 0.42), (0.065, 0.465), (0, 0.475)]


def yuz_cevresi(tip):
    return ayna(catmull(YUZ_KADIN if tip == "kadin" else YUZ_ERKEK, 8))


def sac_tutami(x0, y0, x1, y1, gen, egri=0.0, n=6):
    """Kökten uca incelen, hafif kıvrık saç tutamı çevresi."""
    t = np.linspace(0, 1, n)
    cx = x0 + (x1 - x0) * t + egri * np.sin(t * math.pi)
    cy = y0 + (y1 - y0) * t
    dx, dy = (y1 - y0), -(x1 - x0)
    L = math.hypot(dx, dy) or 1
    dx, dy = dx / L, dy / L
    w = gen * (1 - t) ** 0.8
    sol = np.stack([cx + dx * w, cy + dy * w], -1)
    sag = np.stack([cx - dx * w, cy - dy * w], -1)
    return catmull(np.vstack([sol, sag[::-1]]), 3, kapali=True)


class Portre:
    """Parametrik, animasyonlu portre. `tanim` sözlüğü görünümü belirler."""

    X0, X1, Y0, Y1 = -1.35, 1.35, -1.25, 3.3

    def __init__(self, tanim, S=380):
        self.T = dict(VARSAYILAN)
        self.T.update(tanim)
        self.S = S
        self.boyun_pivot = (0.0, 0.62)
        self._katmanlar()

    def _tuval(self, x0=None, x1=None, y0=None, y1=None):
        return Tuval(self.S, self.X0 if x0 is None else x0, self.X1 if x1 is None else x1,
                     self.Y0 if y0 is None else y0, self.Y1 if y1 is None else y1)

    def _ofset(self, x, y):
        return int(round((x - self.X0) * self.S)), int(round((y - self.Y0) * self.S))

    # -- durağan katmanlar
    def _katmanlar(self):
        T = self.T
        isik, kenar = T["isik"], T["kenar"]
        ten = _p(T["ten"])
        ten_golge = ten * _p([0.74, 0.60, 0.62]) + _p([0.02, 0.0, 0.03])
        self.ten, self.ten_golge = ten, ten_golge
        sac = _p(T["sac"])
        sac_golge = sac * _p([0.45, 0.40, 0.52])
        tip = T["tip"]

        # arka saç
        tv = self._tuval()
        if T["sac_boy"] > 0:
            L, gen = T["sac_boy"], T["sac_gen"]
            yarim = [(0, -0.62), (0.26, -0.60), (0.42, -0.46), (0.47 * gen, -0.20), (0.48 * gen, 0.15),
                     (0.50 * gen + 0.04 * L, 0.30 + L * 0.55), (0.46 * gen + 0.06 * L, 0.30 + L),
                     (0.28, 0.36 + L * 1.02), (0.10, 0.32 + L), (0.0, 0.35 + L)]
            m = tv.poli(tv.bos_maske(), ayna(catmull(yarim, 6)))
            for i in range(9):
                x = -0.42 * gen + i * 0.105 * gen
                tv.poli(m, sac_tutami(x, 0.2 + L * 0.6, x + 0.02 * (i - 4), 0.40 + L * 1.05, 0.06))
            tv.boya(m, sac * 0.62, sac_golge * 0.55, isik=isik, sigma=0.10, kenar=kenar, yumusaklik=(0.25, 0.75),
                    sekil="elipsoid", sekil_agirlik=0.5,
                    doku=self._sac_dokusu(tv, 11, 0.16, dikey=True), hat=0.5)
        if T.get("kukuleta") is not None:
            kk = _p(T["kukuleta"])
            yarim = [(0.0, -0.78), (0.30, -0.74), (0.52, -0.52), (0.60, -0.15), (0.62, 0.25), (0.68, 0.62),
                     (0.80, 0.95), (0.55, 1.05), (0.0, 0.95)]
            m = tv.poli(tv.bos_maske(), ayna(catmull(yarim, 6)))
            tv.boya(m, kk * 0.75, kk * 0.35, isik=isik, sigma=0.10, kenar=kenar, yumusaklik=(0.25, 0.75),
                    sekil="elipsoid", sekil_agirlik=0.5, doku=self._kivrim_dokusu(tv, 9), hat=0.6)
        self.sac_arka = tv.kirp()

        # gövde ve kıyafet
        tv = self._tuval()
        omuz = T["omuz"]
        m = tv.poli(tv.bos_maske(), [(-0.145, 0.30), (0.145, 0.30), (0.16, 0.82), (-0.16, 0.82)])
        tv.boya(m, ten * 0.93, ten_golge * 0.9, isik=isik, sigma=0.05, yumusaklik=(0.25, 0.75), hat=0.4,
                sekil="silindir", sekil_agirlik=0.8)
        m = tv.poli(tv.bos_maske(), catmull([(-0.16, 0.34), (0.16, 0.34), (0.17, 0.52), (0.0, 0.60), (-0.17, 0.52)], 4, True))
        tv.duz(m, ten_golge * 0.7, opak=0.55, bulanik=0.035, carp=True)
        yarim = [(0.13, 0.70), (0.28, 0.76), (0.50 * omuz, 0.84), (0.70 * omuz, 0.95), (0.80 * omuz, 1.15),
                 (0.84 * omuz, 1.60), (0.90 * omuz, 2.45), (0.93 * omuz, 3.3)]
        govde = np.vstack([catmull(yarim, 8), [(0, 3.3)]])
        cevre = np.vstack([govde, (govde[::-1] * _p([-1, 1]))[1:]])
        m = tv.poli(tv.bos_maske(), cevre)
        kiy = _p(T["kiyafet"])
        tv.boya(m, kiy, T.get("kiyafet_golge") or kiy * _p([0.45, 0.42, 0.58]), isik=isik, sigma=0.10, kenar=kenar,
                yumusaklik=(0.25, 0.75), doku=self._kivrim_dokusu(tv, 3), hat=0.6, sekil="silindir", sekil_agirlik=0.7)
        yaka = T["yaka"]
        yaka_pts = None
        if yaka == "v":
            yaka_pts = [(-0.16, 0.72), (0.0, 1.12), (0.16, 0.72)]
            m = tv.poli(tv.bos_maske(), [(-0.16, 0.72), (0.16, 0.72), (0.0, 1.12)])
            tv.boya(m, ten * 0.92, ten_golge * 0.85, isik=isik, sigma=0.05, duzlem=0.4)
        elif yaka == "kare":
            yaka_pts = catmull([(-0.28, 0.75), (-0.24, 0.96), (0, 1.01), (0.24, 0.96), (0.28, 0.75)], 6)
            m = tv.poli(tv.bos_maske(), yaka_pts)
            tv.boya(m, ten * 0.92, ten_golge * 0.85, isik=isik, sigma=0.05, duzlem=0.4)
        if T["pelerin"] is not None:
            pel = _p(T["pelerin"])
            for s in (-1, 1):
                pts = [(0.17 * s, 0.72), (0.40 * s * omuz, 0.80), (0.72 * s * omuz, 0.93), (0.86 * s * omuz, 1.25),
                       (0.97 * s * omuz, 3.3), (0.70 * s * omuz, 3.3), (0.60 * s * omuz, 1.45), (0.36 * s, 0.96)]
                m = tv.poli(tv.bos_maske(), catmull(pts, 5, kapali=True))
                tv.boya(m, pel, pel * _p([0.50, 0.48, 0.62]), isik=isik, sigma=0.10, kenar=kenar,
                        yumusaklik=(0.25, 0.75), doku=self._kivrim_dokusu(tv, 5 + s), hat=0.6, sekil="silindir",
                        sekil_agirlik=0.5)
        if T["sus"] is not None:
            renk = _p(T["sus"])
            if yaka_pts is not None:
                m = tv.cizgi(tv.bos_maske(), yaka_pts, 0.032)
                tv.boya(m, renk, renk * 0.35, isik=isik, sigma=0.012, parlama=0.7, hat=0.4)
            m = tv.cizgi(tv.bos_maske(), catmull([(-0.62 * omuz, 1.02), (-0.3, 1.18), (0, 1.22), (0.3, 1.18),
                                                  (0.62 * omuz, 1.02)], 6), 0.012)
            tv.duz(m, renk, opak=0.0)
            gy = 1.20 if yaka == "v" else 1.07
            m = tv.elips(tv.bos_maske(), 0.0, gy, 0.058, 0.058)
            tv.boya(m, renk, renk * 0.35, isik=isik, sigma=0.015, parlama=0.8, hat=0.5)
            m = tv.elips(tv.bos_maske(), 0.0, gy, 0.036, 0.036)
            tv.boya(m, T["tas"], _p(T["tas"]) * 0.3, isik=isik, sigma=0.012, parlama=1.2, isima=_p(T["tas"]) * 0.25)
        self.govde = tv.kirp()

        # kafa: kulaklar, yüz, burun, yanaklar, göz çukuru
        tv = self._tuval(-0.8, 0.8, -1.0, 0.8)
        for s in (-1, 1):
            if T["kulak"] == "sivri":
                kul = [(0.33 * s, -0.08), (0.46 * s, -0.26), (0.43 * s, -0.10), (0.41 * s, 0.06), (0.37 * s, 0.15),
                       (0.33 * s, 0.12)]
            else:
                kul = [(0.335 * s, -0.08), (0.395 * s, -0.12), (0.425 * s, -0.03), (0.415 * s, 0.08),
                       (0.385 * s, 0.15), (0.34 * s, 0.13)]
            m = tv.poli(tv.bos_maske(), catmull(kul, 5, kapali=True))
            tv.boya(m, ten * 0.96, ten_golge, isik=isik, sigma=0.03, kenar=kenar, hat=0.6)
        cevre = yuz_cevresi(tip)
        m = tv.poli(tv.bos_maske(), cevre)
        tv.boya(m, ten, ten_golge, isik=isik, sigma=0.08, kenar=kenar, yumusaklik=(0.30, 0.60), parlama=0.05,
                hat=0.55, sekil="elipsoid", sekil_agirlik=0.8)
        yuz_m = m
        for s in (-1, 1):
            tv.duz(tv.elips(tv.bos_maske(), 0.20 * s, 0.13, 0.085, 0.05), T["allik"], opak=0.17, bulanik=0.04)
            tv.duz(tv.elips(tv.bos_maske(), 0.145 * s, -0.055, 0.085, 0.035), ten_golge * 0.85, opak=0.30, bulanik=0.03,
                   carp=True)
            tv.duz(tv.elips(tv.bos_maske(), 0.30 * s, 0.10, 0.05, 0.16), ten_golge * 0.85, opak=0.25, bulanik=0.04,
                   carp=True)
        burun = catmull([(0.02, -0.06), (0.036, 0.05), (0.055, 0.13), (0.048, 0.178), (0.02, 0.192)], 5)
        tv.duz(tv.poli(tv.bos_maske(), np.vstack([burun, [(0.0, 0.19), (0.0, -0.06)]])), ten_golge * 0.8, opak=0.40,
               bulanik=0.012, carp=True)
        tv.duz(tv.elips(tv.bos_maske(), 0.0, 0.198, 0.055, 0.013), ten_golge * 0.65, opak=0.45, bulanik=0.01, carp=True)
        for s in (-1, 1):
            tv.duz(tv.elips(tv.bos_maske(), 0.03 * s, 0.177, 0.013, 0.006, aci=15 * s), ten_golge * 0.35, opak=0.75)
        tv.duz(tv.elips(tv.bos_maske(), -0.01, 0.145, 0.016, 0.011), (1, 1, 1), opak=0.22, bulanik=0.008)
        tv.duz(tv.cizgi(tv.bos_maske(), catmull([(-0.04, 0.17), (-0.05, 0.19), (-0.03, 0.2)], 4), 0.005),
               ten_golge * 0.5, opak=0.5)
        tv.duz(tv.elips(tv.bos_maske(), 0.0, 0.40, 0.08, 0.03), ten_golge * 0.8, opak=0.18, bulanik=0.03, carp=True)
        if T["sac_on"] != "yok":
            sm = self._on_sac_maskesi(tv)
            kay = int(0.035 * self.S)
            golge = np.zeros_like(sm)
            golge[kay:] = sm[:-kay]
            tv.duz(cv2.bitwise_and(golge, yuz_m), ten_golge * 0.55, opak=0.6, bulanik=0.03, carp=True)
        if T.get("kukuleta") is not None:
            cer = tv.poli(tv.bos_maske(), ayna(catmull([(0.0, -0.60), (0.24, -0.55), (0.40, -0.36), (0.44, 0.02),
                                                         (0.41, 0.45)], 6)))
            ic_m = tv.poli(tv.bos_maske(), ayna(catmull([(0.0, -0.50), (0.18, -0.46), (0.30, -0.32), (0.33, 0.0),
                                                          (0.30, 0.40)], 6)))
            tv.duz(cv2.bitwise_and(cv2.subtract(cer, ic_m), yuz_m), ten_golge * 0.45, opak=0.7, bulanik=0.05, carp=True)
        tac_golge = T.get("tac_golge")
        if tac_golge:
            tv.duz(cv2.bitwise_and(tv.elips(tv.bos_maske(), 0.0, tac_golge, 0.40, 0.06), yuz_m), ten_golge * 0.5,
                   opak=0.5, bulanik=0.03, carp=True)
        self.kafa = tv.kirp()

        # sakal ve bıyık
        tv = self._tuval(-0.8, 0.8, -1.0, 1.9)
        if T["sakal"] > 0:
            L = T["sakal"]
            sakal_c = _p(T["sakal_renk"])
            dis = [(0.37, -0.06), (0.36, 0.12), (0.32, 0.30), (0.24, 0.45 + L * 0.35), (0.13, 0.52 + L * 0.8),
                   (0.04, 0.54 + L), (0.0, 0.555 + L)]
            ic = [(0.0, 0.405), (0.07, 0.39), (0.125, 0.335), (0.165, 0.26), (0.225, 0.20), (0.285, 0.11), (0.31, -0.06)]
            ic_c = catmull(ic, 5)
            sag = np.vstack([catmull(dis, 6), ic_c[::-1][1:]])
            sag_ters = sag[::-1].copy()
            sag_ters[:, 0] *= -1
            m = tv.poli(tv.bos_maske(), np.vstack([catmull(dis, 6)[::-1] * _p([-1, 1]), catmull(dis, 6)]))
            delik = tv.poli(tv.bos_maske(), np.vstack([ic_c, ic_c[::-1] * _p([-1, 1])]))
            m = cv2.subtract(m, delik)
            for i in range(7):
                x = -0.24 + i * 0.08
                tv.poli(m, sac_tutami(x * (1 - 0.3 * L), 0.35 + L * 0.5, x * 0.5, 0.56 + L * 1.02, 0.05))
            yy = (np.arange(tv.y, dtype=np.float32)[:, None] / self.S + tv.y0)
            xx = (np.arange(tv.g, dtype=np.float32)[None, :] / self.S + tv.x0)
            yumusak_ust = puruz(yy - 0.06 * np.abs(xx) / 0.36, -0.02, 0.16)
            m = (m.astype(np.float32) * yumusak_ust).astype(np.uint8)
            tv.boya(m, sakal_c, sakal_c * _p([0.50, 0.46, 0.60]), isik=isik, sigma=0.08, kenar=kenar,
                    yumusaklik=(0.25, 0.75), sekil="elipsoid", sekil_agirlik=0.6,
                    doku=self._sac_dokusu(tv, 23, 0.20, dikey=True), hat=0.25)
        self.sakal = tv.kirp()
        tv = self._tuval(-0.8, 0.8, -1.0, 1.9)
        if T["sakal"] > 0:
            yarim = [(0.0, 0.232), (0.05, 0.226), (0.10, 0.25), (0.135, 0.30), (0.145, 0.35), (0.11, 0.31),
                     (0.06, 0.277), (0.0, 0.27)]
            biyik = np.vstack([_p(yarim), (_p(yarim)[::-1] * _p([-1, 1]))])
            m = tv.poli(tv.bos_maske(), catmull(biyik, 3, kapali=True))
            tv.boya(m, sakal_c, sakal_c * 0.5, isik=isik, sigma=0.02, kenar=kenar,
                    doku=self._sac_dokusu(tv, 29, 0.16, dikey=True), hat=0.5)
        self.biyik = tv.kirp()

        # ön saç
        tv = self._tuval(-0.8, 0.8, -1.0, 1.9)
        if T["sac_on"] != "yok":
            m = self._on_sac_maskesi(tv)
            tv.boya(m, sac, sac_golge, isik=isik, sigma=0.07, kenar=kenar, parlama=0.10, yumusaklik=(0.28, 0.72),
                    sekil="elipsoid", sekil_agirlik=0.6, doku=self._sac_dokusu(tv, 5, 0.22), hat=0.6)
            parilti = tv.cizgi(tv.bos_maske(), catmull([(-0.30, -0.42), (-0.15, -0.53), (0.02, -0.56), (0.20, -0.52)], 6),
                               0.045)
            tv.duz(cv2.bitwise_and(parilti, m), (1, 1, 1), opak=0.22, bulanik=0.02)
            self._sac_telleri(tv, m, sac, sac_golge)
        self.sac_on = tv.kirp()

        # taç / başlık
        tv = self._tuval(-0.8, 0.8, -1.25, 1.9)
        if T.get("kukuleta") is not None:
            kk = _p(T["kukuleta"])
            dis = catmull([(-0.50, 0.55), (-0.53, 0.05), (-0.47, -0.42), (-0.28, -0.66), (0.0, -0.72), (0.28, -0.66),
                           (0.47, -0.42), (0.53, 0.05), (0.50, 0.55)], 6)
            ic = catmull([(0.37, 0.45), (0.40, 0.02), (0.36, -0.36), (0.20, -0.53), (0.0, -0.57), (-0.20, -0.53),
                          (-0.36, -0.36), (-0.40, 0.02), (-0.37, 0.45)], 6)
            m = tv.poli(tv.bos_maske(), np.vstack([dis, ic]))
            tv.boya(m, kk, kk * 0.45, isik=isik, sigma=0.05, kenar=kenar, yumusaklik=(0.28, 0.72),
                    sekil="elipsoid", sekil_agirlik=0.5, doku=self._kivrim_dokusu(tv, 11), hat=0.6)
        if T["tac"] is not None:
            T["tac"](self, tv)
        self.tac = tv.kirp()

    def _on_sac_maskesi(self, tv):
        T = self.T
        ust = [(-0.39, -0.02), (-0.42, -0.25), (-0.34, -0.49), (-0.17, -0.61), (0.0, -0.635), (0.17, -0.61),
               (0.34, -0.49), (0.42, -0.25), (0.39, -0.02)]
        if T["sac_on"] == "orta":
            alt = [(0.37, -0.06), (0.35, -0.24), (0.26, -0.35), (0.12, -0.385), (0.0, -0.34), (-0.12, -0.385),
                   (-0.26, -0.35), (-0.35, -0.24), (-0.37, -0.06)]
        elif T["sac_on"] == "yan":
            alt = [(0.37, -0.06), (0.34, -0.28), (0.20, -0.37), (0.02, -0.33), (-0.14, -0.41), (-0.28, -0.34),
                   (-0.35, -0.20), (-0.37, -0.06)]
        else:
            alt = [(0.36, -0.16), (0.30, -0.35), (0.16, -0.43), (0.0, -0.44), (-0.16, -0.43), (-0.30, -0.35),
                   (-0.36, -0.16)]
        m = tv.poli(tv.bos_maske(), np.vstack([catmull(ust, 6), catmull(alt, 6)]))
        if T["sac_boy"] > 0.2:
            L = min(T["sac_boy"], 1.0)
            for s in (-1, 1):
                tv.poli(m, sac_tutami(0.385 * s, -0.22, 0.50 * s, 0.30 + 0.45 * L, 0.038, egri=0.03 * s, n=8))
                tv.poli(m, sac_tutami(0.41 * s, -0.12, 0.56 * s, 0.18 + 0.40 * L, 0.03, egri=0.02 * s, n=8))
        return m

    def _sac_telleri(self, tv, m, sac, sac_golge):
        """Ön saça ayrımdan yanlara akan tel çizgileri (koyu oluklar, açık parıltılar)."""
        T = self.T
        ayrim = 0.0 if T["sac_on"] == "orta" else (0.02 if T["sac_on"] == "yan" else None)
        koyu, acik = tv.bos_maske(), tv.bos_maske()
        for i in range(7):
            f = 0.12 + i * 0.13
            for s in (-1, 1):
                x0 = (ayrim if ayrim is not None else 0.0) + 0.015 * s
                pts = catmull([(x0, -0.63 + 0.02 * f), (0.16 * s * f + x0, -0.60 + 0.05 * f),
                               (0.30 * s * f + x0 * 0.5, -0.50 + 0.10 * f), (0.40 * s * f, -0.30 + 0.12 * f),
                               (0.40 * s * (0.5 + 0.5 * f), -0.02)], 6)
                tv.cizgi(koyu if i % 2 == 0 else acik, pts, 0.008 if i % 2 == 0 else 0.006)
        tv.duz(cv2.bitwise_and(koyu, m), sac_golge * 0.8, opak=0.35, bulanik=0.006)
        tv.duz(cv2.bitwise_and(acik, m), _p(sac) * 1.25 + 0.05, opak=0.18, bulanik=0.006)
        if ayrim is not None:
            tv.duz(cv2.bitwise_and(tv.cizgi(tv.bos_maske(), [(ayrim, -0.64), (ayrim * 0.8, -0.50), (0.0, -0.38)], 0.012), m),
                   sac_golge * 0.6, opak=0.55, bulanik=0.006)

    def _sac_dokusu(self, tv, tohum, guc, dikey=False):
        rng = np.random.default_rng(tohum)
        d = np.zeros((tv.y, tv.g), np.float32)
        for _ in range(420):
            x, y = rng.uniform(0, tv.g), rng.uniform(0, tv.y)
            uz = rng.uniform(0.15, 0.5) * tv.S
            aci = math.pi / 2 + rng.normal(0, 0.18) if dikey else rng.uniform(0, math.pi)
            p1 = (int((x - math.cos(aci) * uz / 2) * 16), int((y - math.sin(aci) * uz / 2) * 16))
            p2 = (int((x + math.cos(aci) * uz / 2) * 16), int((y + math.sin(aci) * uz / 2) * 16))
            cv2.line(d, p1, p2, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA, 4)
        return cv2.GaussianBlur(d, (0, 0), 0.6) * guc

    def _kivrim_dokusu(self, tv, tohum=3):
        rng = np.random.default_rng(tohum)
        d = np.zeros((tv.y, tv.g), np.float32)
        for _ in range(9):
            x = rng.uniform(0.15, 0.85) * tv.g
            ust = tv.y * rng.uniform(0.60, 0.75)
            pts = np.array([[x + (yy - ust) * rng.uniform(-0.12, 0.12), yy] for yy in np.linspace(ust, tv.y, 8)])
            cv2.polylines(d, [np.int32(pts * 16)], False, -1.0, int(tv.S * 0.05), cv2.LINE_AA, 4)
        d = cv2.GaussianBlur(d, (0, 0), tv.S * 0.04)
        return d * 0.28

    # -- dinamik yüz öğeleri (küçük yüz tuvalinde)
    def _yuz_tuvali(self):
        return self._tuval(-0.46, 0.46, -0.30, 0.46)

    def _gozler(self, tv, d):
        T = self.T
        kapak = float(np.clip(d.get("kapak", 0.0), 0, 1))
        gx, gy = d.get("bakis", (0.0, 0.0))
        yon = d.get("donus", 0.0) * 0.03
        kadin = T["tip"] == "kadin"
        gw = 0.08 if kadin else 0.075
        tepe = 0.056 if kadin else 0.048
        alt_y = 0.030
        for s in (-1, 1):
            cx, cy = 0.145 * s + yon, -0.02
            ic_x, dis_x = cx - gw * s, cx + gw * s
            ust = catmull([(ic_x, cy + 0.002), (cx - gw * 0.55 * s, cy - tepe * 0.75), (cx + 0.01 * s, cy - tepe),
                           (cx + gw * 0.62 * s, cy - tepe * 0.72), (dis_x, cy - 0.010)], 5)
            alt = catmull([(dis_x, cy - 0.010), (cx + gw * 0.5 * s, cy + alt_y * 0.8), (cx, cy + alt_y),
                           (cx - gw * 0.55 * s, cy + alt_y * 0.75), (ic_x, cy + 0.002)], 5)
            alt_ters = alt[::-1]
            ust_k = ust.copy()
            hedef = np.interp(ust[:, 0] * s, alt_ters[:, 0] * s, alt_ters[:, 1]) - 0.003
            ust_k[:, 1] = ust[:, 1] + (hedef - ust[:, 1]) * kapak
            goz_m = tv.poli(tv.bos_maske(), np.vstack([ust_k, alt]))
            if kapak < 0.95:
                tv.boya(goz_m, T["goz_beyaz"], _p(T["goz_beyaz"]) * 0.70, isik=(0, -1, 0.5), sigma=0.03,
                        yumusaklik=(0.2, 0.9), sicak_gecis=False)
                ix, iy = cx + gx * 0.028, cy - 0.010 + gy * 0.012
                r = 0.036 if kadin else 0.034
                iris = cv2.bitwise_and(tv.elips(tv.bos_maske(), ix, iy, r, r), goz_m)
                goz_renk = _p(d.get("goz_renk", T["goz"]))
                tv.boya(iris, goz_renk, goz_renk * 0.30, isik=(0.2, 0.8, 0.6), sigma=0.02, yumusaklik=(0.2, 0.9),
                        isima=goz_renk * d.get("goz_isima", T.get("goz_isima", 0.0)), sicak_gecis=False)
                tv.duz(cv2.bitwise_and(tv.elips(tv.bos_maske(), ix, iy, r * 0.42, r * 0.42), goz_m), (0.02, 0.02, 0.03))
                halka = tv.cizgi(tv.bos_maske(), [(ix + r * math.cos(a), iy + r * math.sin(a))
                                                  for a in np.linspace(0, 2 * math.pi, 25)], 0.006)
                tv.duz(cv2.bitwise_and(halka, goz_m), goz_renk * 0.22, opak=0.85)
                tv.duz(cv2.bitwise_and(tv.elips(tv.bos_maske(), ix - r * 0.35, iy - r * 0.38, r * 0.26, r * 0.26), goz_m),
                       (1, 1, 1), opak=0.95)
                tv.duz(cv2.bitwise_and(tv.elips(tv.bos_maske(), ix + r * 0.35, iy + r * 0.3, r * 0.12, r * 0.12), goz_m),
                       (1, 1, 1), opak=0.7)
                tv.duz(cv2.bitwise_and(tv.cizgi(tv.bos_maske(), ust_k, 0.03), goz_m), (0.25, 0.12, 0.12), opak=0.35,
                       bulanik=0.008)
            kal = 0.017 if kadin else 0.015
            tv.duz(tv.cizgi(tv.bos_maske(), ust_k, kal), T["cizgi"])
            if kadin or T.get("kirpik", False):
                tv.duz(tv.cizgi(tv.bos_maske(), [ust_k[-1], (dis_x + 0.022 * s, cy - 0.032)], 0.010), T["cizgi"])
            tv.duz(tv.cizgi(tv.bos_maske(), alt[:int(len(alt) * 0.7)], 0.0045), T["cizgi"], opak=0.45)
            kir = ust_k.copy()
            kir[:, 1] -= 0.022 + 0.01 * (1 - kapak)
            tv.duz(tv.cizgi(tv.bos_maske(), kir[2:-2], 0.005), self.ten_golge * 0.7, opak=0.45, bulanik=0.004)

    def _kaslar(self, tv, d):
        T = self.T
        kalk, catik, uzgun = d.get("kas_kalk", 0.0), d.get("kas_catik", 0.0), d.get("uzgun", 0.0)
        yon = d.get("donus", 0.0) * 0.03
        kal = T["kas_kalin"]
        for s in (-1, 1):
            ic = (0.055 * s + yon + 0.012 * catik * s, -0.105 - 0.03 * kalk + 0.018 * catik - 0.03 * uzgun)
            tepe = (0.16 * s + yon, -0.14 - 0.035 * kalk + 0.005 * catik + 0.008 * uzgun)
            dis = (0.255 * s + yon, -0.105 - 0.02 * kalk + 0.014 * uzgun)
            ust = catmull([ic, tepe, dis], 6)
            alt = ust.copy()
            alt[:, 1] += np.linspace(kal, kal * 0.35, len(ust))
            m = tv.poli(tv.bos_maske(), np.vstack([ust, alt[::-1]]))
            tv.boya(m, T["kas_renk"], None, isik=T["isik"], sigma=0.01, sicak_gecis=False)

    def _agiz(self, tv, d):
        T = self.T
        acik = float(np.clip(d.get("agiz", 0.0), 0, 1))
        gul = d.get("gulus", T.get("gulus", 0.0))
        yon = d.get("donus", 0.0) * 0.025
        kadin = T["tip"] == "kadin"
        yw = (0.062 if kadin else 0.075) * (1 + 0.15 * max(0.0, gul))
        cy = 0.285
        kose = cy - 0.018 * gul
        sol, sag = (-yw + yon, kose), (yw + yon, kose)
        orta_ust = (yon, cy - 0.002 - 0.004 * gul)
        ust_dudak = catmull([sol, (-yw * 0.45 + yon, cy - 0.022), (-0.018 + yon, cy - 0.026), (yon, cy - 0.020),
                             (0.018 + yon, cy - 0.026), (yw * 0.45 + yon, cy - 0.022), sag], 4)
        acilma = acik * 0.06
        cizgi_ust = catmull([sol, (-yw * 0.5 + yon, cy + 0.002 - 0.008 * gul), orta_ust,
                             (yw * 0.5 + yon, cy + 0.002 - 0.008 * gul), sag], 5)
        cizgi_alt = cizgi_ust.copy()
        cizgi_alt[:, 1] += acilma * np.sin(np.linspace(0, math.pi, len(cizgi_alt)))
        alt_dudak = catmull([sag, (yw * 0.5 + yon, cy + 0.030 + acilma), (yon, cy + 0.038 + acilma),
                             (-yw * 0.5 + yon, cy + 0.030 + acilma), sol], 5)
        if acik > 0.05:
            m = tv.poli(tv.bos_maske(), np.vstack([cizgi_ust, cizgi_alt[::-1]]))
            tv.duz(m, (0.16, 0.04, 0.05))
            dis = tv.poli(tv.bos_maske(), np.vstack([cizgi_ust, (cizgi_ust + _p([0, acilma * 0.32]))[::-1]]))
            tv.duz(cv2.bitwise_and(dis, m), (0.94, 0.92, 0.88))
        dudak = _p(T["dudak"])
        tv.boya(tv.poli(tv.bos_maske(), np.vstack([ust_dudak, cizgi_ust[::-1]])), dudak * 0.85, dudak * 0.55,
                isik=T["isik"], sigma=0.01, sicak_gecis=False)
        tv.boya(tv.poli(tv.bos_maske(), np.vstack([cizgi_alt, alt_dudak])), dudak, dudak * 0.6, isik=T["isik"],
                sigma=0.015, parlama=0.25, sicak_gecis=False)
        tv.duz(tv.cizgi(tv.bos_maske(), cizgi_ust, 0.006), _p(T["cizgi"]) * 0.8, opak=0.8)
        if gul > 0.2:
            for s in (-1, 1):
                tv.duz(tv.cizgi(tv.bos_maske(), catmull([(s * (yw + 0.03) + yon, kose - 0.06), (s * (yw + 0.035) + yon, kose),
                                                          (s * (yw + 0.02) + yon, kose + 0.03)], 4), 0.006),
                       self.ten_golge * 0.6, opak=0.35 * gul, bulanik=0.004)
        tv.duz(tv.elips(tv.bos_maske(), yon, cy + 0.075 + acilma, 0.045, 0.012), self.ten_golge, opak=0.22,
               bulanik=0.012, carp=True)

    # -- kare
    def kare(self, t, d):
        """Portrenin bu karedeki RGBA (premultiplied) hâli; kanvas Portre.X0..X1, Y0..Y1."""
        bas = self._tuval(-0.8, 0.8, -1.25, 1.9)
        oy = int(round((-1.0 - (-1.25)) * self.S))
        bas.ekle(self.kafa.rgb, self.kafa.a, self.kafa.y0 + oy, self.kafa.x0)
        yx = int(round((-0.46 - (-0.8)) * self.S))
        yy = int(round((-0.30 - (-1.25)) * self.S))
        yz = self._yuz_tuvali()
        self._gozler(yz, d)
        self._kaslar(yz, d)
        if d.get("yas", 0) > 0:
            self._gozyasi(yz, t, d["yas"])
        k = yz.kirp()
        bas.ekle(k.rgb, k.a, k.y0 + yy, k.x0 + yx)
        ruzgar = d.get("ruzgar", 0.6)
        oy2 = int(round((-1.0 - (-1.25)) * self.S))
        sakal = self._dalgalandir(self.sakal, t, ruzgar * 0.5, 0.35, -1.0)
        bas.ekle(sakal.rgb, sakal.a, sakal.y0 + oy2, sakal.x0)
        yz = self._yuz_tuvali()
        self._agiz(yz, d)
        k = yz.kirp()
        bas.ekle(k.rgb, k.a, k.y0 + yy, k.x0 + yx)
        bas.ekle(self.biyik.rgb, self.biyik.a, self.biyik.y0 + oy2, self.biyik.x0)
        sac_on = self._dalgalandir(self.sac_on, t, ruzgar * 0.7, -0.05, -1.0)
        bas.ekle(sac_on.rgb, sac_on.a, sac_on.y0 + oy2, sac_on.x0)
        bas.ekle(self.tac.rgb, self.tac.a, self.tac.y0, self.tac.x0)
        if "ek_bas" in d:
            d["ek_bas"](self, bas, t)
        bas_k = bas.kirp()

        egim, nefes = d.get("egim", 0.0), d.get("nefes", 0.0)
        sonuc = self._tuval()
        pv = ((self.boyun_pivot[0] - self.X0) * self.S, (self.boyun_pivot[1] - self.Y0) * self.S)
        M = cv2.getRotationMatrix2D(pv, egim, 1.0)
        M[1, 2] += -nefes * 0.012 * self.S
        sac_arka = self._dalgalandir(self.sac_arka, t, ruzgar, 0.1, self.Y0)
        self._donusturup_ekle(sonuc, sac_arka, M)
        Mg = np.float32([[1, 0, 0], [0, 1, -nefes * 0.018 * self.S]])
        self._donusturup_ekle(sonuc, self.govde, Mg)
        if "ek_govde" in d:
            d["ek_govde"](self, sonuc, t)
        bas_ofx, bas_ofy = self._ofset(-0.8, -1.25)
        bas_k.x0 += bas_ofx
        bas_k.y0 += bas_ofy
        self._donusturup_ekle(sonuc, bas_k, M)
        if "ek_on" in d:
            d["ek_on"](self, sonuc, t)
        return np.dstack([sonuc.rgb, sonuc.a])

    def nokta(self, x, y, d):
        """Baş koordinatındaki (x, y) noktasının, bu karedeki (eğim ve nefes uygulanmış) tuval pikseli."""
        pv = ((self.boyun_pivot[0] - self.X0) * self.S, (self.boyun_pivot[1] - self.Y0) * self.S)
        M = cv2.getRotationMatrix2D(pv, d.get("egim", 0.0), 1.0)
        M[1, 2] += -d.get("nefes", 0.0) * 0.012 * self.S
        return M @ np.array([(x - self.X0) * self.S, (y - self.Y0) * self.S, 1.0])

    def _donusturup_ekle(self, hedef, k, M):
        """Kırpılmış katmanı hedef tuvale afin dönüşümle ekler (yalnızca ilgili bölgede)."""
        pay = 12 + int(abs(M[0, 1]) * 400)
        x0, y0 = max(0, k.x0 - pay), max(0, k.y0 - pay)
        x1, y1 = min(hedef.g, k.x0 + k.a.shape[1] + pay), min(hedef.y, k.y0 + k.a.shape[0] + pay)
        if x1 <= x0 or y1 <= y0:
            return
        M2 = M.copy().astype(np.float64)
        M2[0, 2] += M[0, 0] * k.x0 + M[0, 1] * k.y0 - x0
        M2[1, 2] += M[1, 0] * k.x0 + M[1, 1] * k.y0 - y0
        rgb = cv2.warpAffine(k.rgb, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
        a = cv2.warpAffine(k.a, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
        hedef.ekle(rgb, a, y0, x0)

    def _dalgalandir(self, k, t, guc, y_bas, y_tuval0):
        if guc <= 0.001 or k.a.size <= 1:
            return k
        hh, ww = k.a.shape
        yy = (np.arange(hh, dtype=np.float32) + k.y0) / self.S + y_tuval0
        agirlik = np.clip((yy - y_bas) / 1.2, 0, 1) ** 1.5
        dx = (np.sin(t * 1.4 + yy * 3.0) * 0.022 + np.sin(t * 2.3 + yy * 7.0) * 0.006) * self.S * guc * agirlik
        pay = int(np.abs(dx).max()) + 2
        rgb = cv2.copyMakeBorder(k.rgb, 0, 0, pay, pay, cv2.BORDER_CONSTANT, value=0)
        a = cv2.copyMakeBorder(k.a, 0, 0, pay, pay, cv2.BORDER_CONSTANT, value=0)
        ww2 = ww + 2 * pay
        harita_x = (np.arange(ww2, dtype=np.float32)[None, :] - dx[:, None]).astype(np.float32)
        harita_y = np.repeat(np.arange(hh, dtype=np.float32)[:, None], ww2, 1)
        return Katman(cv2.remap(rgb, harita_x, harita_y, cv2.INTER_LINEAR, borderValue=0),
                      cv2.remap(a, harita_x, harita_y, cv2.INTER_LINEAR, borderValue=0), k.x0 - pay, k.y0)

    def _gozyasi(self, tv, t, guc):
        for s, faz in ((-1, 0.0), (1, 1.7)):
            ilerle = ((t * 0.35 + faz) % 1.0)
            x = 0.16 * s + 0.01 * s * ilerle
            y = 0.02 + ilerle * 0.40
            m = tv.elips(tv.bos_maske(), x, y, 0.011, 0.017)
            tv.boya(m, (0.8, 0.9, 1.0), (0.4, 0.5, 0.7), isik=(-0.5, -0.7, 0.5), sigma=0.008, parlama=1.0,
                    opak=0.85 * guc, sicak_gecis=False)
            tv.duz(tv.cizgi(tv.bos_maske(), [(0.16 * s, 0.02), (x, y - 0.02)], 0.006), (0.85, 0.9, 1.0), opak=0.25 * guc)


VARSAYILAN = {
    "tip": "erkek", "ten": (0.93, 0.78, 0.66), "allik": (0.95, 0.45, 0.42), "dudak": (0.72, 0.40, 0.38),
    "goz": (0.25, 0.45, 0.80), "goz_beyaz": (0.96, 0.95, 0.93), "cizgi": (0.12, 0.07, 0.06),
    "sac": (0.35, 0.22, 0.12), "sac_boy": 0.6, "sac_gen": 1.0, "sac_on": "orta",
    "sakal": 0.0, "sakal_renk": (0.35, 0.22, 0.12),
    "kas_renk": (0.25, 0.15, 0.10), "kas_kalin": 0.028, "kulak": "insan",
    "kiyafet": (0.20, 0.30, 0.60), "kiyafet_golge": None, "pelerin": None, "yaka": "v", "sus": (0.9, 0.75, 0.4),
    "tas": (0.4, 0.6, 1.0), "omuz": 1.0, "tac": None, "kukuleta": None,
    "isik": (-0.55, -0.6, 0.58), "kenar": (0.4, 0.6, 1.0),
}


# ------------------------------------------------------------------ canlandırma ve yerleştirme

def bosta(t, tohum=0, kirpma_araligi=3.4, **ek):
    """Karakterin kendi başına yaşadığı hâl: göz kırpma, nefes, baş salınımı, bakış kaymaları."""
    rng = np.random.default_rng(tohum)
    araliklar = rng.uniform(0.55, 1.45, 120) * kirpma_araligi
    anlar = np.cumsum(araliklar) - rng.uniform(0.5, 2.0)
    ciftler = rng.random(120) < 0.18
    kapak = 0.0
    for an, cift in zip(anlar, ciftler):
        for k in ((0.0, 0.26) if cift else (0.0,)):
            dt = t - an - k
            if 0 <= dt < 0.2:
                kapak = max(kapak, math.sin(math.pi * min(1.0, dt / 0.2)) ** 0.7)
    fazlar = rng.uniform(0, 2 * math.pi, 6)
    nefes = math.sin(2 * math.pi * t / 4.2 + fazlar[0])
    egim = 1.3 * math.sin(0.42 * t + fazlar[1]) + 0.45 * math.sin(1.07 * t + fazlar[2])
    donus = 0.35 * math.sin(0.31 * t + fazlar[3])
    # bakış: 1.2–3 sn arayla hedef değiştirir, 0.12 sn'de sıçrar
    hedefler = rng.uniform(-0.55, 0.55, (60, 2)) * _p([1.0, 0.45])
    hedefler[::3] = 0
    sureler = np.cumsum(rng.uniform(1.2, 3.0, 60)) - 2.0
    i = int(np.searchsorted(sureler, t))
    i = min(max(i, 1), 59)
    f = puruz((t - sureler[i - 1]) / 0.12)
    bakis = tuple(hedefler[i - 1] * (1 - f) + hedefler[i] * f) if t >= sureler[i - 1] else tuple(hedefler[i - 1])
    d = dict(kapak=kapak, nefes=nefes, egim=egim, donus=donus, bakis=bakis, ruzgar=0.6)
    for k, v in ek.items():
        if k == "kapak":
            d["kapak"] = max(d["kapak"], v)
        else:
            d[k] = v
    return d


def kareye(portre, px, py, x, y, olcek=1.0):
    """Portre tuvalindeki piksel konumunu, yerlestir() ile bindirilmiş karedeki konuma çevirir."""
    return x + (px + portre.X0 * portre.S) * olcek, y + (py + portre.Y0 * portre.S) * olcek


def yerlestir(kare, rgba, portre, x, y, olcek=1.0, alfa=1.0, ton=(1.0, 1.0, 1.0), aydinlik=1.0,
              alt_karartma=None, alt_soldur=None, ekran=False):
    """Portre karesini (baş merkezi x, y pikselde olacak şekilde) kareye bindirir.

    ton: renk çarpanı (sahne ışığı), aydinlik: 0 siluet … 1 tam ışık,
    alt_karartma: (y_bas, y_bit) piksel aralığında gövdeyi karanlığa gömer,
    alt_soldur: (y_bas, y_bit) aralığında gövdeyi saydamlaştırır,
    ekran: True ise "ekran" karışımıyla ışıktan bir görüntü gibi bindirir (gökteki yüzler).
    """
    if alfa <= 0.003:
        return
    Hk, Wk = kare.shape[:2]
    ys, xs = np.nonzero(rgba[..., 3] > 0.004)
    if len(xs) == 0:
        return
    kx0, kx1, ky0, ky1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    parca = rgba[ky0:ky1, kx0:kx1]
    s = olcek
    if abs(s - 1) > 1e-3:
        parca = cv2.resize(parca, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    merkez_x = (-portre.X0 * portre.S - kx0) * s
    merkez_y = (-portre.Y0 * portre.S - ky0) * s
    x0, y0 = int(round(x - merkez_x)), int(round(y - merkez_y))
    hh, ww = parca.shape[:2]
    a0, a1 = max(0, y0), min(Hk, y0 + hh)
    b0, b1 = max(0, x0), min(Wk, x0 + ww)
    if a1 <= a0 or b1 <= b0:
        return
    p = parca[a0 - y0:a1 - y0, b0 - x0:b1 - x0]
    rgb = p[..., :3] * (_p(ton) * aydinlik)
    a = p[..., 3:4] * alfa
    if alt_karartma is not None:
        yy = np.arange(a0, a1, dtype=np.float32)
        k = 1 - puruz((yy - alt_karartma[0]) / max(1.0, alt_karartma[1] - alt_karartma[0]))
        rgb = rgb * k[:, None, None]
    if alt_soldur is not None:
        yy = np.arange(a0, a1, dtype=np.float32)
        k = 1 - puruz((yy - alt_soldur[0]) / max(1.0, alt_soldur[1] - alt_soldur[0]))
        rgb = rgb * k[:, None, None]
        a = a * k[:, None, None]
    bolge = kare[a0:a1, b0:b1].astype(np.float32)
    if ekran:
        kare[a0:a1, b0:b1] = np.clip(255 - (255 - bolge) * (1 - np.clip(rgb * alfa, 0, 1)), 0, 255).astype(np.uint8)
        return
    kare[a0:a1, b0:b1] = np.clip(bolge * (1 - a) + rgb * (255.0 * alfa), 0, 255).astype(np.uint8)
