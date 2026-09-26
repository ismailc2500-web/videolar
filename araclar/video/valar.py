"""Valar'ın portre tanımları: görünüm, renkler, taçlar ve aksesuarlar."""
import math

import numpy as np

import portre as P
from portre import _p, catmull


def _metal(tv, m, renk, isik, kenar, parlama=0.9):
    renk = _p(renk)
    tv.boya(m, renk, renk * _p([0.35, 0.35, 0.45]), isik=isik, sigma=0.025, parlama=parlama, kenar=kenar, hat=0.5)


def _tas(tv, x, y, r, renk, isik, isima=0.3):
    renk = _p(renk)
    m = tv.elips(tv.bos_maske(), x, y, r, r)
    tv.boya(m, renk, renk * 0.25, isik=isik, sigma=0.01, parlama=1.3, isima=renk * isima, hat=0.4)


def _bant(tv, y_orta, kalin, gen=0.37, yay=0.06):
    ust = catmull([(-gen, y_orta - kalin / 2 + yay * 0.3), (-gen * 0.5, y_orta - kalin / 2 - yay * 0.7),
                   (0, y_orta - kalin / 2 - yay), (gen * 0.5, y_orta - kalin / 2 - yay * 0.7),
                   (gen, y_orta - kalin / 2 + yay * 0.3)], 6)
    alt = ust[::-1].copy()
    alt[:, 1] += kalin
    return np.vstack([ust, alt])


def yildiz(x, y, r_dis, r_ic, uc=8, aci=0.0):
    pts = []
    for i in range(uc * 2):
        r = r_dis if i % 2 == 0 else r_ic
        a = aci + i * math.pi / uc - math.pi / 2
        pts.append((x + r * math.cos(a), y + r * math.sin(a)))
    return pts


# ------------------------------------------------------------------ taçlar

def tac_manwe(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.40, 0.085, 0.38, 0.07))
    for x, yuk, gen in ((0.0, 0.42, 0.07), (-0.14, 0.31, 0.055), (0.14, 0.31, 0.055), (-0.28, 0.20, 0.045),
                        (0.28, 0.20, 0.045)):
        taban = -0.47 + abs(x) * 0.18
        tv.poli(m, catmull([(x - gen, taban + 0.02), (x - gen * 0.5, taban - yuk * 0.55), (x, taban - yuk),
                            (x + gen * 0.5, taban - yuk * 0.55), (x + gen, taban + 0.02)], 4))
    _metal(tv, m, (0.88, 0.90, 0.96), T["isik"], T["kenar"])
    for x, y, r in ((0.0, -0.41, 0.036), (0.0, -0.86, 0.022), (-0.14, -0.74, 0.016), (0.14, -0.74, 0.016)):
        _tas(tv, x, y, r, (0.35, 0.60, 1.0), T["isik"])


def tac_varda(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.335, 0.022, 0.37, 0.07))
    _metal(tv, m, (0.90, 0.92, 0.98), T["isik"], T["kenar"])
    m = tv.poli(tv.bos_maske(), yildiz(0.0, -0.415, 0.085, 0.028, 8))
    tv.boya(m, (1.0, 1.0, 1.0), (0.75, 0.80, 0.95), isik=T["isik"], sigma=0.01, parlama=1.0, isima=_p([0.4, 0.45, 0.6]))
    for x in (-0.24, -0.12, 0.12, 0.24):
        y = -0.345 - 0.05 * (1 - abs(x) / 0.37) + 0.01
        m = tv.poli(tv.bos_maske(), yildiz(x, y, 0.028, 0.009, 4))
        tv.boya(m, (0.95, 0.97, 1.0), (0.7, 0.75, 0.9), isik=T["isik"], sigma=0.005, isima=_p([0.3, 0.35, 0.5]))


def tac_ulmo(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.42, 0.06, 0.38, 0.07))
    for i, x in enumerate((-0.26, -0.13, 0.0, 0.13, 0.26)):
        taban = -0.46 + abs(x) * 0.15
        boy = 0.20 if i == 2 else (0.15 if i in (1, 3) else 0.11)
        yon = 1 if x >= 0 else -1
        tv.poli(m, catmull([(x - 0.05, taban + 0.02), (x - 0.055, taban - boy * 0.6), (x + 0.01 * yon, taban - boy),
                            (x + 0.07 * yon, taban - boy * 0.8), (x + 0.03 * yon, taban - boy * 0.72),
                            (x + 0.03, taban - boy * 0.35), (x + 0.05, taban + 0.02)], 4))
    tv.boya(m, (0.85, 1.0, 0.98), (0.25, 0.55, 0.60), isik=T["isik"], sigma=0.03, parlama=0.6, kenar=T["kenar"], hat=0.5)
    _tas(tv, 0.0, -0.42, 0.03, (0.3, 0.95, 0.9), T["isik"], 0.35)


def tac_aule(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.36, 0.05, 0.38, 0.06))
    _metal(tv, m, (0.80, 0.52, 0.28), T["isik"], T["kenar"], 0.7)
    _tas(tv, 0.0, -0.39, 0.03, (1.0, 0.45, 0.10), T["isik"], 0.5)


def tac_orome(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.37, 0.035, 0.38, 0.06))
    for x in (-0.20, -0.10, 0.10, 0.20):
        y = -0.40 - 0.05 * (1 - abs(x) / 0.38)
        tv.elips(m, x, y - 0.01, 0.022, 0.045, aci=25 * (1 if x > 0 else -1))
    _metal(tv, m, (0.95, 0.78, 0.35), T["isik"], T["kenar"], 0.8)
    _tas(tv, 0.0, -0.435, 0.028, (0.3, 0.85, 0.35), T["isik"], 0.3)


def tac_yavanna(pr, tv):
    T = pr.T
    rng = np.random.default_rng(4)
    m = tv.bos_maske()
    for i in range(26):
        a = math.pi * (0.08 + 0.84 * i / 25)
        x, y = -0.40 * math.cos(a), -0.12 - 0.44 * math.sin(a)
        tv.elips(m, x, y, 0.055, 0.022, aci=math.degrees(a) + rng.uniform(-30, 30))
    tv.boya(m, (0.35, 0.68, 0.25), (0.08, 0.28, 0.08), isik=T["isik"], sigma=0.02, kenar=T["kenar"], hat=0.5)
    renkler = [(1.0, 0.95, 0.9), (1.0, 0.62, 0.72), (1.0, 0.85, 0.35), (0.85, 0.7, 1.0)]
    for i in range(9):
        a = math.pi * (0.12 + 0.76 * i / 8)
        x, y = -0.41 * math.cos(a), -0.12 - 0.45 * math.sin(a)
        c = renkler[i % 4]
        m = tv.bos_maske()
        for k in range(5):
            b = k * 2 * math.pi / 5 + i
            tv.elips(m, x + 0.022 * math.cos(b), y + 0.022 * math.sin(b), 0.02, 0.013, aci=math.degrees(b))
        tv.boya(m, c, _p(c) * 0.5, isik=T["isik"], sigma=0.008, hat=0.3)
        _tas(tv, x, y, 0.011, (1.0, 0.85, 0.3), T["isik"], 0.2)


# ------------------------------------------------------------------ tanımlar

VALAR = {
    "manwe": dict(tip="erkek", ten=(0.94, 0.80, 0.70), sac=(0.86, 0.88, 0.93), sac_boy=0.95, sac_gen=0.88,
                  sac_on="orta", sakal=0.30, sakal_renk=(0.86, 0.88, 0.93), kas_renk=(0.62, 0.64, 0.70),
                  kas_kalin=0.026, goz=(0.20, 0.48, 0.95), goz_isima=0.15, kiyafet=(0.10, 0.22, 0.58),
                  pelerin=(0.82, 0.86, 0.94), sus=(0.85, 0.88, 0.95), tas=(0.35, 0.6, 1.0), tac=tac_manwe,
                  tac_golge=-0.30, kenar=(0.45, 0.70, 1.0)),
    "varda": dict(tip="kadin", ten=(0.97, 0.87, 0.82), sac=(0.12, 0.12, 0.24), sac_boy=1.35, sac_gen=0.95,
                  sac_on="orta", kas_renk=(0.12, 0.10, 0.18), kas_kalin=0.018, goz=(0.55, 0.72, 1.0), goz_isima=0.35,
                  dudak=(0.78, 0.45, 0.50), kiyafet=(0.88, 0.90, 0.98), pelerin=(0.10, 0.14, 0.38),
                  sus=(0.90, 0.92, 1.0), tas=(0.9, 0.95, 1.0), tac=tac_varda, kenar=(0.6, 0.75, 1.0), yaka="kare",
                  omuz=0.9),
    "ulmo": dict(tip="erkek", ten=(0.80, 0.85, 0.83), sac=(0.42, 0.66, 0.64), sac_boy=1.25, sac_gen=1.0, sac_on="orta",
                 sakal=0.75, sakal_renk=(0.45, 0.70, 0.68), kas_renk=(0.25, 0.42, 0.42), kas_kalin=0.028,
                 goz=(0.15, 0.90, 0.85), goz_isima=0.45, kiyafet=(0.12, 0.42, 0.44), pelerin=(0.04, 0.18, 0.30),
                 sus=(0.85, 0.95, 0.95), tas=(0.3, 0.95, 0.9), tac=tac_ulmo, tac_golge=-0.33, kenar=(0.30, 0.95, 0.85),
                 dudak=(0.62, 0.45, 0.45), omuz=1.08),
    "aule": dict(tip="erkek", ten=(0.86, 0.64, 0.50), sac=(0.34, 0.19, 0.09), sac_boy=0.35, sac_on="kisa", sakal=0.40,
                 sakal_renk=(0.40, 0.22, 0.10), kas_renk=(0.25, 0.13, 0.06), kas_kalin=0.034, goz=(0.85, 0.55, 0.15),
                 goz_isima=0.1, kiyafet=(0.38, 0.23, 0.13), yaka="kare", sus=(0.75, 0.45, 0.2), tas=(1.0, 0.5, 0.1),
                 tac=tac_aule, tac_golge=-0.32, isik=(0.6, -0.45, 0.6), kenar=(1.0, 0.55, 0.20), omuz=1.12),
    "yavanna": dict(tip="kadin", ten=(0.92, 0.74, 0.60), sac=(0.60, 0.33, 0.13), sac_boy=1.35, sac_gen=1.0,
                    sac_on="yan", kas_renk=(0.35, 0.18, 0.08), kas_kalin=0.018, goz=(0.30, 0.72, 0.30),
                    goz_isima=0.15, dudak=(0.80, 0.42, 0.40), kiyafet=(0.20, 0.50, 0.22), pelerin=(0.55, 0.52, 0.18),
                    sus=(0.95, 0.80, 0.35), tas=(0.3, 0.85, 0.35), tac=tac_yavanna, kenar=(0.75, 0.95, 0.40), omuz=0.9),
    "mandos": dict(tip="erkek", ten=(0.80, 0.76, 0.75), sac=(0.14, 0.14, 0.18), sac_boy=0.9, sac_on="orta",
                   sakal=0.12, sakal_renk=(0.15, 0.15, 0.18), kas_renk=(0.10, 0.10, 0.12), kas_kalin=0.03,
                   goz=(0.55, 0.62, 0.80), goz_isima=0.35, dudak=(0.55, 0.40, 0.40), kiyafet=(0.08, 0.08, 0.11),
                   pelerin=(0.16, 0.16, 0.20), sus=(0.75, 0.78, 0.85), tas=(0.55, 0.65, 0.9), kukuleta=(0.20, 0.20, 0.25),
                   kenar=(0.45, 0.55, 0.85), isik=(-0.3, -0.7, 0.55)),
    "nienna": dict(tip="kadin", ten=(0.90, 0.83, 0.82), sac=(0.62, 0.63, 0.68), sac_boy=1.1, sac_on="orta",
                   kas_renk=(0.40, 0.40, 0.45), kas_kalin=0.016, goz=(0.55, 0.62, 0.72), goz_isima=0.1,
                   dudak=(0.70, 0.48, 0.50), kiyafet=(0.42, 0.44, 0.50), sus=(0.85, 0.87, 0.92), tas=(0.75, 0.85, 1.0),
                   kukuleta=(0.62, 0.64, 0.70), kenar=(0.75, 0.80, 0.95), omuz=0.88),
    "orome": dict(tip="erkek", ten=(0.85, 0.66, 0.52), sac=(0.20, 0.13, 0.07), sac_boy=0.75, sac_on="yan", sakal=0.12,
                  sakal_renk=(0.22, 0.14, 0.08), kas_renk=(0.15, 0.09, 0.05), kas_kalin=0.032, goz=(0.70, 0.52, 0.20),
                  goz_isima=0.12, kiyafet=(0.14, 0.34, 0.17), pelerin=(0.45, 0.30, 0.12), sus=(0.95, 0.78, 0.35),
                  tas=(0.3, 0.85, 0.35), tac=tac_orome, kenar=(0.70, 0.95, 0.45), omuz=1.05),
    "tulkas": dict(tip="erkek", ten=(0.93, 0.70, 0.56), sac=(0.96, 0.76, 0.30), sac_boy=0.85, sac_gen=1.12,
                   sac_on="yan", sakal=0.42, sakal_renk=(0.94, 0.72, 0.28), kas_renk=(0.75, 0.50, 0.15),
                   kas_kalin=0.034, goz=(0.30, 0.58, 0.92), goz_isima=0.1, gulus=0.6, kiyafet=(0.55, 0.20, 0.12),
                   yaka="kare", sus=(0.85, 0.60, 0.25), tas=(1.0, 0.7, 0.2), kenar=(1.0, 0.80, 0.35), omuz=1.2),
}


def olustur(ad, S=380):
    return P.Portre(VALAR[ad], S=S)
