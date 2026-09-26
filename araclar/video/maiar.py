"""Maiar'ın portre tanımları: Olórin, Gandalf (Gri ve Ak), Mairon (Sauron), Melian, Eönwë, Ossë, Uinen."""
import math

import numpy as np

import portre as P
from portre import _p, catmull
from valar import _bant, _metal, _tas


# ------------------------------------------------------------------ başlıklar

def sapka_gandalf(pr, tv):
    """Geniş kenarlı, ucu hafifçe yana düşmüş sivri gezgin şapkası."""
    T = pr.T
    renk = _p(T["sapka"])
    golge = renk * _p([0.42, 0.42, 0.52])
    koni = catmull([(-0.34, -0.40), (-0.27, -0.62), (-0.16, -0.86), (-0.04, -1.02), (0.10, -1.13), (0.21, -1.19),
                    (0.16, -1.08), (0.10, -0.96), (0.17, -0.80), (0.26, -0.60), (0.34, -0.40)], 5)
    m = tv.poli(tv.bos_maske(), koni)
    tv.boya(m, renk, golge, isik=T["isik"], sigma=0.07, kenar=T["kenar"], yumusaklik=(0.28, 0.72),
            sekil="silindir", sekil_agirlik=0.55, doku=pr._kivrim_dokusu(tv, 17), hat=0.6)
    # kıvrım gölgeleri
    for pts in ([(-0.16, -0.50), (-0.10, -0.75), (-0.02, -0.95)], [(0.12, -0.52), (0.14, -0.72), (0.10, -0.92)]):
        tv.duz(tv.cizgi(tv.bos_maske(), catmull(pts, 5), 0.018), golge * 0.7, opak=0.35, bulanik=0.012, carp=True)
    bant = _bant(tv, -0.425, 0.06, 0.345, 0.03)
    m = tv.poli(tv.bos_maske(), bant)
    koyu = renk * 0.55
    tv.boya(m, koyu, koyu * 0.5, isik=T["isik"], sigma=0.02, hat=0.5)
    # kenar (önden görülen elips; yanlarda hafifçe aşağı sarkar)
    ust = [(-0.74, -0.33), (-0.55, -0.44), (-0.30, -0.50), (0.0, -0.52), (0.30, -0.50), (0.55, -0.44), (0.74, -0.33)]
    alt = [(0.74, -0.33), (0.60, -0.30), (0.34, -0.33), (0.0, -0.345), (-0.34, -0.33), (-0.60, -0.30), (-0.74, -0.33)]
    m = tv.poli(tv.bos_maske(), np.vstack([catmull(ust, 6), catmull(alt, 6)]))
    tv.boya(m, renk * 1.05, golge, isik=T["isik"], sigma=0.03, kenar=T["kenar"], yumusaklik=(0.3, 0.7),
            sekil="elipsoid", sekil_agirlik=0.4, hat=0.7)


def tac_mairon(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.36, 0.035, 0.38, 0.06))
    _metal(tv, m, (0.85, 0.60, 0.30), T["isik"], T["kenar"], 0.8)
    _tas(tv, 0.0, -0.39, 0.026, T.get("tac_tas", (1.0, 0.35, 0.10)), T["isik"], 0.5)


def tac_melian(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.345, 0.02, 0.37, 0.07))
    for s in (-1, 1):
        for i in range(4):
            x = s * (0.08 + i * 0.075)
            y = -0.39 - 0.055 * (1 - abs(x) / 0.37) + 0.01
            tv.elips(m, x, y, 0.028, 0.012, aci=-s * (20 + i * 12))
    _metal(tv, m, (0.88, 0.90, 0.96), T["isik"], T["kenar"], 0.9)
    _tas(tv, 0.0, -0.405, 0.03, (0.92, 0.94, 1.0), T["isik"], 0.45)


def tac_eonwe(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.37, 0.04, 0.38, 0.06))
    for s in (-1, 1):
        kanat = [(0.30 * s, -0.36)]
        for i in range(5):
            a = math.radians(20 + i * 16)
            r = 0.20 - i * 0.022
            kanat.append((s * (0.32 + r * math.cos(a)), -0.37 - r * math.sin(a)))
            kanat.append((s * (0.33 + (r - 0.06) * math.cos(a + 0.12)), -0.37 - (r - 0.06) * math.sin(a + 0.12)))
        kanat.append((0.36 * s, -0.33))
        tv.poli(m, kanat)
    _metal(tv, m, (0.90, 0.92, 0.98), T["isik"], T["kenar"], 0.9)
    _tas(tv, 0.0, -0.40, 0.03, (0.35, 0.60, 1.0), T["isik"], 0.35)


def tac_osse(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.42, 0.05, 0.38, 0.07))
    rng = np.random.default_rng(9)
    for i, x in enumerate(np.linspace(-0.30, 0.30, 7)):
        taban = -0.45 + abs(x) * 0.15
        boy = rng.uniform(0.10, 0.22)
        yon = rng.choice([-1, 1])
        tv.poli(m, catmull([(x - 0.035, taban + 0.02), (x - 0.03, taban - boy * 0.5), (x + 0.02 * yon, taban - boy),
                            (x + 0.06 * yon, taban - boy * 0.85), (x + 0.025, taban - boy * 0.4),
                            (x + 0.035, taban + 0.02)], 3))
    tv.boya(m, (0.80, 0.95, 0.95), (0.18, 0.40, 0.45), isik=T["isik"], sigma=0.03, parlama=0.5, kenar=T["kenar"],
            hat=0.5)
    _tas(tv, 0.0, -0.42, 0.028, (0.2, 0.8, 0.85), T["isik"], 0.35)


def tac_uinen(pr, tv):
    T = pr.T
    m = tv.poli(tv.bos_maske(), _bant(tv, -0.34, 0.016, 0.37, 0.07))
    _metal(tv, m, (0.85, 0.92, 0.95), T["isik"], T["kenar"], 0.7)
    for i in range(11):
        x = -0.30 + i * 0.06
        y = -0.35 - 0.05 * (1 - abs(x) / 0.37) + 0.018
        _tas(tv, x, y, 0.017 if i != 5 else 0.028, (0.95, 0.97, 1.0), T["isik"], 0.25)


# ------------------------------------------------------------------ tanımlar

MAIAR = {
    "olorin": dict(tip="erkek", ten=(0.93, 0.82, 0.74), sac=(0.72, 0.72, 0.76), sac_boy=0.55, sac_gen=0.95,
                   sac_on="yan", kas_renk=(0.45, 0.45, 0.50), kas_kalin=0.024, goz=(0.45, 0.55, 0.68),
                   goz_isima=0.12, dudak=(0.70, 0.45, 0.42), kiyafet=(0.46, 0.47, 0.52), pelerin=(0.62, 0.63, 0.68),
                   yaka="v", sus=(0.85, 0.87, 0.92), tas=(0.9, 0.92, 1.0), kenar=(0.75, 0.80, 0.95), omuz=0.95,
                   gulus=0.2),
    "gandalf": dict(tip="erkek", ten=(0.90, 0.76, 0.66), sac=(0.80, 0.80, 0.82), sac_boy=0.95, sac_gen=1.05,
                    sac_on="yok", sakal=1.0, sakal_renk=(0.88, 0.88, 0.90), kas_renk=(0.70, 0.70, 0.72),
                    kas_kalin=0.045, goz=(0.40, 0.52, 0.66), goz_isima=0.1, dudak=(0.62, 0.42, 0.40),
                    kiyafet=(0.40, 0.41, 0.46), pelerin=(0.52, 0.53, 0.58), yaka="v", sus=(0.80, 0.82, 0.88),
                    tas=(0.95, 0.35, 0.15), sapka=(0.42, 0.46, 0.60), tac=sapka_gandalf, tac_golge=-0.29,
                    kenar=(0.75, 0.80, 0.95), omuz=1.02),
    "gandalf_ak": dict(tip="erkek", ten=(0.93, 0.82, 0.74), sac=(0.96, 0.96, 0.98), sac_boy=0.95, sac_gen=1.05,
                       sac_on="orta", sakal=0.95, sakal_renk=(0.97, 0.97, 0.99), kas_renk=(0.82, 0.82, 0.86),
                       kas_kalin=0.042, goz=(0.40, 0.58, 0.85), goz_isima=0.3, dudak=(0.66, 0.44, 0.42),
                       kiyafet=(0.93, 0.94, 0.97), pelerin=(0.98, 0.98, 1.0), yaka="v", sus=(0.90, 0.92, 0.98),
                       tas=(0.95, 0.35, 0.15), kenar=(0.85, 0.90, 1.0), omuz=1.02),
    "mairon": dict(tip="erkek", ten=(0.94, 0.80, 0.68), sac=(0.80, 0.46, 0.18), sac_boy=0.65, sac_gen=0.98,
                   sac_on="yan", kas_renk=(0.45, 0.22, 0.08), kas_kalin=0.026, goz=(0.95, 0.62, 0.18),
                   goz_isima=0.25, dudak=(0.72, 0.42, 0.38), kiyafet=(0.42, 0.14, 0.08), pelerin=(0.18, 0.07, 0.05),
                   yaka="kare", sus=(0.85, 0.60, 0.30), tas=(1.0, 0.35, 0.10), tac=tac_mairon,
                   isik=(0.6, -0.45, 0.6), kenar=(1.0, 0.55, 0.20), omuz=1.0, gulus=0.25),
    "melian": dict(tip="kadin", ten=(0.96, 0.88, 0.86), sac=(0.08, 0.07, 0.12), sac_boy=1.45, sac_gen=1.0,
                   sac_on="orta", kas_renk=(0.10, 0.08, 0.14), kas_kalin=0.017, goz=(0.55, 0.70, 0.95),
                   goz_isima=0.3, dudak=(0.76, 0.44, 0.50), kiyafet=(0.90, 0.92, 0.98), pelerin=(0.20, 0.16, 0.40),
                   yaka="kare", sus=(0.90, 0.92, 1.0), tas=(0.92, 0.94, 1.0), tac=tac_melian,
                   kenar=(0.70, 0.75, 1.0), omuz=0.9, gulus=0.3),
    "eonwe": dict(tip="erkek", ten=(0.95, 0.84, 0.76), sac=(0.95, 0.82, 0.45), sac_boy=0.6, sac_gen=0.95,
                  sac_on="yan", kas_renk=(0.70, 0.55, 0.25), kas_kalin=0.024, goz=(0.25, 0.50, 0.95),
                  goz_isima=0.2, dudak=(0.72, 0.44, 0.40), kiyafet=(0.80, 0.84, 0.92), pelerin=(0.10, 0.22, 0.58),
                  yaka="kare", sus=(0.90, 0.92, 0.98), tas=(0.35, 0.6, 1.0), tac=tac_eonwe,
                  kenar=(0.50, 0.70, 1.0), omuz=1.0),
    "osse": dict(tip="erkek", ten=(0.82, 0.80, 0.76), sac=(0.22, 0.48, 0.55), sac_boy=0.9, sac_gen=1.2, sac_on="yan",
                 sakal=0.30, sakal_renk=(0.25, 0.50, 0.56), kas_renk=(0.12, 0.25, 0.28), kas_kalin=0.036,
                 goz=(0.15, 0.85, 0.80), goz_isima=0.35, dudak=(0.60, 0.42, 0.40), kiyafet=(0.08, 0.28, 0.32),
                 pelerin=(0.04, 0.14, 0.20), yaka="v", sus=(0.80, 0.92, 0.92), tas=(0.2, 0.8, 0.85), tac=tac_osse,
                 tac_golge=-0.34, kenar=(0.30, 0.90, 0.85), omuz=1.1),
    "uinen": dict(tip="kadin", ten=(0.93, 0.88, 0.86), sac=(0.50, 0.72, 0.78), sac_boy=1.5, sac_gen=1.08,
                  sac_on="orta", kas_renk=(0.25, 0.38, 0.42), kas_kalin=0.016, goz=(0.30, 0.75, 0.85),
                  goz_isima=0.25, dudak=(0.74, 0.46, 0.50), kiyafet=(0.20, 0.45, 0.62), pelerin=(0.70, 0.85, 0.90),
                  yaka="kare", sus=(0.88, 0.94, 0.96), tas=(0.95, 0.97, 1.0), tac=tac_uinen,
                  kenar=(0.50, 0.90, 0.95), omuz=0.9, gulus=0.35),
}


def olustur(ad, S=400):
    return P.Portre(MAIAR[ad], S=S)
