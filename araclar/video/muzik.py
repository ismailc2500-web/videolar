#!/usr/bin/env python3
"""Bölüm müziğini ve ses efektlerini sentezler (telifsiz, tamamen koddan üretilir).

Müzik, seslendirme.py'nin ürettiği zaman.json'daki cümle zamanlarına göre
yerleştirilir; böylece akorlar ve efektler anlatımla eşzamanlı olur.

Kullanım:
  python3 muzik.py bolumler/bolum01.json --zaman build/zaman.json --cikti build/muzik.wav
"""
import argparse
import json
import pathlib

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
RNG = np.random.default_rng(7)

NOTA = {"C": -9, "C#": -8, "Db": -8, "D": -7, "D#": -6, "Eb": -6, "E": -5, "F": -4,
        "F#": -3, "Gb": -3, "G": -2, "G#": -1, "Ab": -1, "A": 0, "A#": 1, "Bb": 1, "B": 2}


def hz(ad):
    """'D4', 'F#3' gibi nota adını frekansa çevirir."""
    isim, oktav = ad[:-1], int(ad[-1])
    return 440.0 * 2 ** ((NOTA[isim] + 12 * (oktav - 4)) / 12)


class Parca:
    def __init__(self, sure):
        self.n = int(round(sure * SR))
        self.sol = np.zeros(self.n, np.float32)
        self.sag = np.zeros(self.n, np.float32)

    def ekle(self, t0, x, pan=0.0, kazanc=1.0):
        i = int(round(t0 * SR))
        if i >= self.n or len(x) == 0:
            return
        x = x[: self.n - i] * kazanc
        a = np.cos((pan + 1) * np.pi / 4)
        b = np.sin((pan + 1) * np.pi / 4)
        self.sol[i:i + len(x)] += x * a
        self.sag[i:i + len(x)] += x * b


def zarf(n, atak, birakma, egri=1.0):
    e = np.ones(n, np.float32)
    ia, ib = min(n, int(atak * SR)), min(n, int(birakma * SR))
    if ia:
        e[:ia] = np.linspace(0, 1, ia) ** egri
    if ib:
        e[n - ib:] *= np.linspace(1, 0, ib) ** egri
    return e


def toplamsal(f, sure, genlikler, sapma_cent=(0,), vibrato=0.0, vib_hiz=5.0):
    """Harmonik genlik listesinden, hafif akortsuz katmanlarla ses üretir."""
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = np.zeros(n, np.float32)
    for c in sapma_cent:
        f0 = f * 2 ** (c / 1200)
        faz_mod = 0.0
        if vibrato:
            faz_mod = (vibrato * f0 / vib_hiz) * np.sin(2 * np.pi * vib_hiz * t + RNG.uniform(0, 6.28))
        for k, g in enumerate(genlikler, 1):
            if g < 1e-4 or k * f0 > 16000:
                continue
            x += g * np.sin(2 * np.pi * k * f0 * t + k * faz_mod + RNG.uniform(0, 6.28))
    return x / max(1, len(sapma_cent))


def ped(f, sure, parlaklik=1800.0):
    h = [1 / k / (1 + (k * f / parlaklik) ** 2) for k in range(1, 24)]
    x = toplamsal(f, sure, h, sapma_cent=(-7, 0, 6))
    return x * zarf(len(x), min(2.5, sure / 3), min(3.0, sure / 3))


def koro(f, sure, atak=1.5, birakma=2.5):
    formant = [(800, 110), (1150, 120), (2900, 200)]
    h = []
    for k in range(1, 30):
        fk = k * f
        g = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2) * w for (fc, bw), w in zip(formant, (1.0, 0.6, 0.25)))
        h.append((0.15 + g) / k ** 0.6)
    x = toplamsal(f, sure, h, sapma_cent=(-9, -3, 4, 10), vibrato=0.004, vib_hiz=5.3)
    return x * zarf(len(x), min(atak, sure / 2), min(birakma, sure / 2))


def can(f, sure=4.0):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = np.zeros(n, np.float32)
    for oran, g, tau in [(1, 1.0, 1.6), (2.0, 0.45, 1.0), (2.76, 0.35, 0.7), (5.4, 0.18, 0.35), (8.9, 0.08, 0.2)]:
        x += g * np.exp(-t / tau) * np.sin(2 * np.pi * f * oran * t)
    return x * zarf(n, 0.004, 0.05)


def saf(f, sure, atak=0.8, birakma=1.5):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = np.sin(2 * np.pi * f * t) + 0.15 * np.sin(4 * np.pi * f * t)
    return x * zarf(n, atak, birakma)


def sert(f, sure, atak=0.05, birakma=0.3, tremolo=7.0):
    h = [1 / k for k in range(1, 40)]
    x = toplamsal(f, sure, h, sapma_cent=(-18, 0, 15))
    t = np.arange(len(x), dtype=np.float32) / SR
    x = np.tanh(2.5 * x) * (0.75 + 0.25 * np.sin(2 * np.pi * tremolo * t))
    return x * zarf(len(x), atak, birakma)


def gurultu(sure):
    return RNG.standard_normal(int(sure * SR)).astype(np.float32)


def suzgec(x, kesim, tur="low", derece=2):
    sos = butter(derece, kesim, btype=tur, fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def gumbur(sure=4.0, f0=95.0, f1=32.0):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.25)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 1.1)
    x += 0.5 * suzgec(gurultu(sure), 300) * np.exp(-t / 0.35)
    return x


def yukselen(sure):
    n = int(sure * SR)
    x = gurultu(sure)
    parcalar = np.array_split(x, 24)
    cikti = [suzgec(p, 400 + 7000 * (i / 23) ** 2) for i, p in enumerate(parcalar)]
    return np.concatenate(cikti) * np.linspace(0, 1, n) ** 2.5


def nabiz(sure=0.7):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    return np.sin(2 * np.pi * 52 * t) * np.exp(-t / 0.12)


def reverb(sol, sag, sure=3.4, islak=0.32):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ciktilar = []
    for kanal in (sol, sag):
        ir = RNG.standard_normal(n).astype(np.float32) * np.exp(-t / 0.85)
        ir = suzgec(ir, 5000)
        ir[: int(0.02 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        islak_k = fftconvolve(kanal, ir)[: len(kanal)].astype(np.float32)
        ciktilar.append((1 - islak) * kanal + islak * islak_k * 1.4)
    return ciktilar


def akor(p, t0, sure, notalar, tur=ped, kazanc=0.1, yayilim=0.6):
    for i, ad in enumerate(notalar):
        pan = yayilim * (i / max(1, len(notalar) - 1) * 2 - 1) if len(notalar) > 1 else 0.0
        p.ekle(t0, tur(hz(ad), sure), pan=pan, kazanc=kazanc)


# ---------------------------------------------------------------- Bölüm 1

def bolum01(p, z):
    birim = z["birimler"]

    def c(sahne, cumle=0):
        return next(b["bas"] for b in birim if b["sahne"] == sahne and b["cumle"] == cumle)

    sahne_bit = [s["bit"] for s in z["sahneler"]]
    akor_ani = sahne_bit[5] + 0.35          # "tek bir akor"
    son = z["sure"]

    # 0 · Kanca: tek bir saf nota, alttan drone, sonunda uğursuz bir ses
    p.ekle(0.0, saf(hz("D5"), 9.0, atak=0.6, birakma=4.0), kazanc=0.16)
    p.ekle(0.0, saf(hz("A5"), 7.0, atak=2.5, birakma=3.0), pan=0.3, kazanc=0.05)
    p.ekle(0.2, ped(hz("D2"), 21.0, 600), kazanc=0.20)
    p.ekle(c(0, 1), sert(hz("Eb2"), 3.8, atak=1.2, birakma=1.6, tremolo=5), kazanc=0.05)

    # 1 · Eru: sıcak, yavaş açılan majör ped ve koro
    t = c(1, 0) - 0.6
    akor(p, t, 10.5, ["D3", "A3", "D4", "F#4"], ped, 0.07)
    akor(p, c(1, 2) - 0.8, 8.0, ["A4", "D5"], koro, 0.05)
    p.ekle(c(1, 2) + 0.9, can(hz("D6")), pan=-0.2, kazanc=0.06)
    p.ekle(c(1, 3) + 1.2, can(hz("A5")), pan=0.3, kazanc=0.05)

    # 2 · Ainur: sesler tek tek, sonra küçük gruplar hâlinde
    t = c(2, 0) - 0.5
    akor(p, t, 8.0, ["B2", "F#3", "B3", "D4"], ped, 0.05)
    akor(p, t + 7.6, 8.4, ["G2", "D3", "G3", "B3"], ped, 0.05)
    teklar = ["A4", "D5", "F#5", "E5", "B4", "A5"]
    for i, ad in enumerate(teklar):
        p.ekle(c(2, 0) + 1.2 + i * 1.25, koro(hz(ad), 2.6, atak=0.4, birakma=1.6), pan=RNG.uniform(-0.7, 0.7), kazanc=0.08)
        p.ekle(c(2, 0) + 1.2 + i * 1.25, can(hz(ad) * 2, 2.5), pan=RNG.uniform(-0.7, 0.7), kazanc=0.02)
    gruplar = [["D5", "F#5"], ["B4", "D5", "G5"], ["A4", "E5"], ["G4", "B4", "D5"]]
    for i, g in enumerate(gruplar):
        akor(p, c(2, 2) + 1.0 + i * 1.2, 2.8, g, koro, 0.055, 0.9)

    # 3 · Büyük Müzik: dolgun, yükselen armoni
    t = c(3, 0) - 0.45
    ilerleme = [["D3", "A3", "D4", "F#4", "A4"], ["B2", "F#3", "B3", "D4", "F#4"],
                ["G2", "D3", "G3", "B3", "D4"], ["A2", "E3", "A3", "C#4", "E4"],
                ["D3", "A3", "D4", "F#4", "A4", "D5"]]
    uzunluk = (sahne_bit[3] - t + 0.4) / len(ilerleme)
    for i, notalar in enumerate(ilerleme):
        k = 0.06 + 0.015 * i
        akor(p, t + i * uzunluk, uzunluk + 1.6, notalar, ped, k * 0.8)
        akor(p, t + i * uzunluk, uzunluk + 1.6, notalar[-3:], koro, k * 0.7, 0.9)
        p.ekle(t + i * uzunluk, ped(hz(notalar[0]) / 2, uzunluk + 1.6, 500), kazanc=0.12)
        for j, ad in enumerate(notalar[1:] + notalar[1:3]):
            p.ekle(t + i * uzunluk + j * uzunluk / 7, can(hz(ad) * 2, 2.2), pan=(j % 3 - 1) * 0.5, kazanc=0.025)

    # 4 · Melkor: karanlık, minör, nabız; uzakta Sönmez Alev'in parıltısı
    t = sahne_bit[3] + 0.3
    uz = c(5, 0) - t
    p.ekle(t, ped(hz("D1"), uz + 1.0, 300), kazanc=0.16)
    akor(p, t, uz / 2 + 1.5, ["D3", "F3", "A3"], ped, 0.05)
    akor(p, t + uz / 2, uz / 2 + 1.5, ["Bb2", "D3", "F3"], ped, 0.05)
    for k in np.arange(t + 0.8, c(5, 0) - 0.3, 1.25):
        p.ekle(k, nabiz(), kazanc=0.22)
    p.ekle(c(4, 2) + 2.5, saf(hz("A6"), c(4, 3) - c(4, 2) - 2.0, atak=1.5, birakma=1.5), pan=0.5, kazanc=0.018)
    p.ekle(c(4, 5), sert(hz("Eb3"), c(5, 0) - c(4, 5) + 0.3, atak=2.0, birakma=0.4, tremolo=6), kazanc=0.03)

    # 5 · Uyumsuzluk ve üçüncü tema
    t = c(5, 0) - 0.2
    p.ekle(t, gumbur(2.0, 120, 50), kazanc=0.35)
    for ad in ["D3", "Eb3", "Ab3", "A2"]:
        p.ekle(t, sert(hz(ad), c(5, 2) - t + 0.3, atak=0.02, birakma=0.3), pan=RNG.uniform(-0.6, 0.6), kazanc=0.05)
    akor(p, t, c(5, 2) - t + 0.5, ["D3", "A3", "F#4"], koro, 0.03)
    # Ilúvatar'ın yeni teması
    t2 = c(5, 2)
    akor(p, t2, 2.0, ["G3", "B3", "D4", "G4"], koro, 0.05)
    akor(p, t2 + 1.6, 2.2, ["A3", "C#4", "E4", "A4"], koro, 0.055)
    # Melkor onu da bastırdı
    t3 = c(5, 3)
    for ad in ["D2", "Eb3", "Ab3", "D4"]:
        p.ekle(t3, sert(hz(ad), c(5, 4) - t3 + 0.2, atak=0.02, birakma=0.2), pan=RNG.uniform(-0.6, 0.6), kazanc=0.06)
    p.ekle(t3, gumbur(2.0, 110, 45), kazanc=0.3)
    # Üçüncü tema: derin ve kederli + gürültülü, tekrar eden motif
    t4 = c(5, 4)
    bit = sahne_bit[5] + 0.2
    keder = [["D2", "A2", "D3", "F3", "A3"], ["Bb1", "F2", "Bb2", "D3", "F3"],
             ["G1", "D2", "G2", "Bb2", "D3"], ["A1", "E2", "A2", "C#3", "E3"]]
    ku = (bit - t4) / len(keder)
    for i, notalar in enumerate(keder):
        akor(p, t4 + i * ku, ku + 1.4, notalar, ped, 0.07)
        akor(p, t4 + i * ku, ku + 1.4, notalar[-2:], koro, 0.05)
    motif = ["D4", "Eb4", "Ab4"]
    uyum = ["D4", "E4", "A4"]
    birlesme = c(5, 7)
    k = t4 + 0.3
    i = 0
    while k < bit - 0.25:
        ad = uyum[i % 3] if k > birlesme + 1.2 else motif[i % 3]
        kaz = 0.04 if k > birlesme + 1.2 else 0.055
        p.ekle(k, sert(hz(ad), 0.28, atak=0.005, birakma=0.12, tremolo=11), pan=0.4 * ((i % 3) - 1), kazanc=kaz)
        k += 0.36
        i += 1
    p.ekle(akor_ani - 2.4, yukselen(2.35), kazanc=0.10)

    # 6 · Son akor: dev majör akor, sonra saygılı bir sessizlik
    notalar = ["D1", "D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"]
    for ad in notalar:
        f = hz(ad)
        p.ekle(akor_ani, ped(f, 7.5, 3500) * zarf(int(7.5 * SR), 0.02, 6.5, 1.6), pan=RNG.uniform(-0.8, 0.8), kazanc=0.07)
        if f > 140:
            p.ekle(akor_ani, koro(f, 7.0, atak=0.05, birakma=6.0), pan=RNG.uniform(-0.8, 0.8), kazanc=0.05)
    p.ekle(akor_ani, gumbur(5.0, 100, 30), kazanc=0.6)
    p.ekle(akor_ani, suzgec(gurultu(2.5), 1500, "high") * np.exp(-np.arange(int(2.5 * SR)) / SR / 0.5), kazanc=0.06)
    t = akor_ani + 5.5
    akor(p, t, sahne_bit[6] - t + 1.5, ["D3", "A3", "D4", "F#4"], ped, 0.05)

    # 7 · Kapanış: umutlu, sakin; dünyanın ilk motifi
    t = sahne_bit[6] + 0.3
    akor(p, t, 6.0, ["D3", "A3", "E4", "F#4", "A4"], ped, 0.05)
    akor(p, t + 5.0, 6.0, ["G2", "D3", "B3", "D4", "F#4"], ped, 0.05)
    akor(p, t + 10.0, son - t - 10.0, ["D3", "A3", "D4", "F#4", "A4"], ped, 0.055)
    akor(p, t + 10.0, son - t - 10.0, ["A4", "D5"], koro, 0.03)
    for j, ad in enumerate(["A4", "D5", "E5", "F#5", "A5", "F#5", "E5", "D5"]):
        p.ekle(t + 1.0 + j * 0.9, can(hz(ad), 3.0), pan=0.3 * np.sin(j), kazanc=0.04)
    p.ekle(sahne_bit[7] + 0.25, can(hz("D6"), 3.5), kazanc=0.05)


BOLUMLER = {1: bolum01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bolum", type=pathlib.Path)
    ap.add_argument("--zaman", type=pathlib.Path, required=True)
    ap.add_argument("--cikti", type=pathlib.Path, required=True)
    a = ap.parse_args()
    bolum = json.loads(a.bolum.read_text(encoding="utf-8"))
    z = json.loads(a.zaman.read_text(encoding="utf-8"))

    p = Parca(z["sure"] + 4.0)
    BOLUMLER[bolum["bolum"]](p, z)
    sol, sag = reverb(p.sol, p.sag)
    n = int(z["sure"] * SR)
    x = np.stack([sol[:n], sag[:n]], axis=1)
    x = suzgec(x.T, 30, "high").T
    son = int(2.5 * SR)
    x[-son:] *= np.linspace(1, 0, son)[:, None] ** 1.5
    x /= max(1e-6, np.abs(x).max())
    x = np.tanh(1.2 * x) / np.tanh(1.2)
    sf.write(a.cikti, (0.9 * x).astype(np.float32), SR)
    print(f"müzik: {a.cikti} ({n / SR:.1f} sn)")


if __name__ == "__main__":
    main()
