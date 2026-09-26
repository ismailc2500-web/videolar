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


# ---------------------------------------------------------------- ek çalgılar (Bölüm 2+)

def ors(f=880.0, sure=1.6):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = np.zeros(n, np.float32)
    for oran, g, tau in [(1, 1.0, 0.7), (2.76, 0.6, 0.45), (5.40, 0.4, 0.25), (8.93, 0.25, 0.15), (13.3, 0.12, 0.08)]:
        x += g * np.exp(-t / tau) * np.sin(2 * np.pi * f * oran * t)
    x += suzgec(gurultu(sure), 3000, "high") * np.exp(-t / 0.02) * 0.8
    return x * zarf(n, 0.001, 0.05)


def ruzgar_ses(sure, parlaklik=900):
    n = int(sure * SR)
    x = gurultu(sure)
    parcalar = np.array_split(x, 40)
    t = np.linspace(0, 1, 40)
    cikti = [suzgec(suzgec(p, 250 + parlaklik * (0.6 + 0.4 * np.sin(6 * ti)), "low"), 150, "high") for p, ti in zip(parcalar, t)]
    x = np.concatenate(cikti)
    return x * (0.6 + 0.4 * np.sin(np.linspace(0, 3 * np.pi, n))) * zarf(n, sure * 0.3, sure * 0.3)


def dalga_ses(sure):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = suzgec(gurultu(sure), 700)
    return x * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.14 * t)) ** 2) * zarf(n, 1.0, 1.5)


def flut(f, sure, atak=0.12, birakma=0.5):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip(t / 0.4, 0, 1)
    faz = 2 * np.pi * f * np.cumsum(vib) / SR
    x = np.sin(faz) + 0.25 * np.sin(2 * faz) + 0.08 * np.sin(3 * faz)
    x += 0.05 * suzgec(gurultu(sure), 3000) 
    return x * zarf(n, atak, min(birakma, sure / 2))


def boru(f, sure, atak=0.5, birakma=1.2):
    h = [1 / k ** 0.9 / (1 + (k * f / 1100) ** 2) for k in range(1, 20)]
    x = toplamsal(f, sure, h, sapma_cent=(-5, 0, 5), vibrato=0.002, vib_hiz=4.5)
    return np.tanh(1.5 * x) * zarf(len(x), atak, birakma)


def davul(f0=95.0, sure=0.9):
    return gumbur(sure, f0, 42.0) * zarf(int(sure * SR), 0.002, 0.3)


def zil(sure=3.0):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    return suzgec(gurultu(sure), 5000, "high") * np.exp(-t / 0.9)


# ---------------------------------------------------------------- Bölüm 2

def bolum02(p, z):
    birim = z["birimler"]

    def c(sahne, cumle=0):
        return next(b["bas"] for b in birim if b["sahne"] == sahne and b["cumle"] == cumle)

    bit = [s["bit"] for s in z["sahneler"]]
    son = z["sure"]

    # 0 · Şarkı → rüya → dünya
    for j, ad in enumerate(["A4", "D5", "E5", "F#5"]):
        p.ekle(0.1 + j * 0.45, can(hz(ad), 3.0), pan=0.3 * (j - 1.5) / 1.5, kazanc=0.08)
    akor(p, 0.0, c(0, 1) + 1.0, ["D3", "A3", "D4", "F#4"], koro, 0.06)
    p.ekle(0.0, ped(hz("D2"), c(0, 1) + 1.5, 600), kazanc=0.12)
    akor(p, c(0, 1) - 0.3, 4.0, ["D3", "A3", "E4", "F#4", "C#5"], ped, 0.04)
    gercek = c(0, 2) + 1.5
    p.ekle(gercek - 1.4, yukselen(1.35), kazanc=0.08)
    p.ekle(gercek, gumbur(3.5, 110, 35), kazanc=0.45)
    akor(p, gercek, bit[0] - gercek + 3.0, ["D2", "A2", "D3", "F#3", "A3", "D4"], ped, 0.06)
    akor(p, gercek, bit[0] - gercek + 3.0, ["A4", "D5", "F#5"], koro, 0.04)

    # 1 · Görü: hayranlık
    t = c(1, 0) - 0.4
    ilerleme = [["B2", "F#3", "A3", "C#4", "D4"], ["G2", "D3", "F#3", "B3", "D4"],
                ["F#2", "D3", "A3", "D4"], ["A2", "E3", "A3", "C#4", "E4"]]
    uz = (bit[1] - t + 0.6) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.5, nt, ped, 0.05)
        akor(p, t + i * uz, uz + 1.5, nt[-2:], koro, 0.03)
    k = t + 0.5
    arp = ["D5", "F#5", "A5", "C#6", "B5", "A5", "F#5", "E5"]
    i = 0
    while k < bit[1]:
        p.ekle(k, can(hz(arp[i % 8]), 2.0), pan=0.5 * np.sin(i), kazanc=0.022 + 0.012 * (k > c(1, 2)))
        k += 0.45
        i += 1

    # 2 · Ilúvatar'ın Çocukları: flüt teması; sonunda Melkor'un kıskançlığı
    t = c(2, 0) - 0.3
    akor(p, t, c(2, 1) - t + 0.5, ["D3", "A3", "D4"], ped, 0.04)
    melodi = [("D5", 0.6), ("E5", 0.4), ("F#5", 0.9), ("A5", 0.6), ("B5", 0.5), ("A5", 0.4), ("F#5", 0.9),
              ("E5", 0.6), ("D5", 0.5), ("E5", 0.4), ("F#5", 0.8), ("D5", 1.6)]
    k = c(2, 1)
    for ad, d in melodi:
        if k > c(2, 4) - 0.3:
            break
        p.ekle(k, flut(hz(ad), d + 0.25), pan=-0.15, kazanc=0.055)
        k += d
    for i, nt in enumerate([["D3", "A3", "F#4"], ["B2", "F#3", "D4"], ["G2", "D3", "B3"], ["A2", "E3", "C#4"]]):
        tt = c(2, 1) + i * 2.3
        if tt > c(2, 4) - 0.5:
            break
        akor(p, tt, 2.8, nt, ped, 0.05)
        for j, ad in enumerate(nt):
            p.ekle(tt + j * 0.12, can(hz(ad) * 2, 1.4), pan=0.4, kazanc=0.025)
    akor(p, c(2, 3) - 0.2, c(2, 4) - c(2, 3) + 0.4, ["D4", "F#4", "A4", "D5"], koro, 0.045)
    t = c(2, 4)
    p.ekle(t, gumbur(2.5, 80, 32), kazanc=0.35)
    for ad in ["D2", "Eb2", "A2"]:
        p.ekle(t, sert(hz(ad), bit[2] - t + 1.2, atak=0.05, birakma=1.0, tremolo=5), kazanc=0.04)

    # 3 · Unsurlar
    for j, ad in enumerate(["D5", "A5", "F#5"]):
        p.ekle(c(3, 0) + 0.6 + j * 0.8, can(hz(ad), 2.5), pan=(j - 1) * 0.6, kazanc=0.05)
    akor(p, c(3, 0), c(3, 1) - c(3, 0) + 0.5, ["D3", "A3", "D4", "E4"], ped, 0.045)
    t = c(3, 1) - 0.3
    p.ekle(t, dalga_ses(c(3, 2) - t + 1.0), kazanc=0.10)
    p.ekle(t, ped(hz("D1"), c(3, 2) - t + 0.8, 250), kazanc=0.20)
    akor(p, t, c(3, 2) - t + 0.8, ["D2", "A2", "D3"], ped, 0.05)
    p.ekle(c(3, 1) + 1.0, boru(hz("D3"), 1.8), kazanc=0.07)
    p.ekle(c(3, 1) + 1.0, boru(hz("A3"), 1.8), kazanc=0.05)
    t = c(3, 2) - 0.3
    p.ekle(t, ruzgar_ses(c(3, 3) - t + 0.8), kazanc=0.12)
    akor(p, t, c(3, 3) - t + 0.8, ["A4", "D5", "E5", "A5"], koro, 0.04)
    t = c(3, 3) - 0.3
    akor(p, t, c(3, 4) - t + 0.6, ["D2", "A2", "F#3"], ped, 0.05)
    for n in range(4):
        vurus = c(3, 3) + n * 1.1 + 0.62 * 1.1
        if vurus < c(3, 4) + 0.4:
            p.ekle(vurus, ors(), pan=0.3, kazanc=0.10)
            p.ekle(vurus, davul(120, 0.5), kazanc=0.25)
    t = c(3, 4) - 0.2
    p.ekle(t, ruzgar_ses(c(3, 5) - t + 1.5, 1600), kazanc=0.12)
    for ad in ["E6", "F6", "B5", "C6"]:
        p.ekle(t, sert(hz(ad), c(3, 5) - t + 0.4, atak=1.0, birakma=0.6, tremolo=9), pan=RNG.uniform(-0.7, 0.7), kazanc=0.006)
    p.ekle(t, ped(hz("D1"), c(3, 5) - t + 0.5, 200), kazanc=0.18)
    t = c(3, 5)
    akor(p, t, c(3, 6) - t + 0.6, ["D3", "A3", "C#4", "F#4"], ped, 0.045)
    for j, ad in enumerate(["D6", "F#6", "A6", "C#7", "D7", "A6", "F#6", "E6", "D6", "A5"]):
        p.ekle(t + 0.3 + j * 0.34, can(hz(ad), 2.2), pan=0.5 * np.sin(j * 1.3), kazanc=0.03)
    t = c(3, 6)
    akor(p, t, bit[3] - t + 1.5, ["G2", "D3", "B3", "D4"], ped, 0.065)
    akor(p, t + 2.2, bit[3] - t - 0.5, ["D3", "A3", "D4", "F#4"], koro, 0.06)
    p.ekle(t + 0.5, boru(hz("D3"), 2.5, atak=0.8), kazanc=0.05)

    # 4 · Görü kayboluyor
    an = c(4, 0) + 1.6
    p.ekle(an - 1.2, yukselen(1.15), kazanc=0.06)
    p.ekle(an, zil(2.5), kazanc=0.05)
    p.ekle(an, gumbur(2.0, 70, 30), kazanc=0.25)
    t = an + 0.4
    p.ekle(t, ped(hz("D1"), bit[4] - t + 1.5, 200), kazanc=0.10)
    p.ekle(t, ped(hz("Eb1"), bit[4] - t + 1.5, 200), kazanc=0.05)
    akor(p, c(4, 2) - 0.3, bit[4] - c(4, 2) + 1.5, ["A2", "D3", "E3", "A3"], ped, 0.04)
    for k in np.arange(c(4, 2), bit[4], 1.4):
        p.ekle(k, nabiz(), kazanc=0.15)

    # 5 · Eä!
    ea = c(5, 1)
    p.ekle(c(5, 0) + 0.3, saf(hz("A6"), ea - c(5, 0), atak=1.5, birakma=0.2), kazanc=0.02)
    p.ekle(ea - 2.2, yukselen(2.15), kazanc=0.12)
    p.ekle(ea, gumbur(5.0, 120, 28), kazanc=0.7)
    p.ekle(ea, zil(3.5), kazanc=0.08)
    for ad in ["D1", "D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"]:
        f = hz(ad)
        p.ekle(ea, ped(f, 7.0, 3200) * zarf(int(7.0 * SR), 0.02, 5.5, 1.5), pan=RNG.uniform(-0.8, 0.8), kazanc=0.06)
        if f > 140:
            p.ekle(ea, koro(f, 6.5, atak=0.05, birakma=5.0), pan=RNG.uniform(-0.8, 0.8), kazanc=0.045)
    for j in range(24):
        p.ekle(ea + 0.3 + j * 0.13 + RNG.uniform(0, 0.08), can(hz(["D6", "F#6", "A6", "E6"][j % 4]), 1.8),
               pan=RNG.uniform(-0.9, 0.9), kazanc=0.018)
    akor(p, c(5, 3) - 0.3, bit[5] - c(5, 3) + 1.5, ["D3", "A3", "D4", "F#4", "A4"], ped, 0.045)
    akor(p, c(5, 3), bit[5] - c(5, 3) + 1.2, ["F#5", "A5"], koro, 0.03)

    # 6 · Valar iniyor: kahramanca tema
    t = c(6, 0) - 0.4
    ilerleme = [["D2", "A2", "D3", "F#3"], ["G2", "D3", "G3", "B3"], ["A2", "E3", "A3", "C#4"], ["D2", "A2", "D3", "F#3"]]
    uz = (c(6, 3) - t) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.0, nt, ped, 0.065)
        for ad in nt[1:]:
            p.ekle(t + i * uz, boru(hz(ad), uz + 0.8), pan=RNG.uniform(-0.5, 0.5), kazanc=0.032)
    for i in range(5):
        inis = c(6, 0) + 0.3 + i * 0.45 + 0.7
        p.ekle(inis, can(hz(["D5", "F#5", "A5", "D6", "F#6"][i]), 2.5), pan=(i - 2) * 0.3, kazanc=0.05)
        p.ekle(inis, davul(100, 0.7), kazanc=0.18)
    akor(p, c(6, 2) - 0.2, 3.5, ["D4", "F#4", "A4", "D5"], koro, 0.05)
    akor(p, c(6, 3), c(6, 4) - c(6, 3) + 0.5, ["B1", "F#2", "B2", "D3"], ped, 0.05)
    t = c(6, 4)
    yuk = [["G2", "D3", "G3", "B3"], ["A2", "E3", "A3", "C#4"], ["B2", "F#3", "B3", "D4"], ["D3", "A3", "D4", "F#4"]]
    uz = (bit[6] - t + 0.6) / len(yuk)
    for i, nt in enumerate(yuk):
        akor(p, t + i * uz, uz + 1.0, nt, ped, 0.05 + 0.01 * i)
        akor(p, t + i * uz, uz + 1.0, nt[-2:], koro, 0.035 + 0.008 * i)
    for k in np.arange(t, bit[6] + 0.3, 0.5):
        p.ekle(k, davul(90, 0.5), kazanc=0.10 + 0.12 * (k - t) / (bit[6] - t + 0.3))

    # 7 · Melkor geliyor
    t = c(7, 0) - 0.3
    p.ekle(t, gumbur(4.0, 70, 26), kazanc=0.55)
    p.ekle(t, ped(hz("D1"), son - t, 220), kazanc=0.11)
    for ad in ["D2", "Eb2", "Ab2"]:
        p.ekle(t, sert(hz(ad), c(7, 2) - t + 0.6, atak=0.8, birakma=0.8, tremolo=4), pan=RNG.uniform(-0.6, 0.6), kazanc=0.03)
    for k in np.arange(t + 0.8, c(7, 2), 0.8):
        p.ekle(k, davul(80, 0.8), kazanc=0.15)
    p.ekle(c(7, 1), gumbur(2.0, 90, 40), kazanc=0.3)
    t = c(7, 2)
    akor(p, t, 2.8, ["G2", "D3", "G3", "B3", "D4"], ped, 0.045)
    akor(p, t + 2.8, c(7, 3) - t - 2.4, ["A2", "E3", "A3", "C#4", "E4"], ped, 0.045)
    p.ekle(t + 0.2, boru(hz("G3"), 2.6), kazanc=0.03)
    p.ekle(t + 2.9, boru(hz("A3"), 2.6), kazanc=0.03)
    boz = c(7, 3)
    for n in range(2):
        t0 = boz + n * 1.2
        p.ekle(t0, can(hz("D5"), 1.5), kazanc=0.05)
        p.ekle(t0 + 0.6, gumbur(1.8, 100, 35), kazanc=0.45)
        p.ekle(t0 + 0.6, suzgec(gurultu(1.2), 900) * np.exp(-np.arange(int(1.2 * SR)) / SR / 0.3), kazanc=0.12)
    t = c(7, 4)
    for k in np.arange(t, c(7, 5) + 0.4, 0.25):
        p.ekle(k, davul(110 if int((k - t) / 0.25) % 2 else 80, 0.4), kazanc=0.13)
    for ad in ["D3", "Eb3", "A3", "D4"]:
        p.ekle(t, sert(hz(ad), c(7, 5) - t + 0.3, atak=0.02, birakma=0.3), pan=RNG.uniform(-0.6, 0.6), kazanc=0.03)
    akor(p, t, c(7, 5) - t + 0.3, ["D4", "F#4", "A4"], koro, 0.05)
    t = c(7, 5)
    akor(p, t, son - t, ["D3", "A3", "D4", "F#4", "A4"], ped, 0.05)
    akor(p, t + 1.0, son - t - 1.0, ["A4", "D5"], koro, 0.03)
    for j, ad in enumerate(["A4", "D5", "E5", "F#5", "A5", "F#5", "E5", "D5"]):
        p.ekle(t + 0.4 + j * 0.55, can(hz(ad), 2.6), pan=0.3 * np.sin(j), kazanc=0.04)
    p.ekle(z["birimler"][-1]["bit"] + 0.2, can(hz("D6"), 3.0), kazanc=0.05)


# ---------------------------------------------------------------- Bölüm 3

def yagmur_ses(sure):
    n = int(sure * SR)
    x = suzgec(gurultu(sure), 5000, "high") * 0.6 + suzgec(gurultu(sure), 1200) * 0.4
    return x * zarf(n, 1.0, 1.5)


def bolum03(p, z):
    from sahneler_b03 import _Z, olaylar
    O = olaylar(z)
    Z = _Z(z)
    c = Z.c
    bit = [s["bit"] for s in z["sahneler"]]
    sinir = Z.sinir
    son = z["sure"]

    # 0 · Kanca: yüz montajı → gökteki tanrılar → Valar
    for i, t in enumerate(O["yuzler"]):
        p.ekle(t, davul(90 + 10 * i, 0.8), kazanc=0.30)
        p.ekle(t, can(hz(["D4", "F4", "A4", "C5", "D5"][i]), 2.0), pan=(i - 2) * 0.3, kazanc=0.06)
    p.ekle(0.0, ped(hz("D1"), O["kanca_b"] + 1.0, 250), kazanc=0.15)
    akor(p, 0.0, O["kanca_b"] + 1.0, ["D3", "A3", "F4"], koro, 0.04)
    t = O["kanca_b"]
    p.ekle(t, gumbur(3.0, 90, 30), kazanc=0.35)
    akor(p, t, O["hayir"] - t + 0.8, ["D3", "A3", "D4", "F#4", "A4"], koro, 0.045)
    akor(p, t, O["hayir"] - t + 0.8, ["D2", "A2", "D3"], ped, 0.06)
    akor(p, O["hayir"], O["kanca_c"] - O["hayir"] + 0.6, ["D2", "A2", "D3", "F3", "Bb3"], ped, 0.06)
    akor(p, O["insanlar"], O["kanca_c"] - O["insanlar"] + 0.6, ["F4", "A4", "D5"], koro, 0.035)
    p.ekle(O["kanca_c"] - 1.2, yukselen(1.15), kazanc=0.10)
    t = O["kanca_c"]
    p.ekle(t, gumbur(3.5, 110, 32), kazanc=0.5)
    p.ekle(t, zil(2.5), kazanc=0.05)
    akor(p, t, bit[0] - t + 1.5, ["D2", "A2", "D3", "F#3", "A3", "D4"], ped, 0.06)
    for ad in ["D3", "A3", "F#4"]:
        p.ekle(t, boru(hz(ad), bit[0] - t + 1.0, atak=0.3), pan=RNG.uniform(-0.5, 0.5), kazanc=0.035)
    for j in range(10):
        p.ekle(t + 0.2 + j * 0.3, can(hz(["D5", "F#5", "A5", "D6"][j % 4]), 2.0), pan=RNG.uniform(-0.8, 0.8),
               kazanc=0.025)

    # 1 · Máhanaxar: tören teması, Melkor'un tahtı çöker, Manwë'ye yakınlaşma
    t = sinir[1] - 0.2
    ilerleme = [["D2", "A2", "D3", "F#3"], ["B1", "F#2", "B2", "D3"], ["G2", "D3", "G3", "B3"]]
    uz = (O["melkor_cokus"] - t) / 2
    for i, nt in enumerate(ilerleme[:2]):
        akor(p, t + i * uz, uz + 1.0, nt, ped, 0.055)
        akor(p, t + i * uz, uz + 1.0, nt[-2:], koro, 0.03)
    for k in np.arange(t + 0.3, O["melkor_cokus"], 0.9):
        p.ekle(k, davul(70, 0.7), kazanc=0.12)
    m = O["melkor_cokus"]
    p.ekle(m - 0.25, sert(hz("Eb2"), 0.8, atak=0.02, birakma=0.3), kazanc=0.04)
    p.ekle(m, gumbur(3.0, 70, 26), kazanc=0.55)
    p.ekle(m, suzgec(gurultu(1.6), 1200) * np.exp(-np.arange(int(1.6 * SR)) / SR / 0.35), kazanc=0.18)
    akor(p, m + 0.4, O["yakinlas"] - m + 0.4, ["G2", "D3", "G3", "Bb3"], ped, 0.05)
    t = O["yakinlas"]
    p.ekle(t, yukselen(sinir[2] - t + 0.1), kazanc=0.12)
    akor(p, t, sinir[2] - t + 0.5, ["A2", "E3", "A3", "C#4", "E4"], koro, 0.04)

    # 2 · Manwë: gök, rüzgâr, kartallar
    t = sinir[2]
    p.ekle(t, zil(3.0), kazanc=0.06)
    p.ekle(t, gumbur(2.5, 100, 35), kazanc=0.25)
    p.ekle(t - 0.2, ruzgar_ses(bit[2] - t + 1.5, 1400), kazanc=0.14)
    ilerleme = [["D3", "A3", "D4", "F#4", "A4"], ["G2", "D3", "B3", "D4", "G4"], ["B2", "F#3", "B3", "D4", "F#4"],
                ["A2", "E3", "A3", "C#4", "E4"]]
    uz = (bit[2] - t + 0.6) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.2, nt, ped, 0.05)
        akor(p, t + i * uz, uz + 1.2, nt[-3:], koro, 0.025)
    melodi = [("A5", 0.5), ("D6", 0.9), ("C#6", 0.4), ("A5", 0.5), ("F#5", 1.2), ("E5", 0.5), ("F#5", 0.5),
              ("A5", 0.9), ("B5", 0.5), ("A5", 1.4)]
    k = t + 0.8
    for ad, d in melodi:
        if k > O["kartal"] - 0.4:
            break
        p.ekle(k, flut(hz(ad), d + 0.25), pan=0.2, kazanc=0.045)
        k += d
    ka = O["kartal"]
    p.ekle(ka, boru(hz("D4"), 1.4), pan=0.4, kazanc=0.06)
    p.ekle(ka + 0.35, boru(hz("A4"), 1.8), pan=0.4, kazanc=0.05)
    p.ekle(ka + 0.2, ruzgar_ses(2.2, 2500), pan=-0.3, kazanc=0.20)

    # 3 · Varda: yıldızlar tek tek yanar; Işık şişesi; Melkor'un korkusu
    t = sinir[3]
    akor(p, t, O["sise"][0] - t + 0.8, ["D3", "A3", "E4", "F#4", "C#5"], koro, 0.045)
    akor(p, t, O["sise"][0] - t + 0.8, ["D2", "A2", "D3"], ped, 0.05)
    for i in range(14):
        p.ekle(O["yildizlar"] + 0.18 * i, can(hz(["A5", "C#6", "E6", "F#6", "A6", "E6", "C#6"][i % 7]), 2.2),
               pan=np.sin(i * 1.1) * 0.8, kazanc=0.035)
    s0, s1 = O["sise"]
    p.ekle(s0 - 0.8, yukselen(0.8), kazanc=0.08)
    p.ekle(s0, zil(3.0), kazanc=0.07)
    akor(p, s0, s1 - s0 + 0.8, ["A3", "E4", "A4", "C#5", "E5", "A5"], koro, 0.05)
    for ad in ["A5", "E6", "A6"]:
        p.ekle(s0, saf(hz(ad), s1 - s0, atak=0.4, birakma=0.6), pan=RNG.uniform(-0.6, 0.6), kazanc=0.015)
    t = s1
    akor(p, t, bit[3] - t + 1.0, ["D2", "A2", "D3", "F3"], ped, 0.05)
    p.ekle(t + 0.3, sert(hz("D2"), bit[3] - t, atak=0.6, birakma=0.6, tremolo=4), kazanc=0.025)
    akor(p, t + 1.5, bit[3] - t, ["D4", "F#4", "A4", "D5"], koro, 0.04)

    # 4 · Ulmo: derinlik, dalgalar, deniz kabuğu borusu
    t = sinir[4]
    p.ekle(t - 0.3, dalga_ses(bit[4] - t + 1.5), kazanc=0.10)
    p.ekle(t, ped(hz("D1"), bit[4] - t + 1.0, 220), kazanc=0.20)
    ilerleme = [["D2", "A2", "D3", "F3", "A3"], ["Bb1", "F2", "Bb2", "D3", "F3"], ["C2", "G2", "C3", "E3", "G3"],
                ["D2", "A2", "D3", "F#3", "A3"]]
    uz = (bit[4] - t + 0.5) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.2, nt, ped, 0.055)
    u = O["ulmo_ses"]
    p.ekle(u, boru(hz("D2"), 2.8, atak=0.6), kazanc=0.10)
    p.ekle(u + 0.1, boru(hz("A2"), 2.6, atak=0.6), kazanc=0.07)
    akor(p, u, 3.0, ["D4", "A4", "D5"], koro, 0.035)
    for j in range(18):
        p.ekle(t + 0.5 + j * 0.55 + RNG.uniform(0, 0.3), can(hz(["D6", "A6", "F6", "E6"][j % 4]), 1.2),
               pan=RNG.uniform(-0.9, 0.9), kazanc=0.012)

    # 5 · Aulë: örs vuruşları; Melkor'la benzerlik; paylaşmak / sahiplenmek
    t = sinir[5]
    p.ekle(t, ped(hz("A1"), O["paylas"] - t + 0.5, 300), kazanc=0.15)
    akor(p, t, c(5, 1) - t + 0.5, ["A2", "E3", "A3", "C4"], ped, 0.05)
    for v in O["ors"]:
        p.ekle(v, ors(760), pan=0.35, kazanc=0.09)
        p.ekle(v, davul(120, 0.5), kazanc=0.22)
    t = c(5, 1)
    akor(p, t, O["paylas"] - t + 0.3, ["A2", "E3", "Bb3", "D4"], ped, 0.05)
    p.ekle(t + 0.3, sert(hz("E2"), O["paylas"] - t, atak=0.8, birakma=0.4, tremolo=4), pan=-0.4, kazanc=0.025)
    t = O["paylas"]
    akor(p, t, O["sahiplen"] - t + 0.4, ["F2", "C3", "F3", "A3", "C4"], ped, 0.06)
    akor(p, t, O["sahiplen"] - t + 0.4, ["A4", "C5"], koro, 0.035)
    t = O["sahiplen"]
    p.ekle(t, gumbur(2.2, 80, 30), kazanc=0.40)
    for ad in ["D2", "Eb2", "A2"]:
        p.ekle(t, sert(hz(ad), bit[5] - t + 0.8, atak=0.05, birakma=0.6, tremolo=5), pan=RNG.uniform(-0.6, 0.6),
               kazanc=0.035)

    # 6 · Yavanna: pastoral flüt, çiçekler, dev ağaç
    t = sinir[6]
    ilerleme = [["G2", "D3", "G3", "B3", "D4"], ["C3", "G3", "C4", "E4"], ["E2", "B2", "E3", "G3", "B3"],
                ["D3", "A3", "D4", "F#4"]]
    uz = (bit[6] - t + 0.6) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.2, nt, ped, 0.05)
    melodi = [("D5", 0.4), ("G5", 0.8), ("A5", 0.4), ("B5", 0.8), ("D6", 0.6), ("B5", 0.4), ("A5", 0.8),
              ("G5", 0.4), ("E5", 0.6), ("G5", 1.2)]
    k = t + 0.4
    for ad, d in melodi:
        if k > bit[6]:
            break
        p.ekle(k, flut(hz(ad), d + 0.25), pan=-0.2, kazanc=0.05)
        k += d
    for j in range(16):
        p.ekle(t + 0.3 + j * 0.18, can(hz(["G5", "B5", "D6", "G6"][j % 4]), 1.5), pan=np.sin(j) * 0.8,
               kazanc=0.018)
    a = O["agac"]
    p.ekle(a, yukselen(2.5) * 0.3, kazanc=0.1)
    akor(p, a, bit[6] - a + 1.2, ["G3", "B3", "D4", "G4", "B4", "D5"], koro, 0.04)

    # 7 · Mandos: derin çan, karanlık salonlar
    t = sinir[7]
    p.ekle(t, gumbur(3.0, 70, 28), kazanc=0.3)
    p.ekle(t, ped(hz("D1"), bit[7] - t + 1.0, 180), kazanc=0.22)
    akor(p, t, bit[7] - t + 1.0, ["D2", "A2", "D3", "F3"], ped, 0.05)
    akor(p, t + 1.0, bit[7] - t, ["D4", "F4", "A4"], koro, 0.03)
    for k in np.arange(t + 0.1, bit[7] + 0.5, 2.2):
        p.ekle(k, can(hz("D3"), 4.0), kazanc=0.10)
        p.ekle(k, can(hz("A3"), 3.0), kazanc=0.04)
    p.ekle(O["mandos_goz"], sert(hz("A1"), 2.5, atak=0.8, birakma=1.0, tremolo=3), kazanc=0.03)

    # 8 · Nienna: yağmur, ağıt; umut
    t = sinir[8]
    p.ekle(t - 0.3, yagmur_ses(bit[8] - t + 1.5), kazanc=0.05)
    akor(p, t, O["nienna_isik"] - t + 0.5, ["D3", "F3", "A3", "D4"], ped, 0.05)
    akor(p, t + 1.0, O["nienna_isik"] - t, ["F4", "A4"], koro, 0.025)
    melodi = [("A4", 0.8), ("F5", 1.0), ("E5", 0.5), ("D5", 0.5), ("C5", 0.8), ("D5", 1.4), ("A4", 0.6), ("Bb4", 0.6),
              ("A4", 1.5)]
    k = t + 0.5
    for ad, d in melodi:
        if k > O["nienna_isik"] - 0.3:
            break
        p.ekle(k, flut(hz(ad), d + 0.3), pan=0.15, kazanc=0.045)
        k += d
    t = O["nienna_isik"]
    akor(p, t, bit[8] - t + 1.2, ["Bb2", "F3", "Bb3", "D4", "F4"], ped, 0.05)
    akor(p, t + 0.4, bit[8] - t + 0.8, ["F4", "Bb4", "D5"], koro, 0.035)
    for j, ad in enumerate(["D5", "F5", "Bb5", "D6"]):
        p.ekle(t + 0.3 + j * 0.35, can(hz(ad), 2.0), pan=(j - 1.5) * 0.3, kazanc=0.03)

    # 9 · Oromë: av borusu, dörtnala at
    for j, b in enumerate(O["boru"]):
        g = 1.0 if j == 0 else 0.6
        p.ekle(b, boru(hz("D3"), 1.2, atak=0.08), kazanc=0.10 * g)
        p.ekle(b + 0.35, boru(hz("A3"), 1.8, atak=0.1), kazanc=0.09 * g)
        p.ekle(b, gumbur(2.0, 90, 35), kazanc=0.3 * g)
    t = sinir[9]
    akor(p, t, bit[9] - t + 1.0, ["D2", "A2", "D3", "F#3"], ped, 0.05)
    a0, a1 = O["at"]
    for k in np.arange(a0, a1 + 0.2, 0.42):
        for d, ff in ((0.0, 95), (0.1, 80), (0.21, 110)):
            p.ekle(k + d, davul(ff, 0.3), kazanc=0.11)
    akor(p, a0, a1 - a0 + 1.0, ["D3", "A3", "D4", "F#4", "A4"], koro, 0.035)

    # 10 · Tulkas: yumruklar, kahkaha, altın enerji
    t = sinir[10]
    ilerleme = [["D2", "A2", "D3", "F#3"], ["G2", "D3", "G3", "B3"], ["A2", "E3", "A3", "C#4"], ["D2", "A2", "D3", "F#3"]]
    uz = (bit[10] - t + 0.6) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.0, nt, ped, 0.055)
        for ad in nt[1:]:
            p.ekle(t + i * uz, boru(hz(ad), uz + 0.6), pan=RNG.uniform(-0.5, 0.5), kazanc=0.028)
    for k in np.arange(t + 0.2, bit[10], 0.5):
        p.ekle(k, davul(100 if int((k - t) / 0.5) % 2 else 75, 0.4), kazanc=0.12)
    for y in O["yumruk"]:
        p.ekle(y, gumbur(1.5, 120, 40), kazanc=0.45)
        p.ekle(y, suzgec(gurultu(0.4), 2500) * np.exp(-np.arange(int(0.4 * SR)) / SR / 0.06), kazanc=0.2)
    kk = O["kuyruklu"]
    p.ekle(kk - 0.2, yukselen(1.2) * 0.5, kazanc=0.1)
    p.ekle(kk + 1.2, gumbur(2.0, 90, 30), kazanc=0.3)

    # 11 · Maiar → Gandalf ve Sauron → kapanış
    t = sinir[11]
    akor(p, t, O["iki_isim"] - t + 0.5, ["D3", "A3", "D4", "E4", "A4"], ped, 0.05)
    r = O["ruhlar"]
    for j in range(30):
        p.ekle(r + j * 0.1 + RNG.uniform(0, 0.08), can(hz(["D6", "E6", "F#6", "A6", "B6"][j % 5]), 1.6),
               pan=RNG.uniform(-0.9, 0.9), kazanc=0.02)
    akor(p, r, O["iki_isim"] - r + 0.5, ["F#4", "A4", "D5"], koro, 0.035)
    t = O["iki_isim"]
    p.ekle(t, ped(hz("D1"), son - t, 200), kazanc=0.18)
    for k in np.arange(t, O["sauron"], 0.7):
        p.ekle(k, nabiz(), kazanc=0.18)
    g = O["gandalf"]
    akor(p, g, 2.2, ["D3", "A3", "D4", "F#4"], koro, 0.045)
    p.ekle(g, can(hz("D5"), 3.0), kazanc=0.05)
    sa = O["sauron"]
    p.ekle(sa, gumbur(3.0, 75, 26), kazanc=0.5)
    for ad in ["D2", "Eb2", "Ab2"]:
        p.ekle(sa, sert(hz(ad), 3.0, atak=0.05, birakma=1.2, tremolo=4), pan=RNG.uniform(-0.6, 0.6), kazanc=0.035)
    t = c(11, 3)
    akor(p, t, son - t, ["D3", "A3", "D4", "F#4", "A4"], ped, 0.05)
    akor(p, t + 0.5, son - t - 0.5, ["A4", "D5"], koro, 0.03)
    p.ekle(z["birimler"][-1]["bit"] + 0.2, can(hz("D6"), 3.0), kazanc=0.05)


# ---------------------------------------------------------------- Bölüm 4

def kirbac_ses(sure=0.6):
    n = int(sure * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = suzgec(gurultu(sure), 2500, "high") * np.exp(-t / 0.03)
    return x + 0.4 * suzgec(gurultu(sure), 600) * np.exp(-t / 0.15)


def bolum04(p, z):
    from sahneler_b04 import _Z, olaylar
    O = olaylar(z)
    Z = _Z(z)
    bit = [s["bit"] for s in z["sahneler"]]
    sinir = Z.sinir
    son = z["sure"]

    # 0 · Kanca: iki yarım (aydınlık / karanlık) → aynı tür → köprü → üç ışık
    p.ekle(0.0, ped(hz("D1"), O["kanca_b"] + 1.0, 220), kazanc=0.16)
    akor(p, 0.0, O["ayni"] + 0.5, ["D3", "A3", "F#4"], koro, 0.04)
    for ad in ["D2", "Eb2"]:
        p.ekle(0.0, sert(hz(ad), O["ayni"] + 0.3, atak=0.5, birakma=0.4, tremolo=4), pan=0.5, kazanc=0.025)
    p.ekle(O["ayni"] - 0.9, yukselen(0.9), kazanc=0.09)
    p.ekle(O["ayni"], zil(2.5), kazanc=0.06)
    akor(p, O["ayni"], O["kanca_b"] - O["ayni"] + 0.8, ["D3", "A3", "D4", "F#4", "A4"], koro, 0.045)
    for j, ad in enumerate(["D5", "A5", "F#5", "D6"]):
        p.ekle(O["ayni"] + 0.1 + j * 0.25, can(hz(ad), 2.0), pan=(j - 1.5) * 0.4, kazanc=0.04)
    t = O["kanca_b"]
    p.ekle(t, gumbur(2.5, 90, 30), kazanc=0.45)
    for k in np.arange(t + 0.1, O["kanca_c"], 0.45):
        p.ekle(k, davul(85, 0.4), kazanc=0.14)
    akor(p, t, O["kanca_c"] - t + 0.5, ["D2", "A2", "D3", "F3"], ped, 0.06)
    p.ekle(t, boru(hz("D3"), O["kanca_c"] - t, atak=0.2), kazanc=0.05)
    t = O["kanca_c"]
    akor(p, t, bit[0] - t + 1.5, ["D3", "A3", "E4", "A4"], koro, 0.045)
    for j in range(3):
        p.ekle(t + 0.8 + j * 0.2, can(hz(["A5", "D6", "E6"][j]), 2.5), pan=(j - 1) * 0.6, kazanc=0.05)

    # 1 · Maiar: göksel, Büyük Müzik'e gönderme, isimler
    t = sinir[1]
    ilerleme = [["D3", "A3", "D4", "F#4"], ["B2", "F#3", "B3", "D4"], ["G2", "D3", "G3", "B3"], ["A2", "E3", "A3", "C#4"]]
    uz = (bit[1] - t + 0.6) / len(ilerleme)
    for i, nt in enumerate(ilerleme):
        akor(p, t + i * uz, uz + 1.2, nt, ped, 0.05)
    for j in range(20):
        p.ekle(t + 0.3 + j * 0.3 + RNG.uniform(0, 0.1), can(hz(["D6", "F#6", "A6", "E6"][j % 4]), 1.4),
               pan=RNG.uniform(-0.9, 0.9), kazanc=0.015)
    akor(p, O["muzik"], O["sayi"] - O["muzik"] + 0.6, ["D4", "F#4", "A4", "D5"], koro, 0.05)
    arp = ["D5", "F#5", "A5", "D6", "C#6", "A5", "F#5", "E5"]
    for j, k in enumerate(np.arange(O["muzik"], O["sayi"], 0.3)):
        p.ekle(k, can(hz(arp[j % 8]), 1.8), pan=0.5 * np.sin(j), kazanc=0.03)
    for i in range(10):
        p.ekle(O["sayi"] + 0.25 + i * 0.28, can(hz(["A5", "B5", "D6", "E6", "F#6"][i % 5]), 1.8),
               pan=RNG.uniform(-0.7, 0.7), kazanc=0.035)
    akor(p, O["yardim"], bit[1] - O["yardim"] + 1.0, ["G3", "B3", "D4", "G4"], koro, 0.035)

    # 2 · Tanıdık Maiar
    y = O["yuzler"] + [sinir[3]]
    t = y[0]
    p.ekle(t, boru(hz("D4"), 0.5, atak=0.03, birakma=0.2), kazanc=0.07)
    p.ekle(t + 0.35, boru(hz("A4"), 0.5, atak=0.03, birakma=0.2), kazanc=0.07)
    p.ekle(t + 0.7, boru(hz("D5"), 1.4, atak=0.05), kazanc=0.08)
    akor(p, t, y[1] - t + 0.4, ["D3", "A3", "D4", "F#4"], ped, 0.05)
    t = y[1]
    p.ekle(t, gumbur(2.0, 80, 30), kazanc=0.35)
    p.ekle(t, dalga_ses(y[2] - t + 0.6) * 1.5, kazanc=0.14)
    for ad in ["D2", "A2", "C3"]:
        p.ekle(t, sert(hz(ad), y[2] - t + 0.3, atak=0.1, birakma=0.4, tremolo=6), pan=RNG.uniform(-0.6, 0.6),
               kazanc=0.03)
    t = y[2]
    p.ekle(t, dalga_ses(y[3] - t + 1.0), kazanc=0.07)
    akor(p, t, y[3] - t + 0.8, ["F#3", "A3", "C#4", "F#4"], ped, 0.045)
    k = t + 0.2
    for ad, d in [("C#5", 0.5), ("F#5", 0.7), ("E5", 0.5), ("C#5", 0.9)]:
        p.ekle(k, flut(hz(ad), d + 0.2), kazanc=0.04)
        k += d
    t = y[3]
    akor(p, t, bit[2] - t + 1.2, ["D3", "A3", "D4", "F#4"], ped, 0.05)
    akor(p, t, bit[2] - t + 1.2, ["A4", "D5"], koro, 0.04)
    for j in range(14):
        k = t + 0.3 + j * 0.28
        p.ekle(k, flut(hz(["A6", "B6", "A6", "F#6", "E6", "F#6", "A6"][j % 7]), 0.18, atak=0.02, birakma=0.08),
               pan=0.6 * np.sin(j), kazanc=0.02)

    # 3 · Olórin: Lórien'in rüya bahçeleri → korku → seçilme → Gandalf
    t = sinir[3]
    akor(p, t, O["nienna"] - t + 0.5, ["D3", "A3", "E4", "F#4"], koro, 0.04)
    akor(p, t, O["nienna"] - t + 0.5, ["D2", "A2", "D3"], ped, 0.045)
    for j, k in enumerate(np.arange(t + 0.3, O["korku"], 0.42)):
        p.ekle(k, can(hz(["D5", "E5", "F#5", "A5", "B5", "A5", "F#5", "E5"][j % 8]), 1.6), pan=0.4 * np.sin(j),
               kazanc=0.022)
    akor(p, O["nienna"], O["gelecek"] - O["nienna"] + 0.5, ["B2", "F#3", "B3", "D4"], ped, 0.045)
    melodi = [("F#5", 0.6), ("E5", 0.4), ("D5", 0.8), ("B4", 0.6), ("D5", 1.2)]
    k = O["nienna"] + 0.2
    for ad, d in melodi:
        p.ekle(k, flut(hz(ad), d + 0.25), kazanc=0.04)
        k += d
    akor(p, O["gelecek"], O["korku"] - O["gelecek"] + 0.5, ["G2", "D3", "G3", "B3"], ped, 0.05)
    p.ekle(O["gelecek"] + 0.3, boru(hz("G3"), 2.5, atak=0.8), kazanc=0.035)
    t = O["korku"]
    p.ekle(t, ped(hz("D1"), O["manwe"] - t + 0.5, 200), kazanc=0.15)
    akor(p, t, O["manwe"] - t + 0.4, ["D2", "F2", "Ab2", "D3"], ped, 0.045)
    p.ekle(t + 0.3, sert(hz("Ab1"), O["manwe"] - t, atak=0.6, birakma=0.4, tremolo=4), kazanc=0.03)
    t = O["manwe"]
    akor(p, t, O["donusum"] - t + 0.4, ["G2", "D3", "G3", "B3", "D4"], ped, 0.05)
    akor(p, t, O["donusum"] - t + 0.4, ["B4", "D5"], koro, 0.035)
    p.ekle(O["gandalf"] - 1.0, yukselen(1.0), kazanc=0.12)
    t = O["gandalf"]
    p.ekle(t, gumbur(3.0, 110, 32), kazanc=0.5)
    p.ekle(t, zil(3.0), kazanc=0.07)
    akor(p, t, bit[3] - t + 1.5, ["D2", "A2", "D3", "F#3", "A3", "D4"], ped, 0.06)
    akor(p, t, bit[3] - t + 1.5, ["F#4", "A4", "D5"], koro, 0.05)
    for ad in ["D3", "A3"]:
        p.ekle(t, boru(hz(ad), bit[3] - t + 1.0, atak=0.2), pan=RNG.uniform(-0.4, 0.4), kazanc=0.045)

    # 4 · Sauron: demirhane → Mairon → düşüş → Saruman → yüzük
    t = O["soru"]
    p.ekle(t, gumbur(2.0, 70, 28), kazanc=0.3)
    p.ekle(t, sert(hz("D2"), O["demirhane"] - t + 0.3, atak=0.3, birakma=0.3, tremolo=4), kazanc=0.025)
    t = O["demirhane"]
    akor(p, t, O["dusus"] - t + 0.4, ["A2", "E3", "A3", "C#4"], ped, 0.05)
    for v in O["ors"]:
        p.ekle(v, ors(760), pan=0.35, kazanc=0.09)
        p.ekle(v, davul(120, 0.5), kazanc=0.2)
    akor(p, O["mairon"], O["dusus"] - O["mairon"] + 0.3, ["E4", "A4", "C#5", "E5"], koro, 0.04)
    t = O["dusus"]
    p.ekle(t, gumbur(3.0, 70, 26), kazanc=0.45)
    p.ekle(t, ped(hz("D1"), O["saruman"] - t + 0.3, 200), kazanc=0.16)
    for ad in ["D2", "Eb2", "Ab2", "D3"]:
        p.ekle(t + 0.3, sert(hz(ad), O["saruman"] - t, atak=1.2, birakma=0.5, tremolo=4), pan=RNG.uniform(-0.6, 0.6),
               kazanc=0.03)
    akor(p, t + 0.8, O["saruman"] - t, ["D4", "F4", "Ab4"], koro, 0.035)
    t = O["saruman"]
    for ad in ["B5", "E6", "F#6"]:
        p.ekle(t, saf(hz(ad), O["yuzuk"] - t + 0.3, atak=0.3, birakma=0.4), pan=RNG.uniform(-0.6, 0.6), kazanc=0.015)
    akor(p, t, O["yuzuk"] - t + 0.3, ["E2", "B2", "E3", "G3"], ped, 0.04)
    t = O["yuzuk"]
    akor(p, t, bit[4] - t + 1.0, ["D2", "A2", "D3", "F3", "Bb3"], ped, 0.05)
    akor(p, t + 0.5, bit[4] - t + 0.5, ["D4", "F4", "A4"], koro, 0.035)

    # 5 · Balroglar
    t = O["cekim"]
    p.ekle(t, yukselen(O["cikis"] - t) * 0.6, kazanc=0.1)
    akor(p, t, O["cikis"] - t + 0.4, ["D2", "Ab2", "D3", "F3"], koro, 0.04)
    t = O["cikis"]
    p.ekle(t, gumbur(4.0, 60, 24), kazanc=0.6)
    p.ekle(t, ped(hz("D1"), bit[5] - t + 1.0, 180), kazanc=0.2)
    for ad in ["D2", "Eb2", "A2"]:
        p.ekle(t, sert(hz(ad), bit[5] - t + 0.8, atak=0.4, birakma=0.8, tremolo=5), pan=RNG.uniform(-0.6, 0.6),
               kazanc=0.035)
    for k in np.arange(t + 0.3, bit[5] + 0.3, 0.6):
        p.ekle(k, davul(65, 0.8), kazanc=0.16)
    for k in O["kirbac"]:
        p.ekle(k, kirbac_ses(), pan=-0.3, kazanc=0.25)
        p.ekle(k, gumbur(1.5, 100, 40), kazanc=0.3)

    # 6 · Köprü
    t = sinir[6]
    akor(p, t, O["yakin"] - t + 0.5, ["D2", "A2", "D3", "F3", "A3"], ped, 0.05)
    for k in np.arange(t + 0.2, O["gecemezsin"], 0.5):
        p.ekle(k, davul(90 if int((k - t) / 0.5) % 2 else 70, 0.5), kazanc=0.10 + 0.06 * (k - t) / (O["gecemezsin"] - t))
    p.ekle(t + 0.3, boru(hz("D3"), O["yakin"] - t, atak=1.0), kazanc=0.04)
    akor(p, O["yakin"], O["gecemezsin"] - O["yakin"] + 0.3, ["Bb2", "F3", "Bb3", "D4"], ped, 0.055)
    p.ekle(O["gecemezsin"] - 0.9, yukselen(0.9), kazanc=0.12)
    t = O["gecemezsin"]
    p.ekle(t, gumbur(3.0, 120, 30), kazanc=0.6)
    p.ekle(t, zil(2.0), kazanc=0.08)
    akor(p, t, O["vurus"] - t + 0.5, ["D3", "A3", "D4", "F#4", "A4", "D5"], koro, 0.05)
    t = O["vurus"]
    p.ekle(t, gumbur(3.5, 90, 26), kazanc=0.6)
    p.ekle(t, suzgec(gurultu(2.0), 900) * np.exp(-np.arange(int(2.0 * SR)) / SR / 0.5), kazanc=0.25)
    p.ekle(O["gandalf_dusus"] - 0.3, kirbac_ses(), kazanc=0.25)
    p.ekle(O["gandalf_dusus"], gumbur(3.0, 70, 24), kazanc=0.4)
    akor(p, t + 0.3, O["ak"] - t, ["D2", "A2", "D3", "F3"], ped, 0.05)
    p.ekle(O["ak"] - 1.2, yukselen(1.2), kazanc=0.12)
    t = O["ak"]
    p.ekle(t, zil(3.5), kazanc=0.08)
    akor(p, t, bit[6] - t + 1.5, ["D3", "A3", "D4", "F#4", "A4", "D5"], ped, 0.06)
    akor(p, t, bit[6] - t + 1.5, ["F#4", "A4", "D5", "F#5"], koro, 0.05)
    for j in range(12):
        p.ekle(t + 0.2 + j * 0.2, can(hz(["D6", "F#6", "A6", "D7"][j % 4]), 2.0), pan=RNG.uniform(-0.8, 0.8),
               kazanc=0.02)

    # 7 · Kapanış: iki Lamba
    t = sinir[7]
    p.ekle(t, ped(hz("D1"), son - t, 200), kazanc=0.12)
    akor(p, t, O["lamba"][0] - t + 0.4, ["D3", "A3", "E4"], koro, 0.035)
    for i, (l, nt) in enumerate(zip(O["lamba"], (["D3", "A3", "D4", "F#4"], ["G2", "D3", "G3", "B3", "D4"]))):
        p.ekle(l, can(hz(["A5", "D6"][i]), 3.0), kazanc=0.06)
        p.ekle(l, gumbur(2.0, 100, 35), kazanc=0.25)
        akor(p, l, (O["lamba"][1] - l + 0.4) if i == 0 else son - l, nt, ped, 0.05)
    akor(p, O["lamba"][1], son - O["lamba"][1], ["B4", "D5", "G5"], koro, 0.035)
    p.ekle(z["birimler"][-1]["bit"] + 0.2, can(hz("D6"), 3.0), kazanc=0.05)


BOLUMLER = {1: bolum01, 2: bolum02, 3: bolum03, 4: bolum04}


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
