#!/usr/bin/env python3
"""Seslendirme ve müziği miksler, görüntüyle birleştirip yayına hazır MP4 üretir.

- Seslendirmeye hafif yankı ve alçak geçiren temizlik uygulanır.
- Müzik, konuşma sırasında otomatik olarak kısılır (ducking).
- Ses, TikTok/YouTube için -14 LUFS'a iki geçişli loudnorm ile ayarlanır.

Kullanım:
  python3 birlestir.py --klasor build/ --cikti bolum01.mp4
"""
import argparse
import json
import pathlib
import re
import subprocess

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MUZIK_GORELI_DB = -11.0   # müzik, konuşma yokken sese göre
KISMA_DB = 8.0            # konuşma sırasında müziğe ek kısma


def zarf_takip(x, atak=0.02, birakma=0.35):
    a_k = np.exp(-1 / (atak * SR))
    b_k = np.exp(-1 / (birakma * SR))
    seviye = np.abs(x)
    # blok bazında hızlı yaklaşık takip
    blok = 256
    n = len(seviye) // blok
    tepe = seviye[: n * blok].reshape(n, blok).max(1)
    cikti = np.zeros(n, np.float32)
    onceki = 0.0
    a_b, b_b = a_k ** blok, b_k ** blok
    for i, v in enumerate(tepe):
        k = a_b if v > onceki else b_b
        onceki = k * onceki + (1 - k) * v
        cikti[i] = onceki
    tam = np.repeat(cikti, blok)
    return np.pad(tam, (0, max(0, len(x) - len(tam))), mode="edge")[: len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", type=pathlib.Path, required=True)
    ap.add_argument("--cikti", type=pathlib.Path, required=True)
    a = ap.parse_args()
    k = a.klasor
    sure = json.loads((k / "zaman.json").read_text(encoding="utf-8"))["sure"]
    n = int(round(sure * SR))

    ses, _ = sf.read(k / "ses.wav", dtype="float32")
    muzik, _ = sf.read(k / "muzik.wav", dtype="float32")
    ses = np.pad(ses, (0, max(0, n - len(ses))))[:n]
    muzik = np.pad(muzik, ((0, max(0, n - len(muzik))), (0, 0)))[:n]

    ses = sosfilt(butter(2, 75, "high", fs=SR, output="sos"), ses).astype(np.float32)
    rng = np.random.default_rng(3)
    t = np.arange(int(0.9 * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-t / 0.22)
    ir = sosfilt(butter(2, 4000, fs=SR, output="sos"), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    yanki = fftconvolve(ses, ir)[:n].astype(np.float32)
    ses = ses + 0.10 * yanki

    aktif = np.abs(ses) > 0.02
    ses_rms = np.sqrt(np.mean(ses[aktif] ** 2)) if aktif.any() else 0.1
    muzik_rms = np.sqrt(np.mean(muzik ** 2)) + 1e-9
    muzik *= ses_rms / muzik_rms * 10 ** (MUZIK_GORELI_DB / 20)

    zarf = zarf_takip(ses)
    zarf = np.clip(zarf / (np.percentile(zarf[zarf > 1e-4], 90) + 1e-9), 0, 1)
    kazanc = 10 ** (-KISMA_DB * zarf / 20)
    mix = muzik * kazanc[:, None] + ses[:, None]
    mix /= max(1.0, np.abs(mix).max() / 0.98)
    ham = k / "mix_ham.wav"
    sf.write(ham, mix, SR)

    olcum = subprocess.run([FFMPEG, "-hide_banner", "-i", str(ham), "-af",
                            "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", olcum, re.S).group(0))
    filtre = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
              f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(k / "goruntu.mp4"), "-i", str(ham),
                    "-af", filtre + ",aresample=48000", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{sure:.3f}", "-movflags", "+faststart", str(a.cikti)], check=True)
    print(f"hazır: {a.cikti}")


if __name__ == "__main__":
    main()
