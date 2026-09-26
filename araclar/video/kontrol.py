#!/usr/bin/env python3
"""Seslendirmeyi Whisper ile geri çözüp beklenen metinle karşılaştırır.

Kulakla dinlemeden telaffuz hatalarını yakalamak içindir: her altyazı birimi
ayrı ayrı yazıya çevrilir ve kelime benzerliği raporlanır.

Model: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  sherpa-onnx-whisper-small
"""
import argparse
import difflib
import json
import pathlib
import re

import numpy as np
import sherpa_onnx
import soundfile as sf
from scipy.signal import resample_poly


def sade(metin):
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"[^\w\s]", " ", metin).split()


def tanıyıcı_yukle(model):
    model = pathlib.Path(model)
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(next(model.glob("*encoder.int8.onnx"))),
        decoder=str(next(model.glob("*decoder.int8.onnx"))),
        tokens=str(next(model.glob("*tokens.txt"))),
        language="tr", task="transcribe", num_threads=4,
    )


def dinle(tanıyıcı, parca, sr):
    parca = resample_poly(parca, 16000, sr).astype(np.float32)
    sessizlik = np.zeros(8000, dtype=np.float32)
    akis = tanıyıcı.create_stream()
    akis.accept_waveform(16000, np.concatenate([sessizlik, parca, sessizlik]))
    tanıyıcı.decode_stream(akis)
    return akis.result.text.strip()


def benzerlik(beklenen, duyulan):
    return difflib.SequenceMatcher(None, sade(beklenen), sade(duyulan)).ratio()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("klasor", type=pathlib.Path, help="seslendirme.py çıktı klasörü")
    p.add_argument("--model", required=True, type=pathlib.Path)
    a = p.parse_args()

    tanıyıcı = tanıyıcı_yukle(a.model)
    ses, sr = sf.read(a.klasor / "ses.wav", dtype="float32")
    zaman = json.loads((a.klasor / "zaman.json").read_text(encoding="utf-8"))

    oranlar = []
    for b in zaman["birimler"]:
        parca = ses[int((b["bas"] - 0.05) * sr): int((b["bit"] + 0.05) * sr)]
        duyulan = dinle(tanıyıcı, parca, sr)
        oran = benzerlik(b["metin"], duyulan)
        oranlar.append(oran)
        isaret = "  " if oran >= 0.8 else "⚠️"
        print(f"{isaret} {oran:.2f} | {b['metin']}\n        → {duyulan}")
    print(f"\nOrtalama kelime benzerliği: {np.mean(oranlar):.3f}")


if __name__ == "__main__":
    main()
