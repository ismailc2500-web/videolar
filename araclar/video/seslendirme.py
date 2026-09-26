#!/usr/bin/env python3
"""Bölüm metnini Piper Türkçe sesiyle seslendirir ve zaman çizelgesini çıkarır.

Her cümle (ve "..." ile ayrılan parçalar) ayrı sentezlenir, baş/son sessizlikleri
kırpılır ve belirlenen duraklarla yan yana dizilir. Konuşma hızı, toplam süre
bölümün hedef süresine (ör. tam 120 sn) oturacak şekilde otomatik ayarlanır.

Çıktılar (--cikti klasörüne):
  ses.wav        44.1 kHz mono seslendirme
  zaman.json     sahne ve altyazı birimlerinin başlangıç/bitiş zamanları

Ses modeli: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  vits-piper-tr_TR-fahrettin-medium (CC0) önerilir.
"""
import argparse
import json
import pathlib
import re

import numpy as np
import sherpa_onnx
from scipy.signal import resample_poly

SR = 44100
BAS = 0.45          # videonun başındaki sessizlik
PARCA = 0.40        # "..." ile ayrılan parçalar arası
CUMLE = 0.28        # cümleler arası
SAHNE = 0.85        # sahneler arası
SON_HEDEF = 3.0     # son cümleden sonra kalan kapanış süresi


def model_yukle(klasor):
    klasor = pathlib.Path(klasor)
    onnx = next(klasor.glob("*.onnx"))
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(onnx),
                tokens=str(klasor / "tokens.txt"),
                data_dir=str(klasor / "espeak-ng-data"),
                noise_scale=0.6,
                noise_scale_w=0.8,
            ),
            num_threads=4,
        )
    )
    return sherpa_onnx.OfflineTts(cfg)


def birimler(bolum):
    """Sahne > cümle > parça listesini düzleştirir."""
    sonuc = []
    for si, sahne in enumerate(bolum["sahneler"]):
        for ci, cumle in enumerate(sahne["cumleler"]):
            parcalar = [p.strip() for p in re.split(r"(?<=\.\.\.)\s+", cumle) if p.strip()]
            for pi, parca in enumerate(parcalar):
                sonuc.append({"sahne": si, "cumle": ci, "parca": pi,
                              "son_parca": pi == len(parcalar) - 1, "metin": parca})
    return sonuc


def okunus(metin, sozluk):
    for yazim in sorted(sozluk, key=len, reverse=True):
        metin = metin.replace(yazim, sozluk[yazim])
    metin = re.sub(r"[\"“”]", "", metin)
    metin = re.sub(r"(\w)['’](\w)", r"\1\2", metin)
    return metin


def kirp(x, esik=0.012, pay=0.03):
    idx = np.flatnonzero(np.abs(x) > esik)
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pay * SR))
    b = min(len(x), idx[-1] + int(pay * SR))
    return x[a:b]


def sentezle(tts, metin, hiz):
    s = tts.generate(metin, sid=0, speed=hiz)
    x = np.asarray(s.samples, dtype=np.float32)
    x = resample_poly(x, SR, s.sample_rate).astype(np.float32)
    return kirp(x)


def dizi(bolum, birim_listesi, sesler):
    """Birimleri zamana yerleştirir; (toplam konuşma, duraklar, zamanlar) döndürür."""
    t = BAS
    zamanlar = []
    onceki = None
    for b, x in zip(birim_listesi, sesler):
        if onceki is not None:
            if b["sahne"] != onceki["sahne"]:
                t += bolum["sahneler"][b["sahne"]].get("on_sessizlik", SAHNE)
            elif b["cumle"] != onceki["cumle"]:
                t += CUMLE
            else:
                t += PARCA
        bas = t
        t += len(x) / SR
        zamanlar.append((bas, t))
        onceki = b
    return zamanlar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bolum", type=pathlib.Path)
    p.add_argument("--model", required=True)
    p.add_argument("--cikti", type=pathlib.Path, required=True)
    p.add_argument("--hiz", type=float, default=1.0, help="başlangıç hızı")
    p.add_argument("--asr", help="Whisper modeli; verilirse her birim için en net okuma seçilir")
    p.add_argument("--deneme", type=int, default=3, help="--asr ile birim başına okuma sayısı")
    a = p.parse_args()

    bolum = json.loads(a.bolum.read_text(encoding="utf-8"))
    hedef = bolum["hedef_sure"]
    tts = model_yukle(a.model)
    liste = birimler(bolum)
    for b in liste:
        b["ses_metni"] = okunus(b["metin"], bolum.get("telaffuz", {}))

    hedef_son = hedef - SON_HEDEF
    denemeler = []
    siradaki = a.hiz
    for tur in range(6):
        hiz = siradaki
        sesler = [sentezle(tts, b["ses_metni"], hiz) for b in liste]
        zamanlar = dizi(bolum, liste, sesler)
        son = zamanlar[-1][1]
        konusma = sum(len(x) for x in sesler) / SR
        print(f"tur {tur}: hız {hiz:.3f} → konuşma {konusma:.1f} sn, son cümle {son:.2f} sn")
        if abs(hedef_son - son) < 0.5:
            break
        denemeler.append((hiz, son))
        if len(denemeler) >= 2 and denemeler[-1][1] != denemeler[-2][1]:
            (h1, s1), (h2, s2) = denemeler[-2:]
            siradaki = h2 + (hedef_son - s2) * (h2 - h1) / (s2 - s1)
        else:
            siradaki = hiz * konusma / (konusma + hedef_son - son)
    if a.asr:
        from kontrol import benzerlik, dinle, tanıyıcı_yukle
        tanıyıcı = tanıyıcı_yukle(a.asr)
        secilen = []
        for b, x in zip(liste, sesler):
            adaylar = [x] + [sentezle(tts, b["ses_metni"], hiz) for _ in range(a.deneme - 1)]
            puanlar = [benzerlik(b["metin"], dinle(tanıyıcı, y, SR)) for y in adaylar]
            en_iyi = int(np.argmax(puanlar))
            print(f"  {max(puanlar):.2f} (aday {en_iyi + 1}/{len(adaylar)}) {b['metin']}")
            secilen.append(adaylar[en_iyi])
        sesler = secilen
        zamanlar = dizi(bolum, liste, sesler)
        son = zamanlar[-1][1]
    if not 1.0 <= hedef - son <= 5.0:
        raise SystemExit(f"Seslendirme hedefe oturmadı: son cümle {son:.2f} sn. Metni kısalt/uzat.")

    toplam = int(round(hedef * SR))
    ses = np.zeros(toplam, dtype=np.float32)
    for (bas, bit), x in zip(zamanlar, sesler):
        i = int(round(bas * SR))
        ses[i:i + len(x)] += x[: max(0, toplam - i)]
    ses *= 0.89 / max(1e-6, np.abs(ses).max())

    a.cikti.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(a.cikti / "ses.wav", ses, SR)

    sahneler = []
    for si in range(len(bolum["sahneler"])):
        idx = [k for k, b in enumerate(liste) if b["sahne"] == si]
        sahneler.append({"bas": zamanlar[idx[0]][0], "bit": zamanlar[idx[-1]][1]})
    (a.cikti / "zaman.json").write_text(json.dumps({
        "hiz": hiz,
        "sure": hedef,
        "sahneler": sahneler,
        "birimler": [dict(b, bas=z[0], bit=z[1]) for b, z in zip(liste, zamanlar)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"son cümle bitişi {zamanlar[-1][1]:.2f} sn / hedef {hedef} sn, hız {hiz:.3f}")


if __name__ == "__main__":
    main()
