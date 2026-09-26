#!/usr/bin/env python3
"""Bölüm videosunun görüntüsünü üretir.

Görseller koddan (prosedürel) üretilir; sahneler sahneler_bNN.py dosyalarında,
ortak araçlar motor.py'dedir. Üzerine sahne başlıkları, zamanlanmış Türkçe
altyazılar ve kapanış kartı eklenir. Kareler paralel işlenip ffmpeg ile kodlanır.

Kullanım:
  python3 goruntu.py bolumler/bolum01.json --klasor build/ --fontlar fonts/
  python3 goruntu.py ... --kare 15.5 --kare 60   # yalnızca önizleme PNG'leri
  python3 goruntu.py ... --kapak 4.0             # kapak.png
"""
import argparse
import json
import pathlib
import subprocess

import cv2
import numpy as np

from motor import FFMPEG, FPS, H, Baglam, Cizer, Zaman, _baslat, _parca_isle, bindir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bolum", type=pathlib.Path)
    ap.add_argument("--klasor", type=pathlib.Path, required=True, help="ses.wav, muzik.wav, zaman.json klasörü")
    ap.add_argument("--fontlar", type=pathlib.Path, required=True)
    ap.add_argument("--cikti", type=pathlib.Path)
    ap.add_argument("--kare", type=float, action="append", help="yalnızca bu saniyelerin PNG önizlemesi")
    ap.add_argument("--surec", type=int, default=4)
    ap.add_argument("--kapak", type=float, help="bu saniyedeki görüntüden kapak.png üret")
    a = ap.parse_args()
    bolum = json.loads(a.bolum.read_text(encoding="utf-8"))
    zaman = json.loads((a.klasor / "zaman.json").read_text(encoding="utf-8"))
    ayarlar = (bolum, zaman, a.klasor, a.fontlar)

    if a.kapak is not None:
        c = Cizer(*ayarlar)
        kare = c.kare(a.kapak, yazilar=False)
        bindir(kare, c.etiket, 0.125 * H, 0.85)
        bindir(kare, c.kapak_yazi, 0.25 * H)
        yol = a.klasor / "kapak.png"
        cv2.imwrite(str(yol), cv2.cvtColor(kare, cv2.COLOR_RGB2BGR))
        print(yol)
        return

    if a.kare:
        c = Cizer(*ayarlar)
        for t in a.kare:
            yol = a.klasor / f"onizleme_{t:06.2f}.png"
            cv2.imwrite(str(yol), cv2.cvtColor(c.kare(t), cv2.COLOR_RGB2BGR))
            print(yol)
        return

    Baglam(Zaman(bolum, zaman), a.klasor)  # doku önbelleği
    toplam = int(round(zaman["sure"] * FPS))
    parca_sayisi = a.surec * 3
    sinirlar = np.linspace(0, toplam, parca_sayisi + 1).astype(int)
    isler = [(sinirlar[i], sinirlar[i + 1], str(a.klasor / f"parca_{i:02d}.mp4")) for i in range(parca_sayisi)]
    from multiprocessing import Pool
    with Pool(a.surec, initializer=_baslat, initargs=(ayarlar,)) as havuz:
        for yol in havuz.imap(_parca_isle, isler):
            print("hazır:", yol, flush=True)
    liste = a.klasor / "parcalar.txt"
    liste.write_text("".join(f"file '{pathlib.Path(y).name}'\n" for _, _, y in isler))
    sessiz = a.klasor / "goruntu.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(liste), "-c", "copy", str(sessiz)], check=True)
    print("görüntü:", sessiz)


if __name__ == "__main__":
    main()
