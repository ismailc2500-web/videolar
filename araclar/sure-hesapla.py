#!/usr/bin/env python3
"""Bölüm senaryolarının seslendirme süresini tahmin eder.

Senaryo dosyalarında seslendirme metni "> " ile başlayan satırlardır ve
"### Sahne N · 0:00–0:00 · Başlık" biçimindeki sahne başlıklarının altında yer alır.

Türkçede hece sayısı ünlü sayısına eşittir; süre, hece sayısı ve noktalama
duraklarından hesaplanır. Varsayılan hız, sakin ve dramatik bir anlatım içindir.

Kullanım:
  python3 araclar/sure-hesapla.py yuzuklerin-efendisi-mitolojisi/sezon-1/*.md
  python3 araclar/sure-hesapla.py --yaz yuzuklerin-efendisi-mitolojisi/sezon-1/*.md
  python3 araclar/sure-hesapla.py --teleprompter yuzuklerin-efendisi-mitolojisi/teleprompter/sezon-1 \
      yuzuklerin-efendisi-mitolojisi/sezon-1/*.md

  --yaz            Sahne zaman damgalarını ve "Tahmini süre" satırını dosyaya yazar.
  --teleprompter   Yalnızca seslendirme metnini içeren .txt dosyaları üretir.
  --hiz            Saniyedeki hece sayısı (varsayılan 5.5). Hızlı konuşuyorsan 6, yavaşsan 5 dene.
"""
import argparse
import pathlib
import re

UNLULER = set("aeıioöuüâîûëéáíóúäAEIİOÖUÜÂÎÛËÉÁÍÓÚÄ")
SAHNE = re.compile(r"^### Sahne (\d+) · \d+:\d{2}–\d+:\d{2} · (.*)$")
SURE_SATIRI = re.compile(r"^\| \*\*Tahmini süre\*\* \|.*\|$")


def hece(metin):
    return sum(1 for h in metin if h in UNLULER)


def kelime(metin):
    return len(re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", metin))


def saniye(metin, hiz):
    duraklama = (
        metin.count("...") * 0.6
        + len(re.findall(r"(?<!\.)[.!?](?!\.)", metin)) * 0.35
        + len(re.findall(r"[,;:—]", metin)) * 0.15
    )
    return hece(metin) / hiz + duraklama


def zaman(sn):
    sn = int(round(sn))
    return f"{sn // 60}:{sn % 60:02d}"


def analiz(yol, hiz):
    satirlar = yol.read_text(encoding="utf-8").splitlines()
    sahneler = []  # [satır_no, başlık, metin]
    for i, satir in enumerate(satirlar):
        m = SAHNE.match(satir)
        if m:
            sahneler.append([i, m.group(2), ""])
        elif satir.startswith("> ") and sahneler:
            sahneler[-1][2] += " " + satir[2:].strip()
    return satirlar, sahneler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dosyalar", nargs="+", type=pathlib.Path)
    p.add_argument("--hiz", type=float, default=5.5)
    p.add_argument("--yaz", action="store_true")
    p.add_argument("--teleprompter", type=pathlib.Path)
    a = p.parse_args()

    for yol in sorted(a.dosyalar):
        satirlar, sahneler = analiz(yol, a.hiz)
        if not sahneler:
            continue
        t = 0.0
        for no, (satir_no, baslik, metin) in enumerate(sahneler, 1):
            bas, t = t, t + saniye(metin, a.hiz)
            satirlar[satir_no] = f"### Sahne {no} · {zaman(bas)}–{zaman(t)} · {baslik}"
        tum = " ".join(s[2] for s in sahneler)
        ozet = f"{zaman(t)} (≈{kelime(tum)} kelime · {hece(tum)} hece)"
        durum = "✅" if 120 <= t <= 150 else "⚠️ 2:00–2:30 dışında"
        print(f"{yol.name:45s} {ozet:32s} {durum}")

        if a.yaz:
            satirlar = [
                f"| **Tahmini süre** | {ozet} |" if SURE_SATIRI.match(s) else s
                for s in satirlar
            ]
            yol.write_text("\n".join(satirlar) + "\n", encoding="utf-8")

        if a.teleprompter:
            a.teleprompter.mkdir(parents=True, exist_ok=True)
            baslik = satirlar[0].lstrip("# ").strip()
            govde = "\n\n".join(s[2].strip() for s in sahneler)
            (a.teleprompter / f"{yol.stem}.txt").write_text(
                f"{baslik}\n\n{govde}\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
