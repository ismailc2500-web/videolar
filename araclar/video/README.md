# Video üretim hattı

Bir bölümün JSON dosyasından, seslendirmesi, müziği, görüntüsü ve altyazısıyla **yayına hazır dikey MP4** üretir. Kullanılan her şey açık lisanslı veya koddan üretilmiştir:

| Parça | Kaynak | Lisans |
|---|---|---|
| Seslendirme | Piper Türkçe ses modeli `tr_TR-fettah-medium` (sherpa-onnx ile) | CC0 |
| Müzik ve efektler | `muzik.py` içinde koddan sentezlenir | Telifsiz, bize ait |
| Görüntü | `goruntu.py` içinde koddan (prosedürel) üretilir | Bize ait |
| Yazı tipleri | Cinzel, Cormorant Garamond, Montserrat (Google Fonts) | SIL OFL |
| Telaffuz kontrolü | Whisper small (sherpa-onnx) | MIT |

## Adımlar

```bash
cd araclar/video
pip install sherpa-onnx numpy scipy opencv-python-headless pillow imageio-ffmpeg soundfile

# 1) Seslendirme: süreyi hedefe (ör. tam 120 sn) oturtur, her cümle için en net okumayı seçer
python3 seslendirme.py bolumler/bolum01.json --model MODELLER/vits-piper-tr_TR-fettah-medium \
    --cikti build/ --hiz 1.15 --asr MODELLER/sherpa-onnx-whisper-small --deneme 4

# 2) (İsteğe bağlı) Telaffuz raporu
python3 kontrol.py build/ --model MODELLER/sherpa-onnx-whisper-small

# 3) Müzik
python3 muzik.py bolumler/bolum01.json --zaman build/zaman.json --cikti build/muzik.wav

# 4) Görüntü (önce birkaç önizleme karesi, sonra tamamı)
python3 goruntu.py bolumler/bolum01.json --klasor build/ --fontlar FONTLAR/ --kare 17.8 --kare 56
python3 goruntu.py bolumler/bolum01.json --klasor build/ --fontlar FONTLAR/
python3 goruntu.py bolumler/bolum01.json --klasor build/ --fontlar FONTLAR/ --kapak 4.0

# 5) Miksaj ve birleştirme
python3 birlestir.py --klasor build/ --cikti bolum01.mp4
```

Modeller: https://github.com/k2-fsa/sherpa-onnx/releases (`tts-models` ve `asr-models` etiketleri).
Yazı tipleri: https://github.com/google/fonts (`ofl/cinzel`, `ofl/cormorantgaramond`, `ofl/montserrat`).

## Bölüm dosyası (`bolumler/bolumNN.json`)

- `hedef_sure`: videonun saniye cinsinden tam süresi.
- `sahneler[].gorsel`: görüntü motorundaki sahne türü (`kanca`, `eru`, `ainur`, `muzik`, `melkor`, `uyumsuzluk`, `akor`, `dunya`).
- `sahneler[].ekran`: sahne başlığı ve alt başlığı; `ekran_cumle`, başlığın hangi cümlede belireceği.
- `sahneler[].cumleler`: seslendirme metni. `...` ile ayrılan yerlerde kısa duraklama yapılır.
- `telaffuz`: ses modeline gönderilen okunuşlar (ör. `Ilúvatar` → `İluvvatar`). Altyazıda doğru yazım kalır.
- `vurgu`: altyazıda altın renkle vurgulanan kelimeler.
- `sonraki`, `kapak`: kapanış kartı ve kapak yazıları.

Yeni bir bölüm için yeni sahne türleri `goruntu.py` içindeki `SAHNELER`, yeni müzik ise `muzik.py` içindeki `BOLUMLER` sözlüğüne eklenir.
