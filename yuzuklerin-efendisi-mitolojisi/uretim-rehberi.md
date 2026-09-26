# Üretim Rehberi

Her bölümü aynı kalitede ve aynı "seri" hissiyle üretmek için gerekenler burada. Kısaca: **dikey 9:16, 2:00–2:30, ilk 3 saniyede kanca, sabit görsel stil, telifsiz müzik, ekranda altyazı.**

## Format özeti

| Ayar | Değer |
|---|---|
| En-boy oranı | 9:16 dikey |
| Çözünürlük | 1080 × 1920, 30 fps |
| Süre | 2:00–2:30 (TikTok ve YouTube Shorts'ta aynı dosya) |
| Ses | Seslendirme −14 LUFS civarı; müzik sesin altında (−20 / −25 dB) |
| Altyazı | Ekrana gömülü, büyük, ortada |
| Kapak | Bölüm numarası + 3–5 kelimelik kapak yazısı (her bölüm dosyasında hazır) |

> YouTube Shorts 3 dakikaya kadar video kabul ediyor, TikTok ise çok daha uzun videolara izin veriyor; 2:00–2:30 ikisine de sorunsuz uyar. Platform kuralları zaman zaman değiştiği için yüklemeden önce süre sınırını bir kez kontrol et.

## Bir bölümün yapısı

Her senaryo 8 sahneden oluşur ve aynı ritmi izler:

1. **Kanca (ilk ~10 saniye):** İlk cümle, izleyicinin bildiği bir şeye (Gandalf, Sauron, Ak Ağaç, Legolas...) bağlanan şaşırtıcı bir iddiadır ve ilk 3 saniyede söylenir; kanca yazısı da o anda ekranda olmalı. "Merhaba arkadaşlar" gibi girişler **yok**; video doğrudan kancayla başlar.
2. **Anlatım (sahne 2–7):** Her sahne 12–20 saniye. Sahne başına 3–5 görsel; her görsel 3–5 saniye ekranda kalır.
3. **Kapanış (son 10 saniye):** Bir sonraki bölümün merak uyandıran fragmanı ve takip çağrısı. Ekranda "SONRAKİ BÖLÜM: ..." yazısı.

Senaryo dosyalarındaki zaman damgaları, sakin bir anlatım hızına göre hesaplanmış tahminlerdir. Kendi kaydından sonra görsel geçişleri gerçek sese göre yerleştir.

## Seslendirme kaydı

- **Metin:** `teleprompter/` klasöründe her bölümün yalnızca okunacak metni var. Telefonda bir teleprompter uygulamasına (CapCut'ın kendi teleprompter'ı, BIGVU vb.) yapıştırabilirsin.
- **İsimler:** Kayıttan önce [telaffuz sözlüğüne](telaffuz-sozlugu.md) bak; her bölüm dosyasının üstünde o bölümün zor isimleri de yazılı.
- **Ortam:** Halı, perde ve yastıklı küçük bir oda en iyisidir. Telefonu ağzından 15–20 cm uzakta tut, mümkünse yaka mikrofonu kullan.
- **Ton:** Masal anlatır gibi, sakin ama canlı. Cümle sonlarında kısa duraklar ver; `...` gördüğün yerde 1 saniyeye yakın bekle.
- **Süre ayarı:** Kayıt 2:30'u geçerse önce duraklamaları kısalt, sonra kurguda sesi %105–110 hızlandır. 2:00'ın altında kalırsa daha yavaş oku.
- **Birden fazla deneme:** Her sahneyi ayrı kaydetmek kurguyu kolaylaştırır; hata yaptığında yalnızca o sahneyi tekrar okursun.

Metni değiştirirsen süreyi yeniden hesaplamak için:

```bash
python3 araclar/sure-hesapla.py yuzuklerin-efendisi-mitolojisi/sezon-1/*.md          # yalnızca rapor
python3 araclar/sure-hesapla.py --yaz yuzuklerin-efendisi-mitolojisi/sezon-1/*.md    # zaman damgalarını güncelle
python3 araclar/sure-hesapla.py --teleprompter yuzuklerin-efendisi-mitolojisi/teleprompter/sezon-1 \
    yuzuklerin-efendisi-mitolojisi/sezon-1/*.md                                         # teleprompter metinlerini yenile
```

Hızlı konuşuyorsan `--hiz 6`, yavaş konuşuyorsan `--hiz 5` ekle.

## Görsel stil

Serinin tanınır olması için **bütün görseller aynı stilde** olmalı. Her sahnedeki AI promptunun sonuna aşağıdaki STİL ekini ekle:

```
STİL: , epic painterly dark fantasy illustration, oil painting texture, cinematic lighting, volumetric light and atmospheric haze, muted earthy palette with luminous gold and silver highlights, highly detailed, vertical 9:16 composition, no text, no watermark
```

Midjourney kullanıyorsan sonuna `--ar 9:16` ekle. ChatGPT, Gemini, Leonardo, Ideogram, Adobe Firefly gibi araçlarda "dikey 9:16" ifadesi yeterli. Kullandığın aracın ticari kullanım şartlarını bir kez kontrol et.

**Hareket:** Durağan görselleri canlandırmak için:
- Her görsele yavaş zoom veya kaydırma (Ken Burns) uygula; CapCut'ta "Animasyon → Yakınlaştır".
- Üstüne hafif sis, toz, kıvılcım veya yıldız kaplamaları ekle (düşük opaklıkta).
- İstersen önemli sahneleri (Ağaçların ölümü, Güneş'in doğuşu) Runway, Kling, Luma gibi görselden-videoya araçlarla 3–5 saniyelik hareketli kliplere çevir.

**Kurallar:**
- Eru Ilúvatar **asla bir kişi olarak gösterilmez**; yalnızca ışık.
- Ainur ve Valar'ın ilk bölümlerde yüzleri net görünmesin; ışık figürleri gibi kalsınlar. Bölüm 3'ten itibaren belirgin karakterlere geçilir.
- Karakterleri film ve dizi oyuncularına benzetme (bkz. Telif ve güvenlik).

## Karakter görünümleri

Aynı karakter farklı bölümlerde görüneceği için promptlarda bu tarifleri kullan; mümkünse ilk ürettiğin başarılı görseli "referans görsel" olarak sakla ve sonraki üretimlerde kullan.

| Karakter | Görünüm (prompta eklenecek) |
|---|---|
| **Melkor / Morgoth** | `towering dark lord in black jagged armor, face in shadow, burning ember eyes, later wearing an iron crown` |
| **Manwë** | `regal king in sapphire-blue robes, silver-white hair, calm wise face, eagles around him` |
| **Varda** | `radiant queen with dark hair crowned with stars, white shimmering gown, light spilling from her hands` |
| **Ulmo** | `colossal sea lord, armor of silver-green scales, crown of sea foam, horn of white shell` |
| **Aulë** | `mighty broad-shouldered smith, bronze skin, leather apron, great hammer, forge light` |
| **Yavanna** | `tall woman robed in green, golden hair with leaves, gentle, tree-like presence` |
| **Mandos** | `stern hooded figure in dark robes, pale face, halls of shadow` |
| **Nienna** | `grey-cloaked mourning woman, silver tears, compassionate` |
| **Oromë** | `hunter lord on a great white horse, long horn, green and gold hunting garb` |
| **Tulkas** | `golden-haired golden-bearded giant warrior, bare fists, laughing` |
| **Olórin (Gandalf)** | `gentle young spirit in soft grey robes` → Orta Dünya'da: `old wanderer in grey cloak, wide pointed hat, staff` |
| **Sauron (erken)** | `handsome copper-haired apprentice smith, eyes that glint red` |
| **Balrog** | `towering demon of shadow and flame, whip of fire` |
| **Elfler (Cuiviénen)** | `tall graceful elves with dark hair, simple primal garments, starlight on their faces` |
| **Cüceler** | `stout dwarves with long braided beards, stone-carved halls` |
| **Fëanor** | `tall intense elven lord, dark hair, piercing grey eyes, smith's hands, fiery aura` |
| **Fingolfin** | `tall noble elven prince, dark hair, blue and silver banners` |
| **Galadriel** | `young elven woman with radiant golden hair, determined gaze` |
| **Thingol / Melian** | `silver-haired elf king in a grey mantle` / `radiant woman spirit with nightingales` |
| **Ungoliant** | `colossal spider made of living darkness, webs of black unlight` |

## Kurgu (CapCut akışı)

1. Yeni proje → 9:16.
2. Ses kaydını ekle; gereksiz boşlukları kes.
3. Görselleri sahne sırasına göre sese hizala (senaryodaki zaman damgaları başlangıç noktası).
4. Her görsele yavaş zoom; sahne geçişlerinde kısa geçiş (karartma/çözülme). Aşırı efektten kaçın.
5. Kaplamalar: sis, toz, kıvılcım (düşük opaklık).
6. Ekran yazıları: senaryodaki **Ekran yazısı** satırları. Büyük, kalın, serif bir yazı tipi (ör. Cinzel, Cormorant) ve ince siyah gölge.
7. Otomatik altyazı: Metin → Otomatik altyazı → Türkçe. **Özel isimleri mutlaka elle düzelt** (Ilúvatar, Fëanor, Silmaril...). Altyazıyı ekran yazılarıyla çakışmayacak şekilde ortaya yerleştir.
8. Müzik: sesin altına; seslendirme sırasında otomatik kısma (ducking).
9. Ses efektleri: gümbürtü, rüzgâr, kılıç, alev; sahne notlarında önerilenler.
10. Güvenli alan: TikTok ve Shorts arayüzü ekranın **alt ~%20'sini ve sağ kenarını** kapatır. Önemli yazıları orta bölgede tut.
11. Dışa aktar: 1080p, 30 fps.

## Müzik ve ses

- Film veya dizi müziklerini (Howard Shore, *Güç Yüzükleri* müzikleri vb.) **kullanma**. Telif talebi gelir; YouTube'da 1 dakikadan uzun Shorts videoları telif talebi aldığında engellenebiliyor.
- TikTok için TikTok'un **Ticari Müzik Kütüphanesi**, YouTube için **YouTube Ses Kitaplığı** güvenlidir. Aynı müziği iki platformda da kullanacaksan, lisansı açıkça her platforma izin veren telifsiz bir kaynak seç ve lisansını sakla.
- Tarz önerisi: ambient, epik orkestral, koro, düşük tempolu. Her bölümde aynı "tema" müziği kullanmak serinin tanınmasına yardımcı olur.

## Telif ve güvenlik

- **Görseller:** Film/dizi karelerini, resmi afişleri ve Alan Lee, John Howe, Ted Nasmith gibi sanatçıların eserlerini izinsiz kullanma. Kendi ürettiğin AI görsellerini veya lisanslı görselleri kullan. Promptlara yaşayan sanatçıların adını yazma.
- **Benzerlik:** Gandalf'ı, Galadriel'i vb. film oyuncularına benzetme; kendi yorumunu oluştur.
- **Logo ve yazı tipi:** Resmi *Yüzüklerin Efendisi* logosunu ve film yazı tipini kullanma.
- **Metin:** Senaryolar kendi anlatımımızdır; kitaptan yalnızca kısa alıntılar (kendi çevirimiz) var. Türkçe çevirilerden uzun pasajlar okuma.
- **Şiddet:** Akraba Katliamı gibi sahnelerde kan ve vahşet göstermeden, ima ederek anlat; platform kurallarına takılmaz.
- Açıklamalarda kaynak olarak Tolkien'in eserlerini belirtmek (her bölümde hazır) hem şeffaflık hem de keşfedilme için iyidir.

## Yayın stratejisi

**Takvim:** Haftada 3 bölüm (örneğin Pazartesi, Çarşamba, Cuma) → Sezon 1 yaklaşık 5 haftada biter. Aynı saatlerde yüklemek izleyiciye alışkanlık kazandırır; akşam saatleri genellikle iyi çalışır, ama birkaç haftalık analitiğe bakıp kendi en iyi saatini bul.

**Başlamadan önce:** İlk 3 bölümü hazır et ve arka arkaya (ör. 3 gün üst üste) yükle. Yeni gelen izleyici profilinde "devamını" bulabilsin.

**TikTok:**
- Kapakta bölüm numarası ve kapak yazısı mutlaka olsun (profil ızgarasında seri düzenli görünür).
- Profilinde **Oynatma listesi** özelliği açıksa "Tolkien Mitolojisi — Sezon 1" listesi oluştur ve her bölümü ekle.
- Bölüm 1'i profilde sabitle.
- Açıklamada 3–5 hashtag yeterli (her bölümde hazır).
- 1 dakikadan uzun videolar TikTok'un içerik üretici gelir programına uygun olabilir; şartları TikTok'ta güncel olarak kontrol et.

**YouTube Shorts:**
- Başlık kalıbı: `[Merak uyandıran cümle] | Tolkien Mitolojisi #N`.
- "Tolkien Mitolojisi — Sezon 1" oynatma listesi oluştur.
- Her Short'a **İlgili video** olarak bir sonraki (veya önceki) bölümü bağla.
- Sezon bitince bütün bölümleri birleştirip ~30 dakikalık yatay (16:9) bir "Sezon 1 derlemesi" yükle. Bunun için görselleri baştan 16:9 da üretmek veya dikey videoyu bulanık arka planla yatay kadraja oturtmak gerekir.

**Etkileşim:**
- Her bölüm dosyasında bir **sabitlenecek yorum** sorusu var; yükledikten hemen sonra kendi yorumun olarak yaz ve sabitle.
- İlk saatte gelen yorumlara cevap ver.
- Tolkien hayranları detaylara çok dikkat eder; hata yakalayan yorumlara teşekkür edip doğrusunu sabitlemek güven kazandırır.

## Hashtag havuzu

- **Her bölümde:** `#yüzüklerinefendisi #tolkien #silmarillion #ortadünya`
- **Konuya göre:** `#gandalf #sauron #morgoth #elfler #cüceler #galadriel #legolas #fëanor #gondor #hobbit #şelob`
- **Genel:** `#mitoloji #fantastik #kitap #kitaptavsiyesi #booktok #fantasy #lotr`
- **YouTube:** Açıklamanın sonuna `#shorts` + 3–4 etiket.

## Bölüm kontrol listesi

- [ ] Telaffuzlar kontrol edildi
- [ ] Ses kaydı 2:00–2:30 arasında
- [ ] Bütün görseller aynı STİL ekiyle üretildi
- [ ] İlk 3 saniyede kanca yazısı ekranda
- [ ] Altyazıdaki özel isimler elle düzeltildi
- [ ] Önemli yazılar güvenli alanda
- [ ] Müzik telifsiz ve lisansı kayıtlı
- [ ] Kapak: bölüm no + kapak yazısı
- [ ] Başlık, açıklama, hashtagler dosyadan kopyalandı
- [ ] Oynatma listesine eklendi, YouTube'da ilgili video bağlandı
- [ ] Sabitlenecek yorum yazıldı ve sabitlendi
