# Gelişmiş Video Takip Sistemi

Gerçek zamanlı video nesne takibi için geliştirilmiş profesyonel Python uygulaması. OpenCV tabanlı gelişmiş algoritmalar, yapay zeka destekli öngörü ve performans optimizasyonu ile donatılmıştır.

## Özellikler

### Takip Algoritmaları
- **CSRT Takip Algoritması** - Yüksek doğruluk ve güvenilirlik
- **Çoklu Ölçekli Şablon Eşleştirme** - Farklı boyutlarda nesne tespiti
- **Rotasyon Değişmez Takip** - Dönen nesneleri takip etme
- **Özellik Tabanlı Eşleştirme** - SIFT/ORB ile gelişmiş tespit

### Yapay Zeka Özellikleri
- **Hareket Öngörüsü** - Kinematik analiz ile pozisyon tahmini
- **Adaptif Öğrenme** - Takip performansından öğrenme
- **Oklüzyon Tespiti** - Nesne gizlendiğinde akıllı kurtarma
- **Kalite Analizi** - Gerçek zamanlı görüntü kalite değerlendirmesi

### Performans Optimizasyonu
- **Adaptif Performans Ayarı** - Otomatik kaynak yönetimi
- **Hafif Mod** - Düşük kaynak kullanımı için optimize
- **Çoklu Performans Seviyesi** - YÜKSEK/ORTA/DÜŞÜK modlar
- **Gerçek Zamanlı İzleme** - FPS ve frame süresi takibi

### Profesyonel Arayüz
- **Analitik Panel** - Detaylı sistem bilgileri
- **Güven Grafikleri** - Takip güvenilirlik göstergesi
- **İz Görselleştirme** - Nesne hareket geçmişi
- **Timeline Kontrolü** - Video ilerleme göstergesi

## Sistem Gereksinimleri

- **İşletim Sistemi:** Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python:** 3.7 veya üzeri
- **RAM:** Minimum 4GB, önerilen 8GB
- **İşlemci:** Çok çekirdekli işlemci önerilir
- **Grafik:** İsteğe bağlı (OpenCV CUDA desteği)

## Kurulum

### Hızlı Kurulum
```bash
# Depoyu klonlayın
git clone <repository-url>
cd chase

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### Gerekli Paketler
```
opencv-python>=4.5.0
numpy>=1.19.0
```

## Kullanım

### Temel Kullanım
```bash
# Video dosyası ile çalıştırma
python enhanced_video_tracker.py video_dosyasi.mp4

# Varsayılan video ile çalıştırma
python enhanced_video_tracker.py
```

### Kontroller

| Tuş | Fonksiyon |
|-----|-----------|
| MOUSE DRAG | Hedef nesne seçimi |
| SPACE | Video oynat/duraklat |
| R | Takibi sıfırla |
| M | Takip modları arasında geçiş |
| A | Analitik paneli aç/kapat |
| T | İz görselleştirme aç/kapat |
| P | Öngörü gösterimi aç/kapat |
| G | Güven grafiği aç/kapat |
| +/- | Oynatma hızını ayarla |
| 1,2,3,4 | Takip modu seç (Hassas/Dengeli/Agresif/Adaptif) |
| Z | Otomatik zoom aç/kapat |
| S | Akıllı kırpma aç/kapat |
| F | Performans modu değiştir |
| O | Adaptif performans aç/kapat |
| L | Hafif mod aç/kapat |
| X | Oklüzyon tespiti aç/kapat |
| ESC | Uygulamadan çık |

## Takip Modları

### Hassas Mod (Precision)
- **Kullanım:** Yüksek doğruluk gereken durumlar
- **Performans:** Orta hızlı
- **Özellik:** En yüksek doğruluk

### Dengeli Mod (Balanced)
- **Kullanım:** Genel amaçlı takip
- **Performans:** İyi denge
- **Özellik:** Hız/doğruluk dengesi

### Agresif Mod (Aggressive)
- **Kullanım:** Zor takip durumları
- **Performans:** Yavaş ama kararlı
- **Özellik:** Asla pes etmez

### Adaptif Mod (Adaptive)
- **Kullanım:** Otomatik optimizasyon
- **Performans:** Duruma göre ayarlanır
- **Özellik:** Yapay zeka destekli

## Performans Ayarları

### Otomatik Performans Optimizasyonu
Sistem otomatik olarak aşağıdaki parametreleri ayarlar:
- **Frame işleme süresi izleme**
- **Performans seviyesi otomatik ayarı**
- **Kaynak kullanımı optimizasyonu**
- **Adaptif algoritma seçimi**

### Hafif Mod
Düşük performanslı sistemler için:
- **Azaltılmış sabır seviyeleri**
- **Sıklık azaltılmış kalite analizi**
- **Basitleştirilmiş oklüzyon tespiti**
- **Optimize edilmiş arama algoritmaları**

## Konfigürasyon

### tracker_config.json
```json
{
  "max_lost_frames": 30,
  "confidence_threshold": 0.7,
  "search_radius": 100,
  "prediction_steps": 1.0,
  "lightweight_mode": true,
  "adaptive_performance": true
}
```

## Teknik Detaylar

### Takip Durum Makinesi
- **WAITING** - Hedef seçimi bekleniyor
- **INITIALIZING** - Takip sistemi başlatılıyor
- **TRACKING** - Aktif takip durumu
- **PREDICTING** - Hareket öngörüsü yapılıyor
- **OCCLUDED** - Oklüzyon durumu tespit edildi
- **SEARCHING** - Derin arama modu
- **RECOVERING** - Kurtarma modu
- **PAUSED** - Sistem duraklatıldı
- **ANALYZING** - Kalite analizi yapılıyor

### Algoritma Detayları

#### Hareket Öngörüsü
- **Pozisyon geçmişi analizi** - Son 50 frame
- **Hız hesaplama** - Son 30 frame
- **İvme analizi** - Son 20 frame
- **Kalite ağırlıklı öngörü** - Görüntü kalitesine göre ağırlıklandırma

#### Oklüzyon Tespiti
- **Güven düşüşü analizi** - Ani güven kaybı tespiti
- **Hareket tutarlılığı** - Hareket desenlerinin analizi
- **Çıkış pozisyonu tahmini** - Nesnenin tekrar görünebileceği yer

#### Şablon Eşleştirme
- **Çoklu ölçek desteği** - 0.8x - 1.3x arasında
- **Rotasyon toleransı** - ±15 derece
- **Özellik tabanlı eşleştirme** - SIFT/ORB fallback

## Kullanım Senaryoları

### Güvenlik ve Gözetim
- **Nesne izleme** - Şüpheli aktivite takibi
- **Çevre güvenliği** - Sınır ihlali tespiti
- **Trafik analizi** - Araç/yaya takibi

### Araştırma ve Geliştirme
- **Algoritma testi** - Takip yöntemlerini karşılaştırma
- **Performans analizi** - Sistem yeteneklerini değerlendirme
- **Veri toplama** - Takip metriklerini dışa aktarma

### Otonom Sistemler
- **Hedef takip** - Hareketli nesneleri takip etme
- **Navigasyon yardımı** - Yol planlama desteği
- **Stabilizasyon** - Hareket kompanzasyonu

## Sorun Giderme

### Sık Karşılaşılan Problemler

| Problem | Çözüm |
|---------|--------|
| Video açılmıyor | Dosya yolunu ve formatını kontrol edin |
| Düşük FPS | Hafif mod açın, çözünürlüğü azaltın |
| Takip başarısız | max_lost_frames artırın, CSRT deneyin |
| Bellek problemi | Diğer uygulamaları kapatın |
| Import hataları | pip install -r requirements.txt çalıştırın |

### Debug Modu
```bash
# Verbose loglama etkinleştir
python enhanced_video_tracker.py --verbose

# OpenCV kurulumunu kontrol et
python -c "import cv2; print(cv2.__version__)"
```

## API Referansı

### VideoTracker Sınıfı
```python
class VideoTracker:
    def __init__(self, video_path: str)
    def update_tracking(self, frame: np.ndarray) -> None
    def handle_controls(self, key: int) -> bool
    def draw_ui(self, frame: np.ndarray) -> np.ndarray
    def run(self) -> None
```

### MotionPredictor Sınıfı
```python
class MotionPredictor:
    def __init__(self)
    def update(self, bbox: Tuple, confidence: float, quality_score: float) -> None
    def predict_position(self, steps_ahead: float, context: Dict) -> Optional[Tuple]
```

### OcclusionDetector Sınıfı
```python
class OcclusionDetector:
    def __init__(self)
    def detect_occlusion(self, current_confidence: float, previous_confidence: float, 
                        motion_data: Dict, frame_count: int) -> bool
    def estimate_exit_position(self, motion_data: Dict, occlusion_duration: int) -> Optional[Tuple]
```

## Performans Metrikleri

### Sistem Performansı
- **İşleme Hızı:** 20-30 FPS (1920x1080)
- **Bellek Kullanımı:** 150-250 MB
- **CPU Kullanımı:** %15-30 (çok çekirdekli)
- **Doğruluk Oranı:** %85-95 (sahneye bağlı)

### Optimizasyon İpuçları
1. **Doğru modu seçin** - Hassas/Dengeli/Agresif/Adaptif
2. **Video kalitesini optimize edin** - Gereksiz yüksek çözünürlükten kaçının
3. **Sistem kaynaklarını yönetin** - Diğer uygulamaları kapatın
4. **Konfigürasyonu ayarlayın** - tracker_config.json ile fine-tuning

## Geliştirici Notları

### Proje Yapısı
```
chase/
├── enhanced_video_tracker.py    # Ana takip uygulaması
├── requirements.txt             # Python bağımlılıkları
├── tracker_config.json          # Konfigürasyon dosyası
├── setup.py                     # Kurulum scripti
├── test_tracker.py              # Test dosyaları
└── README.md                    # Bu dokümantasyon
```

### Geliştirmeye Katkı

1. **Depoyu fork edin**
2. **Feature branch oluşturun:** `git checkout -b feature/YeniOzellik`
3. **Değişiklikleri commit edin:** `git commit -m 'YeniOzellik eklendi'`
4. **Branch'i push edin:** `git push origin feature/YeniOzellik`
5. **Pull Request açın**

### Geliştirme Kuralları
- PEP 8 stil kılavuzunu takip edin
- Kapsamlı docstring'ler ekleyin
- Hata yakalama mekanizmaları dahil edin
- Unit testler yazın
- Dokümantasyonu güncelleyin

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.

## Destek

- **Sorunlar:** GitHub Issues
- **Tartışmalar:** GitHub Discussions
- **E-posta:** destek@example.com

---

**Bilgisayarlı görü topluluğu için geliştirildi**

*Güvenlik sistemleri, otonom araçlar ve araştırma projeleri için ideal* 