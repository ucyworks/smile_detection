# Gülümseme Algılama Sistemi

Bu proje, YOLOv8 ve MediaPipe kullanarak kameradan canlı olarak gülümseme algılayan bir sistem oluşturur. Sistem yüzleri tespit eder, çerçeve içine alır ve kişi güldüğünde gülümseme yüzdesini gösterir.

## Kurulum

1. Gereksinimleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

2. YOLOv8 yüz tespit modelini indirin (uygulama ilk çalıştırmada otomatik olarak indirecektir):
   ```
   # Alternatif olarak manuel indirmek için:
   gdown https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt
   ```

## Kullanım

Jupyter notebook'u açın ve hücreleri sırasıyla çalıştırın:

