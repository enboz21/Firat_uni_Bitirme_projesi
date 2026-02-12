import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

# KONTROL EDİLECEK DOSYA YOLU (Senin oluşturduğun dosya)
TEST_IMAGE_PATH = r"veriCekme+maskeleme\çıktı\T35SPA_20240816T084559_rgb_nir.tif"

def inspect_channels():
    if not os.path.exists(TEST_IMAGE_PATH):
        print("Dosya bulunamadı!")
        return

    print(f"Dosya inceleniyor: {TEST_IMAGE_PATH}")
    
    with rasterio.open(TEST_IMAGE_PATH) as src:
        # Kaç kanal var bakalım?
        print(f"Toplam Kanal Sayısı: {src.count}")
        
        # İlk 4 kanalı okuyalım
        try:
            b1 = src.read(1)
            b2 = src.read(2)
            b3 = src.read(3)
            b4 = src.read(4)
        except IndexError:
            print(f"HATA: Dosyada 4 kanal yok! Sadece {src.count} kanal var.")
            return

        # İstatistikler
        print(f"Kanal 1 (R?) -> Min: {b1.min()}, Max: {b1.max()}, Ortalama: {b1.mean():.2f}")
        print(f"Kanal 2 (G?) -> Min: {b2.min()}, Max: {b2.max()}, Ortalama: {b2.mean():.2f}")
        print(f"Kanal 3 (B?) -> Min: {b3.min()}, Max: {b3.max()}, Ortalama: {b3.mean():.2f}")
        print(f"Kanal 4 (NIR?) -> Min: {b4.min()}, Max: {b4.max()}, Ortalama: {b4.mean():.2f}")

        # Görselleştirme
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.title("Kanal 1 (Kırmızı olması lazım)")
        plt.imshow(b1, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Kanal 2 (Yeşil olması lazım)")
        plt.imshow(b2, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Kanal 3 (Mavi olması lazım)")
        plt.imshow(b3, cmap="gray")
        plt.axis("off")

        # EN ÖNEMLİSİ BU
        plt.subplot(1, 4, 4)
        plt.title("Kanal 4 (NIR - Su SİYAH olmalı)")
        plt.imshow(b4, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    inspect_channels()