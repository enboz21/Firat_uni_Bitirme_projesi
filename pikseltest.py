import numpy as np
from PIL import Image
import rasterio # GeoTIFF özelliklerini (koordinat vs) okumak istersen

def count_pixels(image_path):
    try:
        # Görüntüyü yükle
        with Image.open(image_path) as img:
            # NumPy dizisine çevir
            data = np.array(img)
            
            # Görüntüdeki benzersiz değerleri kontrol et (Örn: [0, 255] mi yoksa [0, 1] mi?)
            unique_values = np.unique(data)
            print(f"Görüntüdeki benzersiz piksel değerleri: {unique_values}")
            
            # Beyaz piksel varsayımı (Genelde 255'tir, binary ise 1 olabilir)
            if 255 in unique_values:
                white_count = np.sum(data == 255)
                target_value = 255
            elif 1 in unique_values:
                white_count = np.sum(data == 1)
                target_value = 1
            else:
                # Farklı bir etiketleme varsa en büyük değeri baz alalım
                target_value = np.max(data)
                white_count = np.sum(data == target_value)

            print(f"Hedef Değer ({target_value}) Piksel Sayısı: {white_count}")
            return white_count

    except Exception as e:
        print(f"Hata oluştu: {e}")

# Dosya yolunu buraya gir
count_pixels("outputs/burdur_01_2025_merged_mask.tif")