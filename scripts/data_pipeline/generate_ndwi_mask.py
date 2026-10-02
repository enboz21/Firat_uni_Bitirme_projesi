import os
import rasterio
import numpy as np
import cv2
from pathlib import Path

# --- AYARLAR ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / 'veriCekme+maskeleme'
inputs_dir = PIPELINE_DIR / 'outputs'
out_dir = PIPELINE_DIR / 'output'    # Çıktıların kaydedileceği yer
K1="B03"
K2="B11"
THRESHOLD = 0.0          # Su eşik değeri (Gerekirse artır/azalt)
MIN_LAKE_AREA = 1000     # Küçük gürültüleri temizle
ISLAND_AREA_THRESHOLD = 500 
# ----------------

os.makedirs(out_dir, exist_ok=True)

# 1. DOSYALARI TARA VE EŞLEŞTİR
all_files = [f for f in os.listdir(inputs_dir) if f.endswith('.jp2') or f.endswith('.tif')]

file_groups = {}

print("Dosyalar taranıyor ve eşleştiriliyor...")

for f in all_files:
    parts = f.split('_')
    
    if len(parts) < 3:
        continue
    
    # Prefix: T35SPA_20240816T084559 gibi
    prefix = "_".join(parts[:2])
    
    if prefix not in file_groups:
        file_groups[prefix] = {}
    
    # --- DEĞİŞİKLİK BURADA: B11 YERİNE B08 ARIYORUZ ---
    if K1 in f:
        file_groups[prefix]['K1'] = os.path.join(inputs_dir, f)
    elif K2 in f:  # Eski kodda burası B11 idi
        file_groups[prefix]['K2'] = os.path.join(inputs_dir, f)

# Sadece hem B03 hem B08'i olan çiftleri filtrele
valid_groups = {k: v for k, v in file_groups.items() if 'K1' in v and 'K2' in v}

print(f"Islenecek goruntu cifti sayisi: {len(valid_groups)}")

# 2. GÖRÜNTÜLERİ İŞLE
for prefix, paths in valid_groups.items():
    path_green = paths['K1']
    path_nir = paths['K2'] # B11 yerine NIR (B08) kullanıyoruz
    
    print(f"İşleniyor: {prefix}...")

    try:
        with rasterio.open(path_green) as src_grn, rasterio.open(path_nir) as src_nir:
            green = src_grn.read(1).astype(float)
            
            # Boyut eşitleme (B08 bazen 10m bazen farklı olabilir, garanti olsun)
            if src_grn.shape != src_nir.shape:
                nir = src_nir.read(1, out_shape=(src_grn.height, src_grn.width), resampling=rasterio.enums.Resampling.bilinear).astype(float)
            else:
                nir = src_nir.read(1).astype(float)
            
            meta = src_grn.meta.copy()

        np.seterr(divide='ignore', invalid='ignore')
        
        # --- NDWI HESABI (McFeeters) ---
        # Formül: (Green - NIR) / (Green + NIR)
        # B11 olsaydı MNDWI olurdu, B08 ile NDWI olur. İkisi de su bulur.
        ndwi = (green - nir) / (green + nir)

        # 1. Ham Maske
        binary_mask = (ndwi > THRESHOLD).astype(np.uint8) * 255

        # 2. Hiyerarşik Kontur Analizi (Temizlik)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros_like(binary_mask)

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                # Parent -1 ise Göl sınırıdır
                if hierarchy[0][i][3] == -1:
                    lake_area = cv2.contourArea(cnt)
                    if lake_area >= MIN_LAKE_AREA:
                        cv2.drawContours(final_mask, [cnt], -1, 255, -1)
                
                # Parent -1 değilse adadır
                else:
                    hole_area = cv2.contourArea(cnt)
                    if hole_area >= ISLAND_AREA_THRESHOLD:
                        cv2.drawContours(final_mask, [cnt], -1, 0, -1)

        # --- KAYIT İŞLEMİ ---
        out_name = f"{prefix}_mask.tif"
        out_path = os.path.join(out_dir, out_name)

        meta.update(dtype=rasterio.uint8, count=1, driver='GTiff', compress='none') 
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(final_mask, 1)
            
        print(f"  [OK] Tamamlandı: {out_name}")

    except Exception as e:
        print(f"  [HATA] {prefix} islenirken hata olustu: {e}")

print("Tum islemler bitti.")
