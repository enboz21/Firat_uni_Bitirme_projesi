import os
import rasterio
import numpy as np
import cv2
import json
from pathlib import Path

# --- AYARLAR ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / 'veriCekme+maskeleme'
inputs_dir = PIPELINE_DIR / 'outputs'
out_dir = PIPELINE_DIR / 'output_geojson'    # Çıktıların kaydedileceği yer (GeoJSON)
K1="B03"
K2="B11"
THRESHOLD = 0.0          # Su eşik değeri (Gerekirse artır/azalt)
MIN_LAKE_AREA = 1000     # Küçük gürültüleri temizle
ISLAND_AREA_THRESHOLD = 500 
# ----------------

os.makedirs(out_dir, exist_ok=True)

# 1. DOSYALARI TARA VE EŞLEŞTİR
# Klasör var mı kontrol et, yoksa hata vermesin, boş liste dönsün
if os.path.exists(inputs_dir):
    all_files = [f for f in os.listdir(inputs_dir) if f.endswith('.jp2') or f.endswith('.tif')]
else:
    print(f"UYARI: Girdi klasörü bulunamadı: {inputs_dir}")
    all_files = []

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
    
    if K1 in f:
        file_groups[prefix]['K1'] = os.path.join(inputs_dir, f)
    elif K2 in f:
        file_groups[prefix]['K2'] = os.path.join(inputs_dir, f)

# Sadece hem K1 hem K2'si olan çiftleri filtrele
valid_groups = {k: v for k, v in file_groups.items() if 'K1' in v and 'K2' in v}

print(f"Islenecek goruntu cifti sayisi: {len(valid_groups)}")

# 2. GÖRÜNTÜLERİ İŞLE
for prefix, paths in valid_groups.items():
    path_green = paths['K1']
    path_nir = paths['K2']
    
    print(f"İşleniyor: {prefix}...")

    try:
        with rasterio.open(path_green) as src_grn, rasterio.open(path_nir) as src_nir:
            green = src_grn.read(1).astype(float)
            
            # Boyut eşitleme
            if src_grn.shape != src_nir.shape:
                nir = src_nir.read(1, out_shape=(src_grn.height, src_grn.width), resampling=rasterio.enums.Resampling.bilinear).astype(float)
            else:
                nir = src_nir.read(1).astype(float)
            
            # Koordinat dönüşümü için transform matrisi
            transform = src_grn.transform
            crs = src_grn.crs

        np.seterr(divide='ignore', invalid='ignore')
        
        # --- NDWI HESABI (McFeeters) ---
        # Formül: (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir)

        # 1. Ham Maske
        binary_mask = (ndwi > THRESHOLD).astype(np.uint8) * 255

        # 2. Hiyerarşik Kontur Analizi
        # RETR_CCOMP: 2 seviyeli hiyerarşi (Dış sınırlar ve Delikler)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []

        if hierarchy is not None:
            hierarchy = hierarchy[0] # [Next, Previous, First_Child, Parent]
            
            for i, cnt in enumerate(contours):
                # Parent -1 ise Göl sınırıdır (Dış halka)
                if hierarchy[i][3] == -1:
                    lake_area = cv2.contourArea(cnt)
                    if lake_area >= MIN_LAKE_AREA:
                        
                        # Dış halka koordinatlarını dönüştür (Pixel -> Map)
                        exterior_ring = []
                        for point in cnt:
                            px, py = point[0]
                            # Rasterio transform: (col, row) -> (x, y)
                            mx, my = transform * (px, py)
                            exterior_ring.append([mx, my])
                        
                        # Poligonu kapat (GeoJSON standardı gereği ilk ve son nokta aynı olmalı)
                        if exterior_ring[0] != exterior_ring[-1]:
                            exterior_ring.append(exterior_ring[0])
                        
                        poly_coords = [exterior_ring]
                        
                        # Çocukları (Adaları/Delikleri) bul
                        child_idx = hierarchy[i][2]
                        while child_idx != -1:
                            child_cnt = contours[child_idx]
                            hole_area = cv2.contourArea(child_cnt)
                            
                            if hole_area >= ISLAND_AREA_THRESHOLD:
                                hole_ring = []
                                for point in child_cnt:
                                    px, py = point[0]
                                    mx, my = transform * (px, py)
                                    hole_ring.append([mx, my])
                                
                                # Poligonu kapat
                                if hole_ring[0] != hole_ring[-1]:
                                    hole_ring.append(hole_ring[0])
                                    
                                poly_coords.append(hole_ring)
                            
                            # Bir sonraki kardeşe geç
                            child_idx = hierarchy[child_idx][0]
                        
                        # Feature oluştur
                        feature = {
                            "type": "Feature",
                            "properties": { "id": str(len(features) + 1) },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": poly_coords
                            }
                        }
                        features.append(feature)

        # --- KAYIT İŞLEMİ (GeoJSON) ---
        out_name = f"{prefix}_mask.geojson"
        out_path = os.path.join(out_dir, out_name)

        # CRS bilgisini oluştur
        crs_urn = "urn:ogc:def:crs:OGC:1.3:CRS84" # Varsayılan
        if crs:
            try:
                epsg = crs.to_epsg()
                if epsg:
                    crs_urn = f"urn:ogc:def:crs:EPSG::{epsg}"
            except:
                pass

        geojson_data = {
            "type": "FeatureCollection",
            "name": f"{prefix}_mask",
            "crs": { "type": "name", "properties": { "name": crs_urn } },
            "features": features
        }
        
        with open(out_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
            
        print(f"  [OK] Tamamlandı: {out_name}")

    except Exception as e:
        print(f"  [HATA] {prefix} islenirken hata olustu: {e}")

print("Tum islemler bitti.")
