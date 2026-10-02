import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path

# --- AYARLAR ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / 'veriCekme+maskeleme'
INPUT_DIR = PIPELINE_DIR / 'outputs'
OUTPUT_DIR = PIPELINE_DIR / 'çıktı'
# İstenen bantlar sırasıyla. B04=Kırmızı, B03=Yeşil, B02=Mavi, B11=SWIR
# Bu liste değiştirilerek farklı bant kombinasyonları yapılabilir.
REQUIRED_BANDS = ['B04', 'B03', 'B02', 'B11']

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_matching_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith('.jp2') or f.endswith('.tif')]
    groups = {}

    for f in files:
        parts = f.split('_')
        if len(parts) < 3: continue
        
        # Dosya isimlendirmesine göre prefix (örn: T36TUK_20230101T000000)
        prefix = "_".join(parts[:2])
        band_type = parts[2] # B04, B03, etc.
        
        if prefix not in groups: groups[prefix] = {}
        
        # Eğer bu bant bizim aradığımız bantlardan biriyse listeye ekle
        if band_type in REQUIRED_BANDS:
            groups[prefix][band_type] = os.path.join(input_dir, f)
    
    # Sadece tüm gerekli bantlara sahip olan grupları filtrele
    valid_groups = {}
    for prefix, paths in groups.items():
        # Aranan tüm bantlar bu grupta var mı?
        if all(band in paths for band in REQUIRED_BANDS):
            valid_groups[prefix] = paths
            
    return valid_groups

def merge_and_standardize(prefix, paths):
    # Çıktı dosya adı
    output_path = os.path.join(OUTPUT_DIR, f"{prefix}_rgb_nir.tif")

    print(f"🛠️ İşleniyor: {prefix}...")

    try:
        bands_data = []
        
        # İlk bandı referans olarak al (Boyutlar ve projeksiyon için)
        # Genellikle 10m çözünürlüklü bantlardan birini (B02, B03, B04) referans almak iyidir.
        ref_band_name = REQUIRED_BANDS[0]
        ref_path = paths[ref_band_name]
        
        with rasterio.open(ref_path) as src_ref:
            meta = src_ref.meta.copy()
            ref_shape = (src_ref.height, src_ref.width)
            
            # Tüm bantları döngüyle oku
            for band_name in REQUIRED_BANDS:
                path = paths[band_name]
                with rasterio.open(path) as src:
                    # Referans boyutuna göre oku (B11 gibi 20m bantlar otomatik upsample edilir)
                    data = src.read(
                        1,
                        out_shape=ref_shape,
                        resampling=Resampling.bilinear
                    )
                    bands_data.append(data)

        # Verileri birleştir (Stack) -> (Band_Sayısı, Yükseklik, Genişlik)
        stack = np.stack(bands_data)

        # Meta verileri güncelle
        meta.update({
            'count': len(REQUIRED_BANDS),
            'driver': 'GTiff',
            # dtype'ı değiştirmiyoruz, orijinal veri neyse o kalıyor (muhtemelen uint16)
            # Sıkıştırma (compress) parametresi eklemiyoruz, ham haliyle kalsın.
        })
        
        # Dosyayı yaz
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(stack)
            
            # Bant isimlerini meta veriye işle
            for i, band_name in enumerate(REQUIRED_BANDS, start=1):
                dst.set_band_description(i, band_name)

        print(f"  ✅ [OK] Kaydedildi: {output_path}")

    except Exception as e:
        print(f"  ❌ [HATA] {prefix}: {e}")

def main():
    print("Dosyalar taranıyor...")
    groups = find_matching_files(INPUT_DIR)
    
    if not groups:
        print("❌ Eşleşen dosya grubu bulunamadı.")
        print(f"📂 Aranan klasör: {INPUT_DIR}")
        print(f"🎯 Aranan bantlar: {REQUIRED_BANDS}")
        return

    print(f"Toplam {len(groups)} grup bulundu, birleştirme başlıyor...")
    
    for prefix, paths in groups.items():
        merge_and_standardize(prefix, paths)
    
    print("\n🏁 İşlem tamam.")

if __name__ == "__main__":
    main()
