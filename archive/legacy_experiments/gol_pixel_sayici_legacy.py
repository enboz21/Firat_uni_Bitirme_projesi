import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
import math
import sys
# from tqdm import tqdm

# Configuration
# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

CONFIG = {
    # Test etmek istediğin herhangi bir _rgb_nir.tif dosyası
    "TEST_IMAGE_PATH": os.path.join(script_dir, "burdur_01_2025_merged.tif"),
    "MODEL_PATH": os.path.join(script_dir, "gol_bulucu_unet_sentinel.pth"),  # Eğittiğimiz model dosyası

    "IMG_HEIGHT": 256,
    "IMG_WIDTH": 256,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# UNET Class Definition
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

def predict_single_image(image_path):
    # print(f"Model yükleniyor: {CONFIG['MODEL_PATH']}")
    
    # 1. Modeli Hazırla
    model = UNET(in_channels=4, out_channels=1).to(CONFIG["DEVICE"])
    try:
        model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"]))
    except FileNotFoundError:
        print("HATA: Model dosyası bulunamadı!")
        return
    model.eval()

    if not os.path.exists(image_path):
        print(f"HATA: Resim bulunamadı: {image_path}")
        return

    # print("Görüntü okunuyor...")
    with rasterio.open(image_path) as src:
        # Band sayısını kontrol et
        band_count = src.count
        
        if band_count >= 4:
            full_image = src.read([1, 2, 3, 4]).transpose(1, 2, 0)
        elif band_count == 3:
            img_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
            h, w, c = img_rgb.shape
            nir_channel = np.zeros((h, w, 1), dtype=img_rgb.dtype)
            full_image = np.concatenate((img_rgb, nir_channel), axis=2)
            # print("UYARI: 3 band tespit edildi. 4. kanal (NIR) 0 olarak eklendi.")
        elif band_count == 1:
            img_gray = src.read(1)
            img_expanded = np.expand_dims(img_gray, axis=2)
            full_image = np.repeat(img_expanded, 4, axis=2)
            # print("UYARI: 1 band tespit edildi. Kanal 4 kez çoğaltıldı.")
        else:
            img_bands = src.read(range(1, band_count + 1)).transpose(1, 2, 0)
            needed = 4 - band_count
            last_band = img_bands[:, :, -1:]
            extras = np.repeat(last_band, needed, axis=2)
            full_image = np.concatenate((img_bands, extras), axis=2)
            # print(f"UYARI: {band_count} band tespit edildi. 4 kanala tamamlandı.")

        original_height, original_width = full_image.shape[:2]
        profile = src.profile 
        
    # print(f"Orijinal Boyutlar: {original_width}x{original_height}")

    # 2. Preprocessing (Resize ve Normalize)
    # Numpy -> Tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(full_image).permute(2, 0, 1).float()
    
    # Normalizasyon
    if img_tensor.max() > 255.0:
        img_tensor = img_tensor / 10000.0
    else:
        img_tensor = img_tensor / 255.0
        
    # Batch boyutu ekle: (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Resize (Modelin beklediği boyuta: 256x256)
    img_resized = torch.nn.functional.interpolate(
        img_tensor, 
        size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]), 
        mode='bilinear', 
        align_corners=False
    )
    
    img_resized = img_resized.to(CONFIG["DEVICE"])

    # 3. Tahmin (Tek Seferde)
    with torch.no_grad():
        prediction = torch.sigmoid(model(img_resized))
        
    # 4. Post-processing (Orijinal boyuta geri döndür)
    prediction_resized = torch.nn.functional.interpolate(
        prediction, 
        size=(original_height, original_width), 
        mode='bilinear', 
        align_corners=False
    )
    
    pred_mask = prediction_resized.squeeze().cpu().numpy()
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # 5. Sonuçları İşle ve Kaydet
    white_pixel_count = np.sum(binary_mask == 1)
    print(f"{white_pixel_count}")
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}_mask.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    save_mask = binary_mask * 255
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(save_mask, 1)
        
    # print(f"Maske kaydedildi: {output_path}")

    # --- GÖRSELLEŞTİRME ---
    print("Görselleştirme yapılıyor...")
    
    # Görüntüyü görselleştirme için hazırla (RGB kanalları)
    # full_image numpy array olarak zaten elimizde var
    display_image = full_image[:, :, :3].copy().astype(np.float32)
    
    if display_image.max() > 255.0:
        display_image /= 10000.0  # 16-bit
    else:
        display_image /= 255.0    # 8-bit
        
    # Parlaklığı artır (uydu görüntüleri karanlık olabilir)
    display_image = np.clip(display_image * 3.5, 0, 1)
    
    plt.figure(figsize=(16, 8))
    
    # 1. Orijinal Görüntü
    plt.subplot(1, 2, 1)
    plt.title(f"Orijinal Görüntü\n{original_width}x{original_height}")
    plt.imshow(display_image)
    plt.axis('off')
    
    # 2. Maske
    plt.subplot(1, 2, 2)
    plt.title(f"Tespit Edilen Maske (Beyaz Piksel: {white_pixel_count})")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Görselleştirme tamamlandı.")

if __name__ == "__main__":
    image_path = CONFIG["TEST_IMAGE_PATH"]
    
    # Eğer komut satırından parametre gelirse onu kullan
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    print(f"İşlenen dosya: {image_path}")
    predict_single_image(image_path)
