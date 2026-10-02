import os
import torch
import torch.nn as nn
import numpy as np
import rasterio
from tqdm import tqdm
import warnings
from pathlib import Path

# Uyarıları gizle (özellikle rasterio ve torch)
warnings.filterwarnings("ignore")

# --- AYARLAR ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "DATA_DIR": PROJECT_ROOT / "dataset",  # Veri seti klasörü
    "MODEL_PATH": PROJECT_ROOT / "gol_bulucu_unet_sentinel.pth", # Model dosyasının yolu
    "IMG_HEIGHT": 512,  # Modelin eğitildiği boyut
    "IMG_WIDTH": 512,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMAGE_SUFFIX": "_rgb_nir.tif",
    "MASK_SUFFIX": "_mask.tif"
}

# --- UNET MODEL TANIMI ---
# (makine öğrenmesi.ipynb ve predict_water_mask.py'den alınmıştır)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
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

# --- METRİK HESAPLAMA FONKSİYONLARI ---
def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    """
    Tek bir görüntü için IoU, Dice ve Accuracy hesaplar.
    Girdi maskeleri (H, W) boyutunda numpy array ve {0, 1} değerlerinde olmalıdır.
    """
    # Düzleştirme (Flatten)
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - intersection
    
    # IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice
    dice = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)
    
    # Accuracy
    # (Doğru tahmin edilen 0'lar + Doğru tahmin edilen 1'ler) / Toplam piksel
    accuracy = (pred_flat == true_flat).sum() / len(true_flat)
    
    return iou, dice, accuracy

# --- GÖRÜNTÜ İŞLEME VE TAHMİN ---
def predict_and_evaluate():
    print(f"Model yükleniyor: {CONFIG['MODEL_PATH']}")
    
    # Modeli başlat
    model = UNET(in_channels=4, out_channels=1).to(CONFIG["DEVICE"])
    
    # Ağırlıkları yükle
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print(f"HATA: Model dosyası bulunamadı: {CONFIG['MODEL_PATH']}")
        return
        
    try:
        model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"]))
    except Exception as e:
        print(f"HATA: Model yüklenirken sorun oluştu: {e}")
        return
        
    model.eval()
    
    # Dosyaları listele
    all_files = os.listdir(CONFIG["DATA_DIR"])
    image_files = [f for f in all_files if f.endswith(CONFIG["IMAGE_SUFFIX"])]
    
    if not image_files:
        print("HATA: Hiç görüntü dosyası bulunamadı!")
        return

    print(f"Toplam {len(image_files)} görüntü üzerinde değerlendirme yapılacak...")
    
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    count = 0
    
    for img_file in tqdm(image_files, desc="Değerlendiriliyor"):
        img_path = os.path.join(CONFIG["DATA_DIR"], img_file)
        mask_file = img_file.replace(CONFIG["IMAGE_SUFFIX"], CONFIG["MASK_SUFFIX"])
        mask_path = os.path.join(CONFIG["DATA_DIR"], mask_file)
        
        # Maske dosyası yoksa atla
        if not os.path.exists(mask_path):
            continue
            
        # 1. Görüntüyü Oku ve Hazırla
        try:
            with rasterio.open(img_path) as src:
                # 4 kanal okuma (R, G, B, NIR)
                if src.count >= 4:
                    image_np = src.read([1, 2, 3, 4]).transpose(1, 2, 0)
                else:
                    # Eksik kanal yönetimi basitçe (burada detaylı ele almıyoruz, genelde 4 kanal beklenir)
                    continue
                    
            with rasterio.open(mask_path) as src_mask:
                true_mask = src_mask.read(1)
                # Maskeyi 0 ve 1 yap
                true_mask = (true_mask > 0).astype(np.uint8)
                
        except Exception as e:
            print(f"Dosya okuma hatası ({img_file}): {e}")
            continue

        original_h, original_w = image_np.shape[:2]
        
        # Preprocessing (Model formatına çevirme)
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        # Normalizasyon
        if img_tensor.max() > 255.0:
            img_tensor /= 10000.0
        else:
            img_tensor /= 255.0
            
        img_tensor = img_tensor.unsqueeze(0).to(CONFIG["DEVICE"])
        
        # Resize (Modele uygun boyuta)
        img_resized = torch.nn.functional.interpolate(
            img_tensor, 
            size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 2. Tahmin
        with torch.no_grad():
            pred_logit = model(img_resized)
            pred_prob = torch.sigmoid(pred_logit)
            
        # Orijinal boyuta geri döndür
        pred_prob_resized = torch.nn.functional.interpolate(
            pred_prob,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )
        
        pred_mask_np = pred_prob_resized.squeeze().cpu().numpy()
        pred_mask_binary = (pred_mask_np > 0.5).astype(np.uint8)
        
        # 3. Metrik Hesapla
        iou, dice, acc = calculate_metrics(pred_mask_binary, true_mask)
        
        total_iou += iou
        total_dice += dice
        total_acc += acc
        count += 1

    if count == 0:
        print("Değerlendirilecek uygun veri bulunamadı.")
        return

    # Ortalamalar
    avg_iou = total_iou / count
    avg_dice = total_dice / count
    avg_acc = total_acc / count
    
    # --- SONUÇLARI GÖSTER ---
    print("\n" + "="*40)
    print(f" {count} Görüntü Üzerinde Ortalama Sonuçlar")
    print("="*40)
    # Tablo benzeri çıktı
    print(f"{'Metrik':<30} | {'Değer':<10}")
    print("-" * 43)
    print(f"{'IoU (Intersection over Union)':<30} | %{avg_iou*100:.2f}")
    print("-" * 43)
    print(f"{'Dice Score':<30} | %{avg_dice*100:.2f}")
    print("-" * 43)
    print(f"{'Accuracy':<30} | %{avg_acc*100:.2f}")
    print("="*40)

if __name__ == "__main__":
    predict_and_evaluate()
