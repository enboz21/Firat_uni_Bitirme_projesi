import torch
import sys
import io

# Windows terminalinde emoji karakterlerin düzgün görünmesi için
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def gpu_kontrol():
    print("PyTorch Sürümü:", torch.__version__)
    
    if torch.cuda.is_available():
        print("\n✅ CUDA (GPU) Kullanılabilir!")
        device_count = torch.cuda.device_count()
        print(f"Bulunan GPU Sayısı: {device_count}")
        
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Şu anki aktif cihazı göster
        current_device = torch.cuda.current_device()
        print(f"\nAktif Cihaz ID: {current_device}")
        print(f"Aktif Cihaz Adı: {torch.cuda.get_device_name(current_device)}")
        
        # Basit bir tensör işlemi ile test et
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            print("\n✅ Test tensörü başarıyla GPU belleğine yüklendi.")
        except Exception as e:
            print(f"\n❌ Tensör GPU'ya yüklenirken hata oluştu: {e}")
            
    else:
        print("\n❌ CUDA (GPU) Bulunamadı. İşlemci (CPU) kullanılıyor.")

if __name__ == "__main__":
    gpu_kontrol()
