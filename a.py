import torch

print("--- GPU KONTROLÜ ---")

# 1. Cihaz kontrolü
print(f"Hedef Cihaz: {CONFIG['DEVICE']}")

# 2. Model GPU'da mı?
# Modelin ilk parametresinin nerede olduğuna bakıyoruz
try:
    print(f"Model Nerede: {next(model.parameters()).device}")
except:
    print("Model henüz tanımlanmadı.")

# 3. GPU Bellek Kullanımı
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB cinsinden
    cached = torch.cuda.memory_reserved(0) / 1024**3
    print(f"GPU Bellek Kullanımı (VRAM): {allocated:.2f} GB")
    print(f"GPU Ayrılan Bellek: {cached:.2f} GB")
else:
    print("GPU algılanmadı!")