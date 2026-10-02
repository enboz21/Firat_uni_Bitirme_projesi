import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np

# Kullanıcıdan gelen ham veri (Raw Logs)
raw_logs = """
1/100      | 0.46828    | 0.63093    | ✅ Kaydedildi
2/100      | 0.34317    | 0.57082    | ✅ Kaydedildi
3/100      | 0.31493    | 0.43575    | ✅ Kaydedildi
4/100      | 0.29098    | 0.32345    | ✅ Kaydedildi
5/100      | 0.27184    | 0.27426    | ✅ Kaydedildi
6/100      | 0.25108    | 0.22206    | ✅ Kaydedildi
7/100      | 0.24505    | 0.23915    | ⚠️ (1/20)
8/100      | 0.23239    | 0.21447    | ✅ Kaydedildi
9/100      | 0.22788    | 0.19616    | ✅ Kaydedildi
10/100      | 0.21616    | 0.22146    | ⚠️ (1/20)
11/100      | 0.21868    | 0.21206    | ⚠️ (2/20)
12/100      | 0.20661    | 0.22328    | ⚠️ (3/20)
13/100      | 0.19788    | 0.17862    | ✅ Kaydedildi
14/100      | 0.19603    | 0.18722    | ⚠️ (1/20)
15/100      | 0.18440    | 0.19591    | ⚠️ (2/20)
16/100      | 0.18790    | 0.20657    | ⚠️ (3/20)
17/100      | 0.17456    | 0.17777    | ✅ Kaydedildi
18/100      | 0.17403    | 0.15148    | ✅ Kaydedildi
19/100      | 0.16803    | 0.18128    | ⚠️ (1/20)
20/100      | 0.16120    | 0.15498    | ⚠️ (2/20)
21/100      | 0.15140    | 0.13514    | ✅ Kaydedildi
22/100      | 0.15267    | 0.17674    | ⚠️ (1/20)
23/100      | 0.14977    | 0.15533    | ⚠️ (2/20)
24/100      | 0.13916    | 0.12478    | ✅ Kaydedildi
25/100      | 0.13558    | 0.11466    | ✅ Kaydedildi
26/100      | 0.13443    | 0.13354    | ⚠️ (1/20)
27/100      | 0.13091    | 0.15703    | ⚠️ (2/20)
28/100      | 0.12851    | 0.14406    | ⚠️ (3/20)
29/100      | 0.12281    | 0.12522    | ⚠️ (4/20)
30/100      | 0.11988    | 0.11948    | ⚠️ (5/20)
31/100      | 0.11920    | 0.13970    | ⚠️ (6/20)
32/100      | 0.11514    | 0.10246    | ✅ Kaydedildi
33/100      | 0.11554    | 0.11074    | ⚠️ (1/20)
34/100      | 0.11195    | 0.11541    | ⚠️ (2/20)
35/100      | 0.10692    | 0.13934    | ⚠️ (3/20)
36/100      | 0.10515    | 0.10730    | ⚠️ (4/20)
37/100      | 0.10075    | 0.08909    | ✅ Kaydedildi
38/100      | 0.10224    | 0.10189    | ⚠️ (1/20)
39/100      | 0.11213    | 0.16902    | ⚠️ (2/20)
40/100      | 0.09674    | 0.10471    | ⚠️ (3/20)
41/100      | 0.09329    | 0.13304    | ⚠️ (4/20)
42/100      | 0.08632    | 0.09968    | ⚠️ (5/20)
43/100      | 0.08455    | 0.12260    | ⚠️ (6/20)
44/100      | 0.09715    | 0.12447    | ⚠️ (7/20)
45/100      | 0.08407    | 0.08329    | ✅ Kaydedildi
46/100      | 0.08410    | 0.08633    | ⚠️ (1/20)
47/100      | 0.08348    | 0.10107    | ⚠️ (2/20)
48/100      | 0.07815    | 0.09421    | ⚠️ (3/20)
49/100      | 0.08344    | 0.08931    | ⚠️ (4/20)
50/100      | 0.07518    | 0.10664    | ⚠️ (5/20)
51/100      | 0.07661    | 0.11069    | ⚠️ (6/20)
52/100      | 0.07234    | 0.07446    | ✅ Kaydedildi
53/100      | 0.07237    | 0.06633    | ✅ Kaydedildi
54/100      | 0.07059    | 0.07846    | ⚠️ (1/20)
55/100      | 0.06840    | 0.09104    | ⚠️ (2/20)
56/100      | 0.06425    | 0.10402    | ⚠️ (3/20)
57/100      | 0.06151    | 0.06482    | ✅ Kaydedildi
58/100      | 0.06406    | 0.04856    | ✅ Kaydedildi
59/100      | 0.07274    | 0.10877    | ⚠️ (1/20)
60/100      | 0.06643    | 0.11301    | ⚠️ (2/20)
61/100      | 0.06625    | 0.11211    | ⚠️ (3/20)
62/100      | 0.07298    | 0.19913    | ⚠️ (4/20)
63/100      | 0.06493    | 0.07937    | ⚠️ (5/20)
64/100      | 0.05783    | 0.06689    | ⚠️ (6/20)
65/100      | 0.06087    | 0.09353    | ⚠️ (7/20)
66/100      | 0.06323    | 0.09133    | ⚠️ (8/20)
67/100      | 0.06025    | 0.06073    | ⚠️ (9/20)
68/100      | 0.05739    | 0.06540    | ⚠️ (10/20)
69/100      | 0.06474    | 0.07430    | ⚠️ (11/20)
70/100      | 0.05441    | 0.06355    | ⚠️ (12/20)
71/100      | 0.04648    | 0.05880    | ⚠️ (13/20)
72/100      | 0.05383    | 0.06941    | ⚠️ (14/20)
73/100      | 0.05894    | 0.08266    | ⚠️ (15/20)
74/100      | 0.05422    | 0.06856    | ⚠️ (16/20)
75/100      | 0.06008    | 0.07450    | ⚠️ (17/20)
76/100      | 0.04888    | 0.05492    | ⚠️ (18/20)
77/100      | 0.05187    | 0.10344    | ⚠️ (19/20)
78/100      | 0.04670    | 0.05316    | ⚠️ (20/20)
"""

# Veriyi Ayrıştırma (Parsing)
epochs = []
train_loss = []
val_loss = []

for line in raw_logs.strip().split('\n'):
    parts = line.split('|')
    if len(parts) >= 3:
        try:
            # Epoch formatı "1/100" veya "  1/100" şeklinde olabilir
            current_epoch = int(parts[0].strip().split('/')[0])
            t_loss = float(parts[1].strip())
            v_loss = float(parts[2].strip())
            
            epochs.append(current_epoch)
            train_loss.append(t_loss)
            val_loss.append(v_loss)
        except ValueError:
            continue

# Pandas DataFrame
df_data = pd.DataFrame({
    'Epoch': epochs,
    'Train_Loss': train_loss,
    'Val_Loss': val_loss
})

# İstatistikler
best_val_idx = df_data['Val_Loss'].idxmin()
best_epoch = df_data.loc[best_val_idx, 'Epoch']
best_val_loss = df_data.loc[best_val_idx, 'Val_Loss']
final_epoch = df_data.iloc[-1]['Epoch']
final_train_loss = df_data.iloc[-1]['Train_Loss']
final_val_loss = df_data.iloc[-1]['Val_Loss']

# Grafik Çizimi
plt.figure(figsize=(12, 7))
plt.plot(df_data['Epoch'], df_data['Train_Loss'], label='Eğitim Kaybı (Train Loss)', color='blue', linewidth=2)
plt.plot(df_data['Epoch'], df_data['Val_Loss'], label='Doğrulama Kaybı (Validation Loss)', color='orange', linewidth=2, linestyle='--')

# En iyi modeli işaretle
plt.scatter(best_epoch, best_val_loss, color='red', zorder=5, label=f'En İyi Model (Epoch {best_epoch})')
plt.annotate(f'{best_val_loss:.4f}', xy=(best_epoch, best_val_loss), 
             xytext=(best_epoch - 5, best_val_loss + 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, ha='center')

# Bitiş (Early Stopping)
plt.annotate('Early Stopping', xy=(final_epoch, final_val_loss), 
             xytext=(final_epoch - 20, final_val_loss + 0.25),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, ha='center')

plt.title('Eğitim ve Doğrulama Kaybı (Gerçek Veri)', fontsize=14)
plt.xlabel('Epoch (Dönem)', fontsize=12)
plt.ylabel('Kayıp (Loss)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Kaydet
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_path = os.path.join(project_root, 'kayip_grafigi_gercek.png')
plt.savefig(output_path, dpi=300)
print(f"Grafik kaydedildi: {output_path}")

# Tablo Oluşturma
print("\n--- Model Performans Özeti ---")
print(f"{'Metrik':<25} | {'Başlangıç (Epoch 1)':<20} | {'En İyi (Best)':<20} | {'Bitiş (Epoch {final_epoch})':<20}")
print("-" * 90)

start_train = df_data.iloc[0]['Train_Loss']
start_val = df_data.iloc[0]['Val_Loss']
best_val = best_val_loss  # Validation üzerinden best seçtik

print(f"{'Eğitim Kaybı':<25} | {start_train:.5f}              | {'-':<20} | {final_train_loss:.5f}")
print(f"{'Doğrulama Kaybı':<25} | {start_val:.5f}              | {best_val:.5f} (Ep. {best_epoch}) | {final_val_loss:.5f}")

print("\n\n--- İnceleme Notları ---")
print("1. Eğitim Kaybı istikrarlı bir şekilde düşüyor, model veriyi öğreniyor.")
print(f"2. Doğrulama Kaybı en düşük seviyesini {best_epoch}. epochta ({best_val_loss}) görmüş.")
print("3. Eğitim 78. epochta 'Early Stopping' ile sonlanmış, yani modelin iyileşmesi durmuş.")
print("4. Overfitting (ezberleme) belirtisi yok, doğrulama kaybı eğitimi takip ediyor.")
