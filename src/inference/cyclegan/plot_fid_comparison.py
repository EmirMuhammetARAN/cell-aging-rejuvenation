import os
import matplotlib.pyplot as plt

# Veriler
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
aging_fid = [58.66, 62.88, 57.33, 63.72, 66.56, 68.08, 59.98, 89.22, 68.40, 56.76, 66.65, 57.68, 64.07, 62.08, 60.79, 53.70, 60.33, 272.05, 65.53, 70.54]
reju_fid = [69.54, 61.59, 87.94, 82.79, 66.45, 66.63, 67.41, 63.64, 65.65, 64.96, 208.81, 65.52, 59.88, 68.49, 79.30, 63.52, 143.67, 120.00, 73.90, 66.44]

ldm_aging_fid = 47.30
ldm_reju_fid = 53.49

# Klasör hazırlığı
out_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'generated', 'cyclegan_v2_sweep')
os.makedirs(out_dir, exist_ok=True)

# Gözlemi kolaylaştırmak için Y eksenini 100'de kesmek mantıklı olabilir çünkü 272 gibi outlierlar 
# grafiğin alt kısmını görünmez yapar. Ama outlier'ları (GAN istikrarsızlığını) göstermek için
# raw bırakıyoruz. İsteyen y eksenini plt.ylim(40, 100) ile sınırlayabilir.

plt.style.use('seaborn-v0_8-whitegrid')

# 1. AGING FID GRAFİĞİ
plt.figure(figsize=(10, 6))
plt.plot(epochs, aging_fid, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='CycleGAN FID')
plt.axhline(y=ldm_aging_fid, color='#d62728', linestyle='--', linewidth=2.5, label=f'LDM Baseline ({ldm_aging_fid})')

plt.title('Aging Task: FID vs Epoch (CycleGAN vs LDM)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('FID Score (Lower is Better)', fontsize=12)
plt.xticks(epochs, rotation=45)
plt.legend(fontsize=12)
plt.tight_layout()

# Grafiği kaydet
aging_out = os.path.join(out_dir, 'aging_fid_comparison.png')
plt.savefig(aging_out, dpi=300)
print(f"Saved Aging plot to {aging_out}")
plt.close()

# 2. REJUVENATION FID GRAFİĞİ
plt.figure(figsize=(10, 6))
plt.plot(epochs, reju_fid, marker='s', linestyle='-', color='#ff7f0e', linewidth=2, label='CycleGAN FID')
plt.axhline(y=ldm_reju_fid, color='#d62728', linestyle='--', linewidth=2.5, label=f'LDM Baseline ({ldm_reju_fid})')

plt.title('Rejuvenation Task: FID vs Epoch (CycleGAN vs LDM)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('FID Score (Lower is Better)', fontsize=12)
plt.xticks(epochs, rotation=45)
plt.legend(fontsize=12)
plt.tight_layout()

# Grafiği kaydet
reju_out = os.path.join(out_dir, 'reju_fid_comparison.png')
plt.savefig(reju_out, dpi=300)
print(f"Saved Rejuvenation plot to {reju_out}")
plt.close()

print("Bitti! Grafikler results/generated/cyclegan_v2_sweep klasörüne kaydedildi.")
