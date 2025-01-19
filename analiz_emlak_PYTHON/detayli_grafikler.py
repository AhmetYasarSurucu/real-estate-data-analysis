import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Excel dosyasını yükleyin
data_path = "cleaned_data2.xlsx"  # Dosya yolunu belirtin
df = pd.read_excel(data_path)

# Fiyatları milyon biriminde göstermek için dönüştürme
df['Fiyat_M'] = df['Fiyat'] / 1_000_000

# Renk paleti oluşturma (otomatik olarak eşleşen bir renk paleti kullanılır)
oda_palette = sns.color_palette("Set2", n_colors=df['Oda_Sayisi'].nunique())

# 1. Fiyat Dağılımı
plt.figure(figsize=(14, 8))
sns.histplot(df['Fiyat_M'], kde=True, bins=50, color='#0072B2', alpha=0.8)
plt.title('Fiyat Dağılımı (Milyon TL)', fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('Fiyat (Milyon TL)', fontsize=14, color='#555555')
plt.ylabel('Frekans', fontsize=14, color='#555555')
plt.axvline(df['Fiyat_M'].mean(), color='#D55E00', linestyle='--', linewidth=2, label='Ortalama Fiyat')
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(fontsize=12, loc='upper right')
plt.show()

# 2. Metrekare ve Fiyat Arasındaki İlişki
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='M2', y='Fiyat_M', hue='Oda_Sayisi', palette=oda_palette, s=120, edgecolor='black', alpha=0.9)
plt.title('Metrekare ve Fiyat Arasındaki İlişki (Milyon TL)', fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('Metrekare (M2)', fontsize=14, color='#555555')
plt.ylabel('Fiyat (Milyon TL)', fontsize=14, color='#555555')
plt.legend(title='Oda Sayısı', fontsize=12, title_fontsize=14, loc='upper left', frameon=True, shadow=True, borderpad=1)
plt.grid(linestyle='--', linewidth=0.7, alpha=0.5)
plt.show()

# 3. Oda Sayısına Göre Fiyat Dağılımı
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Oda_Sayisi', y='Fiyat_M', palette=oda_palette, showfliers=False)
sns.stripplot(data=df, x='Oda_Sayisi', y='Fiyat_M', color='black', size=6, alpha=0.7, jitter=True)
plt.title('Oda Sayısına Göre Fiyat Dağılımı (Milyon TL)', fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('Oda Sayısı', fontsize=14, color='#555555')
plt.ylabel('Fiyat (Milyon TL)', fontsize=14, color='#555555')
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
plt.show()

# 4. Bina Yaşı ve Fiyat Arasındaki İlişki
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='Bina_Yasi', y='Fiyat_M', hue='Bolge', palette='tab20', s=100, edgecolor='black', alpha=0.85)
plt.title('Bina Yaşı ve Fiyat Arasındaki İlişki (Milyon TL)', fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('Bina Yaşı', fontsize=14, color='#555555')
plt.ylabel('Fiyat (Milyon TL)', fontsize=14, color='#555555')
plt.legend(title='Bölge', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True, borderpad=1)
plt.grid(linestyle='--', linewidth=0.7, alpha=0.5)
plt.show()
