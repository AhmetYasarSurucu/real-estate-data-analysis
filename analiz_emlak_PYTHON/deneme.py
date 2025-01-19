import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Excel dosyasını yükleyin
data_path = "cleaned_data2.xlsx"  # Dosya yolunu belirtin
df = pd.read_excel(data_path)

# Fiyatları ve M2 başına fiyatı hesaplama
df['Fiyat_M'] = df['Fiyat'] / 1_000_000
df['Fiyat_M2'] = df['Fiyat'] / df['M2']

# 1. Bölgelere Göre M2 Başına Fiyat Raporu (Detaylı ve Okunaklı)
bolge_m2_fiyat = df.groupby('Bolge')['Fiyat_M2'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(18, 10))
sns.barplot(data=bolge_m2_fiyat, x='Fiyat_M2', y='Bolge', palette='coolwarm', edgecolor='black')
plt.title('Bölgelere Göre Ortalama M2 Başına Fiyat (TL)', fontsize=22, fontweight='bold', color='#333333')
plt.xlabel('M2 Başına Fiyat (TL)', fontsize=18, color='#555555')
plt.ylabel('Bölge', fontsize=18, color='#555555')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Yatırım İçin En Uygun Bölge ve Mahalle (Detaylı ve Renkli)
bolge_mahalle_fiyat = df.groupby(['Bolge', 'Mahalle'])['Fiyat_M2'].mean().reset_index()
en_uygun_yatirim = bolge_mahalle_fiyat.sort_values(by='Fiyat_M2').head(10)

plt.figure(figsize=(18, 10))
sns.barplot(data=en_uygun_yatirim, x='Fiyat_M2', y='Mahalle', hue='Bolge', dodge=False, palette='Spectral', edgecolor='black')
plt.title('Yatırım İçin En Uygun Bölge ve Mahalle (M2 Başına Fiyat)', fontsize=22, fontweight='bold', color='#333333')
plt.xlabel('M2 Başına Fiyat (TL)', fontsize=18, color='#555555')
plt.ylabel('Mahalle', fontsize=18, color='#555555')
plt.legend(title='Bölge', fontsize=14, title_fontsize=16, loc='upper right', frameon=True, shadow=True, borderpad=1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.5)
plt.tight_layout()
plt.show()

# 3. Yatırım Uygunluğu Haritası (Geliştirilmiş ve Net)
bolge_mahalle_pivot = bolge_mahalle_fiyat.pivot_table(index='Mahalle', columns='Bolge', values='Fiyat_M2')
plt.figure(figsize=(20, 12))
sns.heatmap(bolge_mahalle_pivot, annot=True, fmt=".0f", cmap="RdYlGn", linewidths=0.001, cbar_kws={'label': 'M2 Başına Fiyat (TL)'})
plt.title('Yatırım İçin M2 Başına Fiyat Isı Haritası', fontsize=22, fontweight='bold', color='#333333')
plt.xlabel('Bölge', fontsize=5, color='#555555')
plt.ylabel('Mahalle', fontsize=5, color='#555555')
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
