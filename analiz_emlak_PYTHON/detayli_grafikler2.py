import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Excel dosyasını yükleyin
data_path = "cleaned_data2.xlsx"  # Dosya yolunu belirtin
df = pd.read_excel(data_path)

# Fiyatları milyon biriminde göstermek için dönüştürme
df['Fiyat_M'] = df['Fiyat'] / 1_000_000

# M2 başına fiyat hesaplama
df['Fiyat_M2'] = df['Fiyat'] / df['M2']

# 1. Bölgelere Göre M2 Başına Fiyat Raporu
bolge_m2_fiyat = df.groupby('Bolge')['Fiyat_M2'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 8))
bolge_m2_fiyat.plot(kind='bar', color=sns.color_palette('Blues', len(bolge_m2_fiyat)))
plt.title('Bölgelere Göre Ortalama M2 Başına Fiyat (TL)', fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('Bölge', fontsize=14, color='#555555')
plt.ylabel('M2 Başına Fiyat (TL)', fontsize=14, color='#555555')
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
plt.show()

# 2. Yatırım için En Uygun Bölge ve Mahalle
# Yatırım uygunluğu: M2 başına fiyatı düşük olan bölgeler ve mahalleler
bolge_mahalle_fiyat = df.groupby(['Bolge', 'Mahalle'])['Fiyat_M2'].mean().reset_index()
en_uygun_yatirim = bolge_mahalle_fiyat.sort_values(by='Fiyat_M2').head(10)

plt.figure(figsize=(14, 8))
sns.barplot(data=en_uygun_yatirim, x='Fiyat_M2', y='Mahalle', hue='Bolge', dodge=False, palette='Set2')
plt.title('Yatırım İçin En Uygun Bölge ve Mahalle (M2 Başına Fiyat)', fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('M2 Başına Fiyat (TL)', fontsize=14, color='#555555')
plt.ylabel('Mahalle', fontsize=14, color='#555555')
plt.legend(title='Bölge', fontsize=12, title_fontsize=14, loc='lower right', frameon=True, shadow=True, borderpad=1)
plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.5)
plt.show()
