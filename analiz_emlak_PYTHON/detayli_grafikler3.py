import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veriyi yükleme
file_path = 'cleaned_data2.xlsx'  # Excel dosyasının yolu
df = pd.read_excel(file_path)

# Sayısal sütunlar
numerical_columns = ['Fiyat', 'M2', 'Bina_Yasi']

# Histogram ve Yoğunluk Grafiği Oluşturma
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    # Histogram
    sns.histplot(data=df, x=col, kde=True, bins=30, color="blue", alpha=0.6)

    # Ortalama Çizgisi
    mean_value = df[col].mean()
    plt.axvline(mean_value, color="orange", linestyle="--", linewidth=2, label=f'Ortalama {col}')

    # Grafik Ayarları
    plt.title(f"{col} Dağılımı", fontsize=16)
    plt.xlabel(f"{col}", fontsize=14)
    plt.ylabel("Frekans", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
