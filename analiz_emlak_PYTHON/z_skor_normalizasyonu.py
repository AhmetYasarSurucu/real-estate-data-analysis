import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Veri setinizi yükleyin
# Örnek: data = pd.read_excel("veri_dosyaniz.xlsx")
data = pd.read_excel("cleaned_data6.xlsx")

# İncelemek istediğiniz kolonları seçin
columns_to_analyze = [
    "Fiyat",
    "M2",
    "Oda_Sayisi_Encoded",
    "Evin_Bulundugu_Kat_Encoded",
    "Bolge_Encoded",
    "Mahalle_Encoded",
    "Bina_Yasi",
]

# Sadece bu kolonları alarak Z-skor hesaplaması yapıyoruz
data_to_standardize = data[columns_to_analyze]

# Z-skorları hesaplama
z_scores = np.abs((data_to_standardize - data_to_standardize.mean()) / data_to_standardize.std())

# Aykırı değer eşik değeri (genelde 3 kullanılır)
threshold = 3

# Eşik değerin altında kalan satırları filtreleme
rows_within_threshold = (z_scores < threshold).all(axis=1)
filtered_data = data[rows_within_threshold]

# Standartlaştırma işlemi (filtered_data üzerinde)
scaler = StandardScaler()
standardized_columns = scaler.fit_transform(filtered_data[columns_to_analyze])
standardized_data = pd.DataFrame(standardized_columns, columns=columns_to_analyze)

# Orijinal veri setinin diğer kolonlarını ekleme
other_columns = data.drop(columns=columns_to_analyze).loc[rows_within_threshold]
final_data = pd.concat([standardized_data, other_columns.reset_index(drop=True)], axis=1)

# Sonuçları bir dosyaya kaydetmek isterseniz
final_data.to_excel("filtered_standardized_data.xlsx", index=False)

# Filtrelenmiş ve standartlaştırılmış veriyi kontrol etmek için
print(final_data.head())
