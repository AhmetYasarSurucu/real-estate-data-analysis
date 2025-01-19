import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setinizi yükleyin
data = pd.read_excel("antalya_bina_yasi_doldurulmus.xlsx")

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
#"Bina_Yasi",
# Sadece bu kolonları alarak Z-skor hesaplaması yapıyoruz
data_to_standardize = data[columns_to_analyze]

# Z-skorları hesaplama
z_scores = np.abs((data_to_standardize - data_to_standardize.mean()) / data_to_standardize.std())

# Aykırı değer eşik değeri (genelde 3.5 kullanılır)
threshold = 3.5

# Eşik değerin altında kalan satırları filtreleme
rows_within_threshold = (z_scores < threshold).all(axis=1)
filtered_data = data[rows_within_threshold]

# Korelasyon matrisini hesapla (sadece sayısal değişkenler için)
correlation_matrix = filtered_data[columns_to_analyze].corr()

# Korelasyon matrisini ekrana yazdır
print("Korelasyon Matrisi:")
print(correlation_matrix)

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Korelasyon Matrisi")
plt.show()

###################################################
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Veriye sabit bir sütun ekleyerek VIF hesaplama
X = add_constant(data[[
    "M2",
    "Oda_Sayisi_Encoded",
    "Evin_Bulundugu_Kat_Encoded",
    "Bolge_Encoded",
    "Mahalle_Encoded",

]])
vif_data = pd.DataFrame()
vif_data["Değişken"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
