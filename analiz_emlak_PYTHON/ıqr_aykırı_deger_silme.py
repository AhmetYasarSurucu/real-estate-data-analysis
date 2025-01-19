import pandas as pd

# Aykırı değerleri IQR yöntemine göre temizlemek için bir fonksiyon
def remove_outliers_iqr(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)  # 1. çeyrek
        Q3 = df[column].quantile(0.75)  # 3. çeyrek
        IQR = Q3 - Q1  # Interkuartil aralığı
        lower_bound = Q1 - 1.5 * IQR  # Alt sınır
        upper_bound = Q3 + 1.5 * IQR  # Üst sınır
        # Aykırı değerleri filtrele
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Excel dosyasını yükle
file_path = 'cleaned_data6.xlsx'
data = pd.read_excel(file_path)

# Analiz yapılacak sütunlar
columns_to_analyze = [
    "Fiyat",
    "M2",
    "Oda_Sayisi_Encoded",
    "Evin_Bulundugu_Kat_Encoded",
    "Bolge_Encoded",
    "Mahalle_Encoded",
    "Bina_Yasi",
]

# Aykırı değerleri kaldır
cleaned_data = remove_outliers_iqr(data, columns_to_analyze)

# Temizlenmiş veri çerçevesini kaydet
cleaned_file_path = 'cleaned_data7.xlsx'
cleaned_data.to_excel(cleaned_file_path, index=False)

print(f"Aykırı değerler kaldırıldı ve dosya kaydedildi: {cleaned_file_path}")
