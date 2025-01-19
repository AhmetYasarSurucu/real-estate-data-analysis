import subprocess
import sys
import random

# Gerekli kütüphaneler
required_libraries = [
    "pandas",
    "numpy",
    "scikit-learn",
    "seaborn",
    "matplotlib",
    "openpyxl"
]

# Kütüphaneleri kontrol etme ve yükleme
for library in required_libraries:
    try:
        __import__(library)
    except ImportError:
        print(f"{library} kütüphanesi yüklü değil. Yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])
    else:
        print(f"{library} kütüphanesi zaten yüklü.")

print("Tüm kütüphaneler başarıyla kontrol edildi ve eksik olanlar yüklendi.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Veri setinizi yükleyin
data = pd.read_excel("cleaned_data2.xlsx")

# Hedef değişken ve özellikler
target_column = "Fiyat"
feature_columns = [
    "M2",
    "Oda_Sayisi_Encoded",
    "Evin_Bulundugu_Kat_Encoded",
    "Bolge_Encoded",
    "Mahalle_Encoded",
    "Bina_Yasi",
]

# Hedef değişken (y) ve bağımsız değişkenler (X)
X = data[feature_columns]
y = data[target_column]

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Search için hiperparametre aralığı
def random_search_hyperparameters():
    n_estimators = random.choice([50, 100, 200, 300,400])
    max_depth = random.choice([None, 10, 20, 30])
    min_samples_split = random.choice([2, 5, 10,20])
    min_samples_leaf = random.choice([1, 2, 4,8])
    return n_estimators, max_depth, min_samples_split, min_samples_leaf

# Random Search uygulama
best_r2 = -float('inf')
best_params = None

for i in range(30):  # Rastgele 10 kombinasyon dene
    n_estimators, max_depth, min_samples_split, min_samples_leaf = random_search_hyperparameters()

    # Model oluşturma
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Modeli eğitme
    model.fit(X_train, y_train)

    # Test seti üzerindeki tahminler
    y_test_pred = model.predict(X_test)

    # Test R² hesaplama
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Adım {i + 1}: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, Test R²={test_r2}")

    if test_r2 > best_r2:
        best_r2 = test_r2
        best_params = (n_estimators, max_depth, min_samples_split, min_samples_leaf)

# En iyi hiperparametrelerle modeli eğitme
n_estimators, max_depth, min_samples_split, min_samples_leaf = best_params
final_model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
final_model.fit(X_train, y_train)

# Eğitim seti üzerindeki tahminler
y_train_pred = final_model.predict(X_train)
# Test seti üzerindeki tahminler
y_test_pred = final_model.predict(X_test)

# Eğitim seti için başarı metrikleri
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test seti için başarı metrikleri
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Sonuçları ekrana yazdırma
print("En iyi Hiperparametreler:")
print(f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
print("\nEğitim Seti Başarı Metrikleri:")
print(f"Mean Squared Error (MSE): {train_mse}")
print(f"Mean Absolute Error (MAE): {train_mae}")
print(f"R² Score: {train_r2}\n")

print("Test Seti Başarı Metrikleri:")
print(f"Mean Squared Error (MSE): {test_mse}")
print(f"Mean Absolute Error (MAE): {test_mae}")
print(f"R² Score: {test_r2}\n")

# Performans Karşılaştırması
if abs(train_r2 - test_r2) < 0.1:
    print("Model, eğitim ve test setlerinde benzer performans gösteriyor. Overfitting düşük.")
elif train_r2 > test_r2:
    print("Model, eğitim setinde daha iyi performans gösteriyor. Overfitting riski olabilir.")
else:
    print("Model, test setinde daha iyi performans gösteriyor. Bu durum genelde nadirdir.")
