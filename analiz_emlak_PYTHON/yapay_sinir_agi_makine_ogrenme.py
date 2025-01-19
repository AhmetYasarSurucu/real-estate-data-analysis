import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import random

# Veri yükleme
# Veri setinizi yükleyin (örnek dosya adı: "filtered_standardized_data.xlsx")
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
X = data[feature_columns].values
y = data[target_column].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch için tensörlere dönüştürme
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Yapay sinir ağı modeli tanımı
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hiperparametre aralığı
hidden_sizes = [16, 32, 64]
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [50, 100, 200]

# Random Search ile hiperparametre optimizasyonu
best_r2 = -float("inf")
best_params = None
best_model = None

for _ in range(10):  # Rastgele 10 kombinasyon dene
    hidden_size = random.choice(hidden_sizes)
    learning_rate = random.choice(learning_rates)
    epochs = random.choice(epochs_list)

    # Modeli oluşturma
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)

    # Kayıp fonksiyonu ve optimizasyon
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Eğitim
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Test seti üzerinde değerlendirme
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_r2 = r2_score(y_test.numpy(), y_test_pred.numpy())

    print(f"Hidden Size={hidden_size}, Learning Rate={learning_rate}, Epochs={epochs}, Test R²={test_r2}")

    if test_r2 > best_r2:
        best_r2 = test_r2
        best_params = (hidden_size, learning_rate, epochs)
        best_model = model

# En iyi modeli değerlendirme
hidden_size, learning_rate, epochs = best_params
print("En iyi Hiperparametreler:")
print(f"Hidden Size={hidden_size}, Learning Rate={learning_rate}, Epochs={epochs}")

# Eğitim ve test sonuçlarını hesaplama
best_model.eval()
with torch.no_grad():
    y_train_pred = best_model(X_train)
    y_test_pred = best_model(X_test)

# Eğitim seti metrikleri
train_mse = mean_squared_error(y_train.numpy(), y_train_pred.numpy())
train_mae = mean_absolute_error(y_train.numpy(), y_train_pred.numpy())
train_r2 = r2_score(y_train.numpy(), y_train_pred.numpy())

# Test seti metrikleri
test_mse = mean_squared_error(y_test.numpy(), y_test_pred.numpy())
test_mae = mean_absolute_error(y_test.numpy(), y_test_pred.numpy())
test_r2 = r2_score(y_test.numpy(), y_test_pred.numpy())

print("\nEğitim Seti Başarı Metrikleri:")
print(f"Mean Squared Error (MSE): {train_mse}")
print(f"Mean Absolute Error (MAE): {train_mae}")
print(f"R² Score: {train_r2}")

print("\nTest Seti Başarı Metrikleri:")
print(f"Mean Squared Error (MSE): {test_mse}")
print(f"Mean Absolute Error (MAE): {test_mae}")
print(f"R² Score: {test_r2}")

# Performans Karşılaştırması
if abs(train_r2 - test_r2) < 0.1:
    print("Model, eğitim ve test setlerinde benzer performans gösteriyor. Overfitting düşük.")
elif train_r2 > test_r2:
    print("Model, eğitim setinde daha iyi performans gösteriyor. Overfitting riski olabilir.")
else:
    print("Model, test setinde daha iyi performans gösteriyor. Bu durum genelde nadirdir.")
