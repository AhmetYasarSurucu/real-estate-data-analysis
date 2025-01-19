import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Veri setini yükleme
file_path = "encoded_data.xlsx"  # Dosya yolunu güncelleyin
data = pd.read_excel(file_path)

# Kullanılacak sütunların seçilmesi
selected_columns = ["Bina_Yasi", "Fiyat", "M2", "Oda_Sayisi_Encoded", "Evin_Bulundugu_Kat_Encoded", "Bolge_Encoded", "Mahalle_Encoded"]
data = data[selected_columns]

# Eksik olmayan değerlerle eğitim ve test setlerini oluşturma
non_missing_data = data.dropna(subset=["Bina_Yasi"])
X = non_missing_data.drop(columns=["Bina_Yasi"]).values
y = non_missing_data["Bina_Yasi"].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch veri yapısına dönüştürme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Yapay Sinir Ağı modeli tanımlama
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Hiperparametreler için rastgele arama
import random

def random_search_hyperparameters():
    hidden_size = random.choice([16, 32, 64, 128, 256, 512])
    learning_rate = random.choice([0.001, 0.01, 0.1])
    num_epochs = random.choice([50, 100, 200, 400, 800])
    return hidden_size, learning_rate, num_epochs

best_r2 = -float('inf')
best_params = None

for i in range(200):  # Rastgele 350 kombinasyon dene
    hidden_size, learning_rate, num_epochs = random_search_hyperparameters()

    # Model, kayıp fonksiyonu ve optimizasyon
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Eğitim
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Test R^2 hesaplama
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).view(-1).numpy()
        test_predictions_rounded = np.round(test_predictions)  # Tahminleri yuvarla
        r2_test = 1 - (np.sum((y_test - test_predictions_rounded) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"Adım {i + 1}: Hidden Size: {hidden_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Test R^2: {r2_test}")

    if r2_test > best_r2:
        best_r2 = r2_test
        best_params = (hidden_size, learning_rate, num_epochs)

# En iyi hiperparametrelerle modeli eğitme
hidden_size, learning_rate, num_epochs = best_params
final_model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    final_model.train()
    optimizer.zero_grad()
    predictions = final_model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:  # Her 10 epoch'ta bir R^2 yazdır
        final_model.eval()
        with torch.no_grad():
            train_predictions = final_model(X_train_tensor).view(-1).numpy()
            test_predictions = final_model(X_test_tensor).view(-1).numpy()
            train_predictions_rounded = np.round(train_predictions)
            test_predictions_rounded = np.round(test_predictions)
            r2_train = 1 - (np.sum((y_train - train_predictions_rounded) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
            r2_test = 1 - (np.sum((y_test - test_predictions_rounded) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            print(f"Epoch {epoch + 1}: Train R^2: {r2_train}, Test R^2: {r2_test}")

# Eksik Bina_Yasi değerlerini doldurma
final_model.eval()
missing_data = data[data["Bina_Yasi"].isna()]
if not missing_data.empty:
    missing_X = scaler.transform(missing_data.drop(columns=["Bina_Yasi"]).values)
    missing_X_tensor = torch.tensor(missing_X, dtype=torch.float32)
    with torch.no_grad():
        missing_predictions = final_model(missing_X_tensor).view(-1).numpy()
        missing_predictions_rounded = np.round(missing_predictions)  # Tahminleri yuvarla
    data.loc[data["Bina_Yasi"].isna(), "Bina_Yasi"] = missing_predictions_rounded

# Güncellenmiş veri setini Excel'e kaydetme
output_file = "antalya_bina_yasi_doldurulmus.xlsx"
data.to_excel(output_file, index=False)
print(f"Eksik değerler dolduruldu ve veri seti şu dosyaya kaydedildi: {output_file}")
