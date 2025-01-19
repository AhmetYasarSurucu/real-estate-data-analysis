import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri dosyasını yükle
file_path = 'filtered_standardized_data.xlsx'

# Excel dosyasını yükle
data = pd.read_excel(file_path)

# Gerekli sütunların kontrolü
required_columns = ['Fiyat', 'M2', 'Bina_Yasi', 'Oda_Sayisi', 'Oda_Sayisi_Encoded', 'Bolge']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Eksik sütun: {col}")

# Oda_Sayisi sütunu için eşleştirme
oda_sayisi_mapping = data[['Oda_Sayisi_Encoded', 'Oda_Sayisi']].drop_duplicates().set_index('Oda_Sayisi_Encoded')['Oda_Sayisi'].to_dict()
data['Oda_Sayisi'] = data['Oda_Sayisi_Encoded'].map(oda_sayisi_mapping)

# Violin grafikleri için tema ayarları
sns.set_theme(style="whitegrid", palette="muted")

# Her sütun için violin grafiği oluşturma
numerical_columns = ['Fiyat', 'M2', 'Bina_Yasi', 'Oda_Sayisi_Encoded']

for column in numerical_columns:
    plt.figure(figsize=(12, 8))
    try:
        sns.violinplot(data=data, x='Bolge', y=column, palette="muted")

        # Özel y ekseni için kontrol (örnek: Fiyat veya M2 sütunu)
        if column == 'Fiyat':
            max_y = data[column].max()
            step = 500000
            y_ticks = list(range(0, int(max_y + step), step))
            y_labels = [f"{int(y/1000)}K" if y < 1000000 else f"{y/1000000:.1f}M" for y in y_ticks]
            plt.yticks(y_ticks, y_labels)
        elif column == 'M2':
            max_y = data[column].max()
            step = 50
            y_ticks = list(range(0, int(max_y + step), step))
            plt.yticks(y_ticks)
        elif column == 'Oda_Sayisi_Encoded':
            y_ticks = sorted(data['Oda_Sayisi_Encoded'].unique())
            y_labels = [oda_sayisi_mapping.get(tick, tick) for tick in y_ticks]
            plt.yticks(y_ticks, y_labels)

        # Özet istatistiklerin hesaplanması
        min_val = data[column].min()
        max_val = data[column].max()
        mean_val = data[column].mean()
        median_val = data[column].median()

        # Özet istatistiklerin grafiğe eklenmesi
        stats_text = f"Min: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Grafik başlık ve etiketleri
        plt.title(f"{column} için Violin Grafiği", fontsize=14)
        plt.xlabel('Bolge', fontsize=12)
        plt.ylabel(column, fontsize=12)

        # Grafik gösterimi
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"{column} için grafik oluşturulurken bir hata oluştu: {e}")
        raise
