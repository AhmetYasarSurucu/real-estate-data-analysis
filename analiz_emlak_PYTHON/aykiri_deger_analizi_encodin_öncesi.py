import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükleme
file_path = 'antalya_bina_yasi_doldurulmus.xlsx'  # Dosya yolunuzu buraya girin
df = pd.read_excel(file_path)

# Sayısal ve kategorik sütunlar
numerical_columns = ['Fiyat', 'M2', 'Bina_Yasi']
categorical_columns = ['Oda_Sayisi', 'Evin_Bulundugu_Kat', 'Bolge', 'Mahalle']

# Seaborn ayarları
sns.set(style="whitegrid")  # Beyaz arka planlı bir tema

# Fonksiyon: Sayısal sütunlar için box plot ve aykırı değer analizi
def plot_numerical_column(df, col):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=col, color="#6a5d5d")  # Mat bir renk seçildi

    # Logaritmik ölçek seçeneği (isteğe bağlı)
    if df[col].max() > 1000:  # Büyük değerler için logaritmik ölçek
        plt.xscale('log')
        plt.xlabel(f"{col} (Log Scale)", fontsize=14)
    else:
        plt.xlabel(col, fontsize=14)

    # Başlık ve diğer düzenlemeler
    plt.title(f'Aykırı Değer Analizi - {col}', fontsize=16)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.show()

    # İstatistiksel özet ve aykırı değerlerin hesaplanması
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nİstatistiksel Özet: {col}")
    print(df[col].describe())
    print(f"Aykırı Değer Sayısı: {len(outliers)}")
    if not outliers.empty:
        print(f"Aykırı Değerlerin İlk 5 Gözlemi:\n{outliers.head()}")

# Fonksiyon: Kategorik sütunlar için 2D pasta grafiği
def plot_2d_pie_chart(df, col, top_n=None):
    if top_n:  # İlk N değeri göstermek için
        top_values = df[col].value_counts().head(top_n)
    else:
        top_values = df[col].value_counts()

    total = top_values.sum()
    percentages = (top_values / total) * 100
    labels = percentages.index.tolist()
    sizes = percentages.values.tolist()

    # %2'den küçük değerleri "Diğer" olarak birleştir
    other_size = sum([size for size, label in zip(sizes, labels) if size < 2])
    sizes = [size if size >= 2 else 0 for size in sizes]
    labels = [label if size >= 2 else "" for label, size in zip(labels, sizes)]
    if other_size > 0:
        sizes.append(other_size)
        labels.append("Diğer")

    sizes = [size for size in sizes if size > 0]
    labels = [label for label in labels if label]

    # En büyük 3 dilim için hafif dışarı belirteç koyma
    explode = [0.03 if i < 3 else 0.01 for i in range(len(sizes))]  # Daha hafif bir dışarı çıkma

    plt.figure(figsize=(10, 10))

    colors = sns.color_palette("Set2", len(labels))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, startangle=90, autopct='%1.1f%%', colors=colors, explode=explode, textprops={'color': "black"})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)

    plt.title(f"{col}", fontsize=16)
    plt.show()

    # Frekans bilgisi
    print(f"\nFrekans Dağılımı: {col}")
    print(top_values)

# Sayısal sütunlar için analiz ve grafikler
print("==== Sayısal Değişkenler Analizi ====")
for col in numerical_columns:
    plot_numerical_column(df, col)

# Mahalle sütunu için en çok geçen 20 mahalle
top_20_mahalle = df['Mahalle'].value_counts().head(20).index
df['Mahalle'] = df['Mahalle'].where(df['Mahalle'].isin(top_20_mahalle))

# Kategorik sütunlar için analiz ve grafikler
print("\n==== Kategorik Değişkenler Analizi ====")
for col in categorical_columns:
    top_n = 20 if col == 'Mahalle' else None  # Mahalle için ilk 20'yi filtrele
    plot_2d_pie_chart(df, col, top_n)

# Yoğunluk grafikleri oluşturma
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=col, fill=True, color='blue', alpha=0.5)
    plt.title(f"{col} için Yoğunluk Grafiği", fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel("Yoğunluk", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()
