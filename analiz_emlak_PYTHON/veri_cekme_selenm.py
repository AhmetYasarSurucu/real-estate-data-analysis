from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random

# Selenium için tarayıcı ayarları
options = Options()
options.add_argument("--headless")  # Tarayıcıyı görünmez modda çalıştır
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("--disable-extensions")
options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36")

# Veri listeleri
fiyat, m_2a, oda_sayisi, bina_yasi, bulundugu_kat, mahalle = [], [], [], [], [], []

# Web scraping işlemi
for page in range(1, 252):  # Sayfa sayısını düzenleme
    try:
        # ChromeDriver'ı WebDriver Manager ile başlatma
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # Sayfayı açma
        url = f"https://www.hepsiemlak.com/antalya-satilik/daire?counties=antalya-aksu,dosemealti,kepez,konyaalti,muratpasa&page={page}"
        driver.get(url)

        # Sayfanın yüklenmesini bekleme
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "list-view-price"))
        )
        time.sleep(random.uniform(14, 55))  # Rastgele bekleme süresi

        # Verileri çekme
        fiyat_elements = driver.find_elements(By.CLASS_NAME, "list-view-price")
        m2_elements = driver.find_elements(By.CLASS_NAME, "list-view-size")
        oda_elements = driver.find_elements(By.CLASS_NAME, "houseRoomCount")
        bina_yasi_elements = driver.find_elements(By.CLASS_NAME, "buildingAge")
        kat_elements = driver.find_elements(By.CLASS_NAME, "floortype")
        mahalle_elements = driver.find_elements(By.CLASS_NAME, "list-view-location")

        # Verileri listeye ekle
        for i in range(max(len(fiyat_elements), len(m2_elements), len(oda_elements),
                           len(bina_yasi_elements), len(kat_elements), len(mahalle_elements))):
            try:
                fiyat.append(int(fiyat_elements[i].text.strip().replace('.', '').replace('TL', '')))
            except (IndexError, ValueError):
                fiyat.append(None)

            try:
                m_2a.append(int(m2_elements[i].text.strip().replace(' m²', '')))
            except (IndexError, ValueError):
                m_2a.append(None)

            try:
                oda_sayisi.append(oda_elements[i].text.strip())
            except IndexError:
                oda_sayisi.append(None)

            try:
                bina_yasi.append((''.join(filter(str.isdigit, bina_yasi_elements[i].text.strip()))))
            except IndexError:
                bina_yasi.append(None)

            try:
                bulundugu_kat.append(kat_elements[i].text.strip())
            except IndexError:
                bulundugu_kat.append(None)

            try:
                mahalle.append(mahalle_elements[i].text.strip())
            except IndexError:
                mahalle.append(None)

    except Exception as e:
        print(f"Page {page} failed: {e}")
    finally:
        driver.quit()
        # Sayfa geçişleri arasında rastgele bekleme süresi
        time.sleep(random.uniform(15, 30))

# Listeleri eşitle
max_len = max(len(fiyat), len(m_2a), len(oda_sayisi), len(bina_yasi), len(bulundugu_kat), len(mahalle))
fiyat.extend([None] * (max_len - len(fiyat)))
m_2a.extend([None] * (max_len - len(m_2a)))
oda_sayisi.extend([None] * (max_len - len(oda_sayisi)))
bina_yasi.extend([None] * (max_len - len(bina_yasi)))
bulundugu_kat.extend([None] * (max_len - len(bulundugu_kat)))
mahalle.extend([None] * (max_len - len(mahalle)))

# Veriyi DataFrame'e dönüştür
data = pd.DataFrame({
    'Fiyat': fiyat,
    'M2': m_2a,
    'Oda_Sayisi': oda_sayisi,
    'Bina_Yasi': bina_yasi,
    'Evin_Bulundugu_Kat': bulundugu_kat,
    'Konum': mahalle
})

# Veriyi Excel'e kaydet
data.to_excel("antalya_muratpasa_selenium_updated1_252.xlsx", index=False)
print("Scraping completed and saved to Excel.")
