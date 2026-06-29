import os
import requests
import zipfile
from io import BytesIO

# ==========================================
# KONFIGURACJA
# ==========================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "NEARUSDT", "LTCUSDT", "BCHUSDT", "DOGEUSDT", "SHIBUSDT", "LINKUSDT"]
YEARS = ["2023", "2024", "2025"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

INTERVAL = "1m"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DOWNLOAD_DIR = "dane_do_PCA"

# ==========================================

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
print(f"Rozpoczynam pobieranie danych do folderu: {DOWNLOAD_DIR}\n")

for symbol in SYMBOLS:
    for year in YEARS:
        for month in MONTHS:
            # Nazwy plików
            base_name = f"{symbol}-{INTERVAL}-{year}-{month}"
            zip_file_name = f"{base_name}.zip"
            csv_file_name = f"{base_name}.csv"
           
            # ---------------------------------------------------------
            # NOWA FUNKCJA: Sprawdzanie czy plik CSV już istnieje
            # ---------------------------------------------------------
            csv_path = os.path.join(DOWNLOAD_DIR, csv_file_name)
            if os.path.exists(csv_path):
                print(f"Pominęto: {csv_file_name} (plik już istnieje na dysku)")
                continue # Przechodzi do następnego miesiąca bez pobierania
            # ---------------------------------------------------------

            url = f"{BASE_URL}/{symbol}/{INTERVAL}/{zip_file_name}"
            print(f"Sprawdzam: {symbol} za {year}-{month}...")
           
            try:
                response = requests.get(url, timeout=10)
               
                if response.status_code == 200:
                    with zipfile.ZipFile(BytesIO(response.content)) as z:
                        z.extractall(DOWNLOAD_DIR)
                    print(f" -> Sukces! Zapisano CSV dla {zip_file_name}")
                elif response.status_code == 404:
                    print(f" -> Brak danych (jeszcze nie opublikowano).")
                else:
                    print(f" -> Błąd pobierania: {response.status_code}")
                   
            except requests.exceptions.RequestException as e:
                print(f" -> Błąd połączenia: {e}")

print("\nGotowe! Wszystkie dostępne pliki CSV znajdują się w folderze.")