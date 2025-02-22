from fastapi import FastAPI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import re

app = FastAPI()

# Path ke ChromeDriver
chrome_driver_path = "E:/CrowdCounting-using-CRSNet-main/chromedriver-win64/chromedriver.exe"

def scrape_data():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Agar berjalan tanpa GUI
    options.add_argument("--no-sandbox")  
    options.add_argument("--disable-dev-shm-usage")  

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # URL Dashboard
    url = "https://jakartasatu.jakarta.go.id/portal/apps/dashboards/c2b19d6243dd4a2f80fa1e55481fdb11"
    driver.get(url)

    # Tunggu loading halaman
    time.sleep(5)

    # XPATH elemen tabel RT terdampak
    try:
        container = driver.find_element(By.XPATH, "/html/body/div/calcite-shell/div[2]/div[2]/div/div/div/margin-container/full-container/div[6]/margin-container/full-container/dashboard-tab-zone/section/div[2]/div/div/div[2]/div")

        # Ambil teks dari elemen
        text_data = container.text.split("\n")

        # Ekstraksi data menggunakan regex
        data_list = []
        for i in range(0, len(text_data), 2):  # Data berpasangan (Lokasi, Tinggi Genangan)
            lokasi_match = re.search(r"RT (\d+) / RW (\d+), Kelurahan (.+)", text_data[i])
            tinggi_match = re.search(r"Tinggi Genangan : ([\d.]+) cm", text_data[i + 1]) if i + 1 < len(text_data) else None

            if lokasi_match and tinggi_match:
                rt_data = {
                    "RT": lokasi_match.group(1),
                    "RW": lokasi_match.group(2),
                    "Kelurahan": lokasi_match.group(3),
                    "Tinggi Genangan (cm)": float(tinggi_match.group(1))
                }
                data_list.append(rt_data)

        driver.quit()
        return data_list

    except Exception as e:
        driver.quit()
        return {"error": str(e)}

# Endpoint untuk mendapatkan data RT terdampak
@app.get("/rt-terdampak")
def get_rt_terdampak():
    data = scrape_data()
    return {"status": "success", "data": data}
