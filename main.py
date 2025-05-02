from fastapi import FastAPI
from fastapi.responses import JSONResponse
import cv2
import torch
from torchvision import transforms
from model import CSRNet
from PIL import Image
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
import threading
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Inisialisasi model CSRNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    model = CSRNet().to(device)
    weights_path = "E:\CrowdCounting-using-CRSNet-main\weights.pth"
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info("Model CSRNet berhasil dimuat.")
except Exception as e:
    logging.error(f"Gagal memuat model: {e}")
    exit(1)

# Daftar URL CCTV
cctv_urls = {
    "DPR": "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_1/embed.html",
    "Bundaran HI": "https://cctv.balitower.co.id/Menteng-001-700123_5/embed.html",
    "Monas": "https://cctv.balitower.co.id/Monas-Barat-009-506632_2/embed.html",
    "Patung Kuda": "https://cctv.balitower.co.id/JPO-Merdeka-Barat-507357_9/embed.html",
    "Pospol Istana Negara": "https://cctv.balitower.co.id/Pospol-Merdeka-Utara-506818_1/embed.html",
}

# Definisi ROI untuk Patung Kuda dan DPR
roi_polygons = {
    "Patung Kuda": np.array([
        [224, 675],
        [392, 383],
        [644, 377],
        [970, 671]
    ], dtype=np.int32),
    "DPR": np.array([
        [7, 346],
        [1067, 375],
        [1070, 513],
        [5, 454]
    ], dtype=np.int32),
    # Default ROI untuk lokasi lain (full frame)
    "default": None
}

# Inisialisasi ChromeDriver
chrome_driver_path = "E:\CrowdCounting-using-CRSNet-main\chromedriver-win64\chromedriver.exe"
if not os.path.exists(chrome_driver_path):
    logging.error(f"Chromedriver tidak ditemukan di: {chrome_driver_path}")
    exit(1)
driver_service = Service(chrome_driver_path)

# Variabel global untuk menyimpan hasil prediksi terbaru
latest_predictions = {location: {"predicted_count": 0} for location in cctv_urls.keys()}

# Tentukan ukuran frame
new_width = 1080
new_height = 720

def capture_and_predict(location, url):
    global latest_predictions
    driver = None
    try:
        driver = webdriver.Chrome(service=driver_service)
        driver.get(url)
        logging.info(f"Berhasil membuka URL CCTV untuk {location}")
        time.sleep(5)  # Tunggu video dimuat

        while True:
            try:
                # Ambil screenshot
                screenshot_path = f'screenshot_{location}.png'
                if not driver.save_screenshot(screenshot_path):
                    logging.error(f"Gagal mengambil screenshot untuk {location}")
                    break

                # Baca screenshot
                frame = cv2.imread(screenshot_path)
                if frame is None:
                    logging.error(f"Tidak dapat membaca frame untuk {location}")
                    break

                # Ubah ukuran frame
                frame = cv2.resize(frame, (new_width, new_height))

                # Terapkan ROI jika ada
                roi_frame = frame
                roi_polygon = roi_polygons.get(location, roi_polygons["default"])
                if roi_polygon is not None:
                    mask = np.zeros_like(frame[:, :, 0])  # Mask hitam
                    cv2.fillPoly(mask, [roi_polygon], 255)  # Isi poligon dengan putih
                    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

                # Convert ke PIL Image
                img = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Prediksi
                with torch.no_grad():
                    output = model(img_tensor)
                predicted_count = int(output.detach().cpu().sum().numpy())

                # Simpan hasil prediksi
                latest_predictions[location] = {"predicted_count": predicted_count}
                logging.info(f"{location} Predicted Count: {predicted_count}")

                time.sleep(10)  # Tunggu sebelum screenshot berikutnya

            except Exception as e:
                logging.error(f"Error in {location}: {e}")
                break

    except Exception as e:
        logging.error(f"Gagal menginisialisasi driver untuk {location}: {e}")

    finally:
        if driver is not None:
            driver.quit()
        # Hapus file screenshot
        screenshot_path = f'screenshot_{location}.png'
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

# Jalankan thread untuk setiap CCTV
for location, url in cctv_urls.items():
    threading.Thread(target=capture_and_predict, args=(location, url), daemon=True).start()

@app.get("/get_predictions")
def get_predictions():
    return JSONResponse(content=latest_predictions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)