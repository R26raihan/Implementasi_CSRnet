import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from model import CSRNet
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memastikan bahwa CUDA tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Transformasi untuk preprocessing gambar
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Memuat model CSRNet
try:
    model = CSRNet().to(device)
    weights_path = os.path.join(os.path.dirname(__file__), 'weights.pth')  # Relative path
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()  # Set model ke evaluation mode
    logging.info("Model CSRNet berhasil dimuat.")
except Exception as e:
    logging.error(f"Gagal memuat model: {e}")
    exit(1)

# Path ke chromedriver
chrome_driver_path = os.path.join(os.path.dirname(__file__), 'chromedriver-win64', 'chromedriver.exe')
if not os.path.exists(chrome_driver_path):
    logging.error(f"Chromedriver tidak ditemukan di: {chrome_driver_path}")
    exit(1)

# Membuat layanan Chrome Driver
try:
    driver_service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=driver_service)
except Exception as e:
    logging.error(f"Gagal menginisialisasi Chrome driver: {e}")
    exit(1)

# URL CCTV Stream
cctv_url = "https://cctv.balitower.co.id/JPO-Merdeka-Barat-507357_9/embed.html"
try:
    driver.get(cctv_url)
    logging.info("Berhasil membuka URL CCTV.")
    time.sleep(5)  # Tunggu video dimuat
except Exception as e:
    logging.error(f"Gagal membuka URL CCTV: {e}")
    driver.quit()
    exit(1)

# Tentukan ukuran baru untuk frame
new_width = 1080
new_height = 720

# Titik-titik ROI (hilangkan dimensi ekstra)
roi_polygon = np.array([
    [224, 675],
    [392, 383],
    [644, 377],
    [970, 671]
], dtype=np.int32)

try:
    while True:
        # Mengambil screenshot dari halaman web
        screenshot_path = 'screenshot.png'
        if not driver.save_screenshot(screenshot_path):
            logging.error("Gagal mengambil screenshot.")
            break

        # Membaca screenshot dengan OpenCV
        frame = cv2.imread(screenshot_path)
        if frame is None:
            logging.error("Tidak dapat membaca frame. Mungkin CCTV tidak tersedia.")
            break

        # Mengubah ukuran frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Buat mask poligon
        mask = np.zeros_like(frame[:, :, 0])  # Mask hitam
        cv2.fillPoly(mask, [roi_polygon], 255)  # Isi poligon dengan putih

        # Terapkan mask pada frame untuk mendapatkan ROI
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert frame ROI ke PIL Image
        img = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))

        # Apply transformations
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)

        predicted_count = int(output.detach().cpu().sum().numpy())
        logging.info(f"Predicted Count in ROI: {predicted_count}")

        # Gaya teks dan warna
        text = f'Hasil prediksi: {predicted_count}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Menambahkan background untuk teks
        text_x, text_y = 10, 40
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                    (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Proses heatmap
        temp = np.asarray(output.detach().cpu().reshape(output.shape[2], output.shape[3]))
        heatmap = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)  # Avoid division by zero
        heatmap = (heatmap * 255).astype(np.uint8)

        # Threshold untuk area kepadatan tinggi
        threshold = 30
        _, mask_heatmap = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)

        # Terapkan masking ke heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored[mask_heatmap == 0] = 0

        # Resize heatmap
        heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

        # Terapkan heatmap hanya pada area ROI
        heatmap_roi = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=mask)

        # Kombinasikan frame dengan heatmap ROI
        combined = cv2.addWeighted(frame, 0.6, heatmap_roi, 0.4, 0)

        # Gambar ROI poligon
        cv2.polylines(combined, [roi_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        # Menampilkan hasil
        cv2.imshow('Crowd Detection', combined)

        # Exit jika tekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error(f"Terjadi kesalahan selama pemrosesan: {e}")

finally:
    # Bersihkan resource
    logging.info("Membersihkan resource...")
    driver.quit()
    cv2.destroyAllWindows()
    if os.path.exists('screenshot.png'):
        os.remove('screenshot.png')  # Hapus file sementara