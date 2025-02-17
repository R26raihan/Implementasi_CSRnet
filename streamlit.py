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
import time
import streamlit as st

# Memastikan bahwa CUDA tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformasi untuk preprocessing gambar
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Memuat model CSRNet dan memindahkannya ke CUDA
model = CSRNet().to(device)
checkpoint = torch.load('E:\CrowdCounting-using-CRSNet-main\weights.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()  # Set model ke mode evaluasi
print("Model CSRNet berhasil dimuat.")

# Ganti dengan path ke chromedriver yang sudah diunduh dan diekstrak
chrome_driver_path = "E:\CrowdCounting-using-CRSNet-main\chromedriver-win64\chromedriver.exe"

# Daftar URL CCTV Stream
cctv_urls = [
    "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_1/embed.html",
    "https://cctv.balitower.co.id/Gelora-017-700470_2/embed.html",
    "https://cctv.balitower.co.id/Menteng-001-700123_5/embed.html",
    "https://cctv.balitower.co.id/Monas-Barat-009-506632_2/embed.html"
]

# Membuka browser hanya sekali
driver_service = Service(chrome_driver_path)
drivers = [webdriver.Chrome(service=driver_service) for _ in cctv_urls]

for driver, url in zip(drivers, cctv_urls):
    driver.get(url)

time.sleep(5)  # Tunggu beberapa detik agar semua video termuat

# Tentukan ukuran baru untuk frame
new_width = 1080
new_height = 720

# Inisialisasi dashboard
st.title("Dashboard Crowd Counting CCTV")
st.write("Menampilkan jumlah orang dari berbagai lokasi CCTV secara real-time.")

# Layout Grid 2x2
cols = st.columns(2)
placeholders = [cols[i % 2].empty() for i in range(4)]

while True:
    for i, driver in enumerate(drivers):
        # Mengambil screenshot dari halaman web
        screenshot_path = f'screenshot_{i}.png'
        driver.save_screenshot(screenshot_path)

        # Membaca screenshot dengan OpenCV
        frame = cv2.imread(screenshot_path)
        if frame is None:
            print(f"Tidak dapat membaca frame dari {cctv_urls[i]}")
            continue

        # Mengubah ukuran frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert frame ke PIL Image untuk processing
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Prediksi jumlah orang
        with torch.no_grad():
            output = model(img_tensor)
        predicted_count = int(output.detach().cpu().sum().numpy())
        print(f"Predicted Count ({cctv_urls[i]}): {predicted_count}")

        # Generate heatmap
        temp = np.asarray(output.detach().cpu().reshape(output.shape[2], output.shape[3]))
        heatmap = (temp - temp.min()) / (temp.max() - temp.min())
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

        # Gabungkan frame dengan heatmap
        combined = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        combined = cv2.copyMakeBorder(combined, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Perbarui tampilan Streamlit
        placeholders[i].image(combined, caption=f"Lokasi CCTV {i+1} - Jumlah Orang: {predicted_count}", use_container_width=True)
    
    time.sleep(5)  # Update setiap 5 detik

# Tutup browser
for driver in drivers:
    driver.quit()
