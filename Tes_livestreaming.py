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
model.eval()  # Set model ke evaluation mode
print("Model CSRNet berhasil dimuat.")

# Ganti dengan path ke chromedriver yang sudah diunduh
chrome_driver_path = "E:\CrowdCounting-using-CRSNet-main\chromedriver-win64\chromedriver.exe"

# Membuat layanan Chrome Driver
driver_service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=driver_service)

# URL CCTV Stream
cctv_url = "https://cctv.balitower.co.id/JPO-Merdeka-Barat-507357_9/embed.html"  # Ganti dengan URL CCTV Anda
driver.get(cctv_url)

# Tunggu beberapa detik untuk memastikan video dimuat
time.sleep(5)

# Tentukan ukuran baru untuk frame
new_width = 1080
new_height = 720

while True:
    # Mengambil screenshot dari halaman web
    driver.save_screenshot('screenshot.png')

    # Membaca screenshot dengan OpenCV
    frame = cv2.imread('screenshot.png')

    # Memotong bagian video dari frame (sesuaikan koordinatnya)
    frame = frame[300:820, 200:880]  

    if frame is None:
        print("Tidak dapat membaca frame. Mungkin CCTV tidak tersedia atau video telah selesai.")
        break

    # Mengubah ukuran frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Convert frame to PIL Image untuk processing
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)  # Pindahkan tensor ke CUDA

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)

    predicted_count = int(output.detach().cpu().sum().numpy())
    print("Predicted Count: ", predicted_count)

    # Gaya teks dan warna
    text = f'Hasil prediksi: {predicted_count}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Menambahkan background untuk teks
    text_x = 10
    text_y = 40
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    # Proses heatmap agar hanya muncul di area dengan kepadatan tinggi
    temp = np.asarray(output.detach().cpu().reshape(output.shape[2], output.shape[3]))
    heatmap = (temp - temp.min()) / (temp.max() - temp.min())  # Normalize untuk visualisasi
    heatmap = (heatmap * 255).astype(np.uint8)

    # Threshold untuk menghilangkan area dengan kepadatan rendah
    threshold = 30  # Bisa disesuaikan
    _, mask = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)

    # Terapkan masking ke heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored[mask == 0] = 0  # Hanya menampilkan heatmap di area yang memiliki kepadatan tinggi

    # Resize heatmap agar sesuai dengan dimensi frame
    heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

    # Kombinasikan frame dengan heatmap
    combined = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    # Menampilkan hasil
    cv2.imshow('Crowd Detection', combined)

    # Exit jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hentikan proses
driver.quit()
cv2.destroyAllWindows()
