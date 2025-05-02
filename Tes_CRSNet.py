import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt, cm as c
import torchvision.transforms.functional as F
from model import CSRNet
import torch
from torchvision import transforms
import cv2

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
model.eval()
print("Model CSRNet berhasil dimuat.")

# Path ke video
video_path = r"C:\Users\raiha\Videos\Captures\data demo\JPO-Merdeka-Barat-507357_9 - Google Chrome 2025-02-20 16-44-13.mp4"

# Buka video
cap = cv2.VideoCapture(video_path)

# Tentukan ukuran baru untuk frame
new_width = 1080
new_height = 720

roi_polygon = np.array([
    [224, 675],
    [392, 383],
    [644, 377],
    [970, 671]
], dtype=np.int32)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame. Mungkin video telah selesai.")
        break
    else:
        print("Frame berhasil dibaca.")

    # Mengubah ukuran frame
    frame = cv2.resize(frame, (new_width, new_height))
    
    # Salin frame asli untuk ditampilkan terpisah
    original_frame = frame.copy()

    # Buat mask poligon
    mask = np.zeros_like(frame[:, :, 0])  # Mask hitam dengan ukuran sama dengan frame
    cv2.fillPoly(mask, [roi_polygon], 255)  # Isi poligon dengan warna putih (255)

    # Terapkan mask pada frame untuk mendapatkan ROI
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert frame ROI ke PIL Image untuk preprocessing
    img = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prediksi dari model
    with torch.no_grad():
        output = model(img_tensor)

    predicted_count = int(output.detach().cpu().sum().numpy())
    print("Predicted Count in ROI: ", predicted_count)

    # Gaya teks dan warna untuk frame asli
    text = f'Jumlah Masa Demo: {predicted_count}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Menambahkan background dan teks pada frame asli
    text_x = 10
    text_y = 40
    cv2.rectangle(original_frame, (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(original_frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    # Gambar ROI poligon pada frame asli
    cv2.polylines(original_frame, [roi_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # Membuat heatmap dari output model
    temp = np.asarray(output.detach().cpu().reshape(output.shape[2], output.shape[3]))
    
    # Normalisasi heatmap
    temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-5)
    temp = (temp * 255).astype(np.uint8)

    # Perbesar resolusi heatmap sesuai ukuran ROI
    heatmap_resized = cv2.resize(temp, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Terapkan colormap JET
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Smooth heatmap
    heatmap_colored = cv2.GaussianBlur(heatmap_colored, (15, 15), 0)
    
    # Tempelkan heatmap ke frame asli di area ROI
    alpha = 0.5
    frame = cv2.addWeighted(frame, 0.5, heatmap_colored, alpha, 0)

    # Tambahkan border untuk frame asli
    original_frame = cv2.copyMakeBorder(original_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Tampilkan kedua frame di jendela terpisah
    cv2.imshow('Original Frame', original_frame)
    cv2.imshow('Frame with Heatmap in ROI', frame)

    # Exit jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan objek video
cap.release()
cv2.destroyAllWindows()