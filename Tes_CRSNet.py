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
model.eval()  # Set model to evaluation mode
print("Model CSRNet berhasil dimuat.")

# Path to the video file
video_path = r"E:\CrowdCounting-using-CRSNet-main\Drone Report_ Ribuan Mahasiswa Padati Bundaran HI - YouTube and 4 more pages - Personal - Microsoft​ Edge 2025-02-14 13-54-14.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Tentukan ukuran baru untuk frame
new_width = 1080
new_height = 720

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame. Mungkin video telah selesai.")
        break
    else:
        print("Frame berhasil dibaca.")
    
    # Mengubah ukuran frame
    frame = cv2.resize(frame, (new_width, new_height))
    
    # Convert frame to PIL Image for processing
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
    
    # Show heatmap
    temp = np.asarray(output.detach().cpu().reshape(output.shape[2], output.shape[3]))
    heatmap = (temp - temp.min()) / (temp.max() - temp.min())  # Normalize for visualization
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to match frame dimensions
    heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

    # Combine original frame with heatmap
    combined = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    # Menambahkan border pada heatmap
    combined = cv2.copyMakeBorder(combined, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Display the frame
    cv2.imshow('Crowd Detection', combined)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video object
cap.release()
cv2.destroyAllWindows()
