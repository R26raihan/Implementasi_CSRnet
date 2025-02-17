from fastapi import FastAPI
from fastapi.responses import JSONResponse
import cv2
import torch
from torchvision import transforms
from model import CSRNet
from PIL import Image
import numpy as np
import threading
import time

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = CSRNet().to(device)
checkpoint = torch.load('E:/CrowdCounting-using-CRSNet-main/weights.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Video input dari 3 lokasi berbeda
video_sources = {
    "DPR": r"E:\CrowdCounting-using-CRSNet-main\🔴 Demo Mahasiswa BEM Seluruh Indonesia (SI) Pantauan CCTV BaliTower Jl Gatot Subroto, Seberang DPR - YouTube and 1 more page - Personal - Microsoft​ Edge 2025-01-29 10-56-26.mp4",
    "Bundaran HI": r"E:\CrowdCounting-using-CRSNet-main\Drone Report_ Ribuan Mahasiswa Padati Bundaran HI - YouTube and 4 more pages - Personal - Microsoft​ Edge 2025-02-14 13-54-14.mp4",
    "Monas": r"E:\CrowdCounting-using-CRSNet-main\Shalat Jum'at Jutaan Umat Islam di Monas - YouTube and 9 more pages - Personal - Microsoft​ Edge 2025-02-17 15-13-02.mp4",
}

latest_predictions = {loc: {"predicted_count": 0} for loc in video_sources}

def capture_and_predict(location, video_path):
    global latest_predictions
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"Video {location} selesai atau tidak bisa dibuka.")
                break
            
            frame = cv2.resize(frame, (1080, 720))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
            predicted_count = int(output.detach().cpu().sum().numpy())

            latest_predictions[location] = {"predicted_count": predicted_count}
            print(f"{location} - Predicted Count: {predicted_count}")

            time.sleep(5)
        except Exception as e:
            print(f"Error at {location}: {e}")
            break
    cap.release()

# Jalankan thread untuk setiap video
for loc, path in video_sources.items():
    threading.Thread(target=capture_and_predict, args=(loc, path), daemon=True).start()

@app.get("/get_predictions")
def get_predictions():
    return JSONResponse(content=latest_predictions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)