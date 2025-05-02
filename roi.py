import cv2
import numpy as np

# Variabel global untuk menyimpan titik-titik ROI
roi_polygon = []
drawing = False  # Indikator apakah ROI sedang dibuat

def draw_roi(event, x, y, flags, param):
    """
    Fungsi callback untuk mendeteksi klik mouse.
    """
    global roi_polygon, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Ketika tombol kiri mouse ditekan, tambahkan titik ke ROI
        roi_polygon.append([x, y])
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Ketika mouse digerakkan setelah klik, gambar garis ROI
        temp_polygon = roi_polygon.copy()
        temp_polygon.append([x, y])  # Tambahkan titik mouse saat ini
        cv2.polylines(frame, [np.array(temp_polygon)], isClosed=False, color=(0, 0, 255), thickness=2)

    elif event == cv2.EVENT_LBUTTONUP:
        # Ketika tombol kiri mouse dilepas, hentikan pembuatan ROI
        drawing = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Ketika tombol kanan mouse ditekan, hapus semua titik ROI
        roi_polygon.clear()

# Buka video stream
video_path = r"E:\CrowdCounting-using-CRSNet-main\🔴 Demo Mahasiswa BEM Seluruh Indonesia (SI) Pantauan CCTV BaliTower Jl Gatot Subroto, Seberang DPR - YouTube and 1 more page - Personal - Microsoft​ Edge 2025-01-29 10-56-26.mp4"
cap = cv2.VideoCapture(video_path)

# Pastikan video berhasil dibuka
if not cap.isOpened():
    print("Gagal membuka video.")
    exit()

# Tentukan ukuran baru untuk frame
new_width = 1080
new_height = 720

# Membuka jendela untuk menampilkan frame
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame. Mungkin video telah selesai.")
        break

    # Mengubah ukuran frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Gambar ROI poligon jika sudah ada titik
    if len(roi_polygon) > 0:
        # Gambar garis ROI
        cv2.polylines(frame, [np.array(roi_polygon)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Tampilkan teks konfirmasi jika ROI sudah lengkap
        if len(roi_polygon) >= 3:
            cv2.putText(frame, "Tekan 's' untuk simpan ROI", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Frame', frame)

    # Tangani input keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Keluar jika tombol 'q' ditekan
        break
    elif key == ord('s') and len(roi_polygon) >= 3:
        # Simpan ROI jika tombol 's' ditekan dan ROI sudah lengkap
        print("ROI disimpan:")
        print(np.array(roi_polygon))
        break

# Tutup video dan jendela
cap.release()
cv2.destroyAllWindows()