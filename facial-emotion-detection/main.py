from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import requests
import time

# URL IP Webcam
url = 'http://192.168.18.56:8080/shot.jpg'

# Load YOLO model
model = YOLO("best.pt")

print("Memulai deteksi menggunakan model YOLO dan IP Webcam...")
print(f"Mengakses kamera di: {url}")
print("Tekan 'q' untuk keluar")

# Untuk menyimpan prediksi terakhir
last_results = None

try:
    while True:
        try:
            # Ambil gambar dari IP Webcam
            img_resp = requests.get(url, timeout=5)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            
            # Jika frame tidak berhasil dibaca, lewati iterasi ini
            if frame is None:
                print("Tidak dapat membaca frame dari IP Webcam. Pastikan aplikasi IP Webcam berjalan.")
                time.sleep(1)  # Tunggu sebentar sebelum mencoba lagi
                continue
            
            # Jalankan YOLO pada frame
            results = model.predict(source=frame, show=True, conf=0.25)
            last_results = results
            
            # Tekan 'q' untuk keluar (catatan: perlu menekan tombol q dalam jendela yang muncul)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error koneksi ke IP Webcam: {e}")
            print("Pastikan IP Webcam berjalan dan URL benar.")
            time.sleep(2)  # Tunggu sebentar sebelum mencoba lagi
        except Exception as e:
            print(f"Error tidak terduga: {e}")
            time.sleep(1)  # Tunggu sebentar sebelum mencoba lagi

except KeyboardInterrupt:
    print("Program dihentikan dengan keyboard interrupt.")
finally:
    # Tutup semua jendela
    cv2.destroyAllWindows()
    print("Program dihentikan.")