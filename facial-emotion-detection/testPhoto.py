from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load YOLO model
model = YOLO("best.pt")

# Folder input dan output
input_folder = "input_folder"  # Ganti dengan lokasi folder input Anda
output_folder = "hasil_deteksi_emosi"

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder {output_folder} telah dibuat")

print("Memulai deteksi emosi menggunakan model YOLO...")
print(f"Mencari gambar di folder: {input_folder}")
print(f"Hasil akan disimpan di folder: {output_folder}")

# Ekstensi file gambar yang didukung
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

try:
    # Dapatkan semua file gambar dari folder input
    image_files = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file_path)
    
    if not image_files:
        print(f"Tidak ada file gambar di folder {input_folder}")
    else:
        print(f"Ditemukan {len(image_files)} file gambar")
        
        # Proses setiap gambar
        for img_path in image_files:
            try:
                print(f"Memproses: {img_path}")
                
                # Baca gambar
                frame = cv2.imread(img_path)
                
                if frame is None:
                    print(f"Tidak dapat membaca gambar: {img_path}")
                    continue
                
                # Jalankan YOLO pada gambar
                results = model.predict(source=frame, conf=0.25)
                
                # Dapatkan nama file tanpa ekstensi
                base_filename = os.path.basename(img_path)
                filename_without_ext = os.path.splitext(base_filename)[0]
                
                # Gambar dengan anotasi
                annotated_frame = results[0].plot()
                
                # Simpan hasil deteksi (gambar dengan anotasi)
                output_path = os.path.join(output_folder, f"{filename_without_ext}_hasil.jpg")
                cv2.imwrite(output_path, annotated_frame)
                
                # Simpan informasi klasifikasi ke file teks
                info_path = os.path.join(output_folder, f"{filename_without_ext}_hasil.txt")
                with open(info_path, 'w') as f:
                    f.write(f"Hasil deteksi untuk: {base_filename}\n")
                    f.write("=" * 50 + "\n")
                    
                    # Tulis semua kelas yang terdeteksi dengan confidence
                    detected_emotions = []
                    for result in results:
                        for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                            class_name = result.names[int(cls)]
                            confidence = float(conf)
                            emotion_info = f"Emosi: {class_name}, Confidence: {confidence:.2f}"
                            detected_emotions.append(emotion_info)
                            f.write(f"{emotion_info}\n")
                    
                    if not detected_emotions:
                        f.write("Tidak ada emosi yang terdeteksi dalam gambar ini.\n")
                
                print(f"Hasil disimpan di {output_path} dan {info_path}")
                
                # Tampilkan informasi deteksi di konsol
                if detected_emotions:
                    for emotion in detected_emotions:
                        print(f"  {emotion}")
                else:
                    print("  Tidak ada emosi yang terdeteksi")
                    
            except Exception as e:
                print(f"Error memproses {img_path}: {e}")
        
        print(f"\nSelesai memproses {len(image_files)} gambar")

except Exception as e:
    print(f"Error tidak terduga: {e}")

print("Program selesai.")