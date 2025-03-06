from flask import Flask, request, jsonify
from flask_cors import CORS 
from PIL import Image, ImageDraw, ImageFont 
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import mysql.connector
import io
import base64
from ultralytics import YOLO
import cv2
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app) 

# Konfigurasi database MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123',  # Password yang Anda atur di langkah sebelumnya
    'database': 'face_recognition_db',
    'port': 3306
}

# Folder untuk menyimpan gambar wajah
UPLOAD_FOLDER = 'face_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fungsi untuk terhubung ke database
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)
# Fungsi untuk inisialisasi database
def init_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Membuat tabel untuk menyimpan data pengguna dan encoding wajah
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        face_encoding LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Membuat tabel untuk menyimpan data absensi
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        attendance_date DATE NOT NULL,
        check_in_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE KEY unique_attendance (user_id, attendance_date)
    )
    ''')
    # Membuat tabel untuk menyimpan data emosi
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        emotion VARCHAR(50) NOT NULL,
        confidence FLOAT NOT NULL,
        detected_at DATETIME NOT NULL,
        image_path VARCHAR(255),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Periksa apakah kolom image_path sudah ada di tabel emotions
    try:
        cursor.execute('''
        SELECT COUNT(*) AS column_exists
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
        AND table_name = 'emotions'
        AND column_name = 'image_path'
        ''')
        column_exists = cursor.fetchone()[0]
        
        # Jika kolom belum ada, tambahkan
        if not column_exists:
            cursor.execute('''
            ALTER TABLE emotions ADD COLUMN image_path VARCHAR(255) AFTER detected_at
            ''')
            print("Kolom image_path berhasil ditambahkan ke tabel emotions")
    except Exception as e:
        print(f"Error saat memeriksa atau menambahkan kolom image_path: {str(e)}")
    
    # Cara aman untuk membuat indeks jika belum ada
    # Periksa apakah indeks sudah ada
    try:
        cursor.execute('''
        SELECT COUNT(1) IndexIsThere 
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE table_schema=DATABASE() 
        AND table_name='emotions' 
        AND index_name='idx_emotions_user_id'
        ''')
        
        index_exists = cursor.fetchone()[0]
        if not index_exists:
            cursor.execute('CREATE INDEX idx_emotions_user_id ON emotions(user_id)')
    except Exception as e:
        print(f"Error checking or creating index idx_emotions_user_id: {e}")
    
    # Ulangi untuk indeks lainnya
    try:
        cursor.execute('''
        SELECT COUNT(1) IndexIsThere 
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE table_schema=DATABASE() 
        AND table_name='emotions' 
        AND index_name='idx_emotions_detected_at'
        ''')
        
        index_exists = cursor.fetchone()[0]
        if not index_exists:
            cursor.execute('CREATE INDEX idx_emotions_detected_at ON emotions(detected_at)')
    except Exception as e:
        print(f"Error checking or creating index idx_emotions_detected_at: {e}")
    
    try:
        cursor.execute('''
        SELECT COUNT(1) IndexIsThere 
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE table_schema=DATABASE() 
        AND table_name='emotions' 
        AND index_name='idx_emotions_emotion'
        ''')
        
        index_exists = cursor.fetchone()[0]
        if not index_exists:
            cursor.execute('CREATE INDEX idx_emotions_emotion ON emotions(emotion)')
    except Exception as e:
        print(f"Error checking or creating index idx_emotions_emotion: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized with all tables")


# Inisialisasi database saat aplikasi dimulai
init_database()

# Endpoint untuk mendaftarkan wajah baru
@app.route('/api/register', methods=['POST'])
def register_face():
    try:
        data = request.json
        
        if 'name' not in data or 'image' not in data:
            return jsonify({'error': 'Name dan image diperlukan'}), 400
        
        name = data['name']
        image_data = data['image']  # Base64 encoded image
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        except:
            return jsonify({'error': 'Format gambar tidak valid'}), 400
        
        # Simpan gambar ke file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"{name.replace(' ', '_')}_{timestamp}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Deteksi wajah dan buat encoding
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            os.remove(image_path)  # Hapus file jika tidak ada wajah terdeteksi
            return jsonify({'error': 'Tidak ada wajah terdeteksi dalam gambar'}), 400
        
        # Ambil encoding wajah pertama yang ditemukan
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        # Simpan data ke database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Konversi numpy array ke string
        face_encoding_str = ','.join(map(str, face_encoding))
        
        cursor.execute(
            "INSERT INTO users (name, image_path, face_encoding) VALUES (%s, %s, %s)",
            (name, image_path, face_encoding_str)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Wajah berhasil didaftarkan',
            'user_id': user_id
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Endpoint untuk pengenalan wajah
@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'Image diperlukan'}), 400
        
        image_data = request.json['image']  # Base64 encoded image
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        except:
            return jsonify({'error': 'Format gambar tidak valid'}), 400
        
        # Simpan gambar sementara
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_recognize.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Deteksi wajah
        unknown_image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(unknown_image)
        
        if not face_locations:
            os.remove(temp_path)
            return jsonify({
                'success': True,
                'message': 'Tidak ada wajah terdeteksi dalam gambar',
                'faces': [],
                'attendance_results': [],
                'emotion_results': []
            }), 200
        
        # Dapatkan encoding untuk setiap wajah yang terdeteksi
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        
        # Dapatkan semua wajah dari database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, face_encoding FROM users")
        users = cursor.fetchall()
        
        if not users:
            cursor.close()
            conn.close()
            os.remove(temp_path)
            return jsonify({
                'success': True,
                'message': 'Tidak ada wajah terdaftar dalam database',
                'faces': [],
                'attendance_results': [],
                'emotion_results': []
            }), 200
        
        # Konversi encoding wajah dari database kembali ke numpy array
        known_face_encodings = []
        known_face_names = []
        known_face_ids = []
        
        for user in users:
            # Konversi string ke numpy array
            face_encoding = np.array([float(x) for x in user['face_encoding'].split(',')])
            known_face_encodings.append(face_encoding)
            known_face_names.append(user['name'])
            known_face_ids.append(user['id'])
        
        # Load YOLO model untuk deteksi emosi
        model = YOLO("best.pt")
        
        # Deteksi emosi dengan YOLO
        emotion_results = model.predict(source=unknown_image, conf=0.25)
        
        # Buat gambar dengan kotak dan nama
        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Daftar untuk menyimpan hasil pengenalan
        recognized_faces = []
        attendance_results = []
        emotion_data = []
        today_date = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_filename = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Buat folder untuk penyimpanan gambar emosi
        emotion_folder = os.path.join(UPLOAD_FOLDER, 'emotion_detection')
        if not os.path.exists(emotion_folder):
            os.makedirs(emotion_folder)
        
        # Periksa setiap wajah yang terdeteksi
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Bandingkan dengan wajah yang diketahui
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            user_id = None
            confidence = 0
            attendance_status = "Gagal"
            attendance_message = "Wajah tidak dikenali"
            
            # Gunakan wajah dengan jarak terkecil
            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    user_id = known_face_ids[best_match_index]
                    # Konversi jarak ke persentase kepercayaan
                    confidence = (1 - face_distances[best_match_index]) * 100
                    
                    # Periksa apakah pengguna sudah absen hari ini
                    cursor.execute(
                        "SELECT id FROM attendance WHERE user_id = %s AND DATE(attendance_date) = %s",
                        (user_id, today_date)
                    )
                    existing_attendance = cursor.fetchone()
                    
                    if existing_attendance:
                        attendance_status = "Gagal"
                        attendance_message = f"{name} sudah melakukan absensi hari ini"
                    else:
                        try:
                            # Tambahkan data absensi ke database
                            cursor.execute(
                                "INSERT INTO attendance (user_id, attendance_date) VALUES (%s, %s)",
                                (user_id, datetime.now())
                            )
                            conn.commit()
                            attendance_status = "Berhasil"
                            attendance_message = f"{name} berhasil diabsen pada {timestamp}"
                        except mysql.connector.IntegrityError:
                            # Jika terjadi error unique constraint
                            attendance_status = "Gagal"
                            attendance_message = f"{name} sudah melakukan absensi hari ini"
                        except Exception as e:
                            attendance_status = "Gagal"
                            attendance_message = f"Error saat menyimpan absensi: {str(e)}"
            
            # Cari emosi yang terdeteksi di lokasi wajah ini
            detected_emotion = "Tidak dikenali"
            emotion_confidence = 0
            
            # Dapatkan koordinat tengah wajah saat ini
            face_center_x = (left + right) / 2
            face_center_y = (top + bottom) / 2
            
            # Periksa semua hasil deteksi emosi
            for result in emotion_results:
                for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                    # Dapatkan koordinat kotak deteksi emosi
                    e_left, e_top, e_right, e_bottom = box.tolist()
                    # Dapatkan koordinat tengah kotak emosi
                    e_center_x = (e_left + e_right) / 2
                    e_center_y = (e_top + e_bottom) / 2
                    
                    # Periksa apakah kotak emosi berada di dalam kotak wajah yang sedang diproses
                    if (left <= e_center_x <= right and top <= e_center_y <= bottom):
                        # Deteksi emosi untuk wajah saat ini
                        detected_emotion = result.names[int(cls)]
                        emotion_confidence = float(conf) * 100
                        break
            
            padding = 30  # Tambahkan padding di bawah untuk tempat label
            face_image = pil_image.crop((left, top, right, bottom + padding))

            # Buat objek ImageDraw untuk menggambar di gambar crop
            face_draw = ImageDraw.Draw(face_image)

            # Gambar kotak di sekitar wajah (sesuaikan dengan ukuran crop)
            face_draw.rectangle(((0, 0), (right - left, bottom - top)), outline=(0, 0, 255), width=2)

            # Tambahkan label emosi di bagian bawah
            font = ImageFont.load_default()
            label_text = f"{name} - {detected_emotion} ({round(emotion_confidence, 1)}%)"
            text_bbox = face_draw.textbbox((0, bottom - top + 5), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Tambahkan background berwarna untuk label
            # Warna background berdasarkan jenis emosi (dapat disesuaikan)
            emotion_colors = {
                "Senang": (46, 204, 113),     # Hijau
                "Sedih": (52, 152, 219),      # Biru
                "Marah": (231, 76, 60),       # Merah
                "Terkejut": (155, 89, 182),   # Ungu
                "Takut": (241, 196, 15),      # Kuning
                "Jijik": (211, 84, 0),        # Oranye
                "Netral": (149, 165, 166)     # Abu-abu
            }

            # Jika emosi yang terdeteksi ada dalam kamus, gunakan warna tersebut, jika tidak gunakan abu-abu
            bg_color = emotion_colors.get(detected_emotion, (149, 165, 166))

            # Gambar background untuk teks
            face_draw.rectangle(((0, bottom - top + 2), (right - left, bottom - top + padding - 2)), fill=bg_color)

            # Tambahkan teks dengan warna putih
            face_draw.text((5, bottom - top + 5), label_text, fill=(255, 255, 255), font=font)
            
            # Buat sub-folder untuk setiap pengguna jika belum ada
            user_folder = os.path.join(emotion_folder, f"user_{user_id if user_id else 'unknown'}")
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)
                
            # Buat sub-folder untuk setiap jenis emosi jika belum ada
            emotion_type_folder = os.path.join(user_folder, detected_emotion)
            if not os.path.exists(emotion_type_folder):
                os.makedirs(emotion_type_folder)
                
            emotion_image_filename = f"{name.replace(' ', '_')}_{detected_emotion}_{timestamp_filename}_{user_id if user_id else 'unknown'}.jpg"
            emotion_image_path = os.path.join(emotion_type_folder, emotion_image_filename)
            face_image.save(emotion_image_path)
            
            # Path relatif untuk disimpan ke database (tanpa UPLOAD_FOLDER)
            relative_path = os.path.join('emotion_detection', f"user_{user_id if user_id else 'unknown'}", 
                                         detected_emotion, emotion_image_filename)
            
            # Simpan data emosi ke database jika user dikenali
            if user_id is not None:
                try:
                    # Modifikasi tabel emotions untuk menyimpan path gambar
                    cursor.execute(
                        """INSERT INTO emotions 
                           (user_id, emotion, confidence, detected_at, image_path) 
                           VALUES (%s, %s, %s, %s, %s)""",
                        (user_id, detected_emotion, emotion_confidence, datetime.now(), relative_path)
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Error saat menyimpan data emosi: {str(e)}")
            
            # Tambahkan data emosi ke hasil
            emotion_data.append({
                'user_id': user_id,
                'name': name,
                'emotion': detected_emotion,
                'confidence': round(emotion_confidence, 2),
                'timestamp': timestamp,
                'image_path': relative_path
            })
            
            # Tambahkan ke daftar hasil
            recognized_faces.append({
                'name': name,
                'user_id': user_id,
                'confidence': round(float(confidence), 2),
                'emotion': detected_emotion,
                'emotion_confidence': round(emotion_confidence, 2),
                'image_path': relative_path,
                'location': {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            })
            
            # Tambahkan hasil absensi
            if user_id is not None:
                attendance_results.append({
                    'user_id': user_id,
                    'name': name,
                    'status': attendance_status,
                    'message': attendance_message,
                    'timestamp': timestamp
                })
            
            # Gambar kotak di sekitar wajah
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=2)
            
            # Gambar label dengan nama dan emosi di bawah wajah
            font = ImageFont.load_default()
            label_text = f"{name} - {detected_emotion}"
            text_bbox = draw.textbbox((left, bottom), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Warna background label berdasarkan status absensi
            background_color = (0, 0, 255)  # Biru default untuk Unknown
            if user_id is not None:
                attendance_info = next((a for a in attendance_results if a['user_id'] == user_id), None)
                if attendance_info and attendance_info['status'] == 'Berhasil':
                    background_color = (0, 255, 0)  # Hijau untuk absensi berhasil
                elif attendance_info and attendance_info['status'] == 'Gagal':
                    background_color = (255, 0, 0)  # Merah untuk absensi gagal
            
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=background_color)
            draw.text((left + 6, bottom - text_height - 5), label_text, fill=(255, 255, 255), font=font)
        
        # Tutup koneksi database
        cursor.close()
        conn.close()
        
        # Simpan gambar dengan anotasi
        result_path = os.path.join(UPLOAD_FOLDER, f'recognized_result_{timestamp_filename}.jpg')
        pil_image.save(result_path)
        
        # Konversi gambar hasil ke base64
        with open(result_path, 'rb') as img_file:
            result_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Hapus file sementara
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'message': f'Ditemukan {len(recognized_faces)} wajah',
            'faces': recognized_faces,
            'attendance_results': attendance_results,
            'emotion_results': emotion_data,
            'result_image': f'data:image/jpeg;base64,{result_image_base64}'
        }), 200
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print full error stacktrace
        return jsonify({'error': str(e)}), 500

# Tambahkan endpoint untuk mendapatkan data emosi
@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    try:
        # Koneksi ke database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Parameter filter opsional
        user_id = request.args.get('user_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        emotion_type = request.args.get('emotion')
        limit = request.args.get('limit', default=100, type=int)
        
        # Buat query dasar
        query = """
            SELECT e.id, e.user_id, u.name, e.emotion, e.confidence, 
                   e.detected_at, DATE_FORMAT(e.detected_at, '%Y-%m-%d') as date,
                   DATE_FORMAT(e.detected_at, '%H:%i:%s') as time,
                   e.image_path
            FROM emotions e
            JOIN users u ON e.user_id = u.id
            WHERE 1=1
        """
        params = []
        
        # Tambahkan filter jika ada
        if user_id:
            query += " AND e.user_id = %s"
            params.append(int(user_id))  # Pastikan user_id adalah integer
        
        if start_date:
            query += " AND DATE(e.detected_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(e.detected_at) <= %s"
            params.append(end_date)
        
        if emotion_type:
            query += " AND e.emotion = %s"
            params.append(emotion_type)
        
        # Urutkan berdasarkan tanggal terbaru
        query += f" ORDER BY e.detected_at DESC LIMIT {limit}"
        
        # Eksekusi query dan debug
        print("Query:", query)
        print("Params:", params)
        cursor.execute(query, tuple(params))  # Convert params to tuple
        emotions = cursor.fetchall()
        
        # Tambahkan query untuk mendapatkan statistik emosi
        stats_query = """
            SELECT emotion, COUNT(*) as count, 
                   AVG(confidence) as avg_confidence,
                   DATE_FORMAT(MAX(detected_at), '%Y-%m-%d %H:%i:%s') as last_detected
            FROM emotions
            WHERE 1=1
        """
        stats_params = []
        
        # Tambahkan filter yang sama ke query statistik
        if user_id:
            stats_query += " AND user_id = %s"
            stats_params.append(int(user_id))  # Pastikan user_id adalah integer
        
        if start_date:
            stats_query += " AND DATE(detected_at) >= %s"
            stats_params.append(start_date)
        
        if end_date:
            stats_query += " AND DATE(detected_at) <= %s"
            stats_params.append(end_date)
        
        stats_query += " GROUP BY emotion ORDER BY count DESC"
        
        # Eksekusi query statistik dan debug
        print("Stats Query:", stats_query)
        print("Stats Params:", stats_params)
        cursor.execute(stats_query, tuple(stats_params))  # Convert params to tuple
        emotion_stats = cursor.fetchall()
        
        # Tutup koneksi database
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Ditemukan {len(emotions)} data emosi',
            'emotions': emotions,
            'stats': emotion_stats,
            'filters': {
                'user_id': user_id,
                'start_date': start_date,
                'end_date': end_date,
                'emotion': emotion_type,
                'limit': limit
            }
        }), 200
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print full error stacktrace
        return jsonify({'error': str(e)}), 500
# Endpoint untuk mendapatkan daftar semua pengguna
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, image_path, created_at FROM users")
        users = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Konversi datetime ke string
        for user in users:
            if isinstance(user['created_at'], datetime):
                user['created_at'] = user['created_at'].strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            'success': True,
            'count': len(users),
            'users': users
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Parameter filter opsional
        date_filter = request.args.get('date')  # Format: YYYY-MM-DD
        user_id = request.args.get('user_id')
        
        # Buat query dasar
        query = """
        SELECT a.id, a.user_id, u.name, DATE(a.attendance_date) as date, 
               TIME(a.check_in_time) as check_in_time, 
               a.check_in_time as timestamp
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE 1=1
        """
        params = []
        
        # Tambahkan filter jika ada
        if date_filter:
            query += " AND DATE(a.attendance_date) = %s"
            params.append(date_filter)
        
        if user_id:
            query += " AND a.user_id = %s"
            params.append(int(user_id))
        
        # Urutkan berdasarkan waktu absensi terbaru
        query += " ORDER BY a.check_in_time DESC"
        
        cursor.execute(query, params)
        attendance_records = cursor.fetchall()
        
        # Format data untuk JSON serialization
        formatted_records = []
        for record in attendance_records:
            formatted_record = {}
            for key, value in record.items():
                # Handle datetime objects
                if isinstance(value, datetime):
                    formatted_record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                # Handle date objects
                elif hasattr(value, 'strftime'):
                    formatted_record[key] = value.strftime('%Y-%m-%d')
                # Handle time objects
                elif hasattr(value, 'isoformat'):
                    formatted_record[key] = value.isoformat()
                # Handle timedelta objects
                elif isinstance(value, timedelta):
                    total_seconds = value.total_seconds()
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = int(total_seconds % 60)
                    formatted_record[key] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    formatted_record[key] = value
            formatted_records.append(formatted_record)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'count': len(formatted_records),
            'attendance': formatted_records
        }), 200
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Cetak error lengkap untuk debugging
        return jsonify({'error': str(e)}), 500

# Endpoint untuk menghapus pengguna
@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Dapatkan path file gambar
        cursor.execute("SELECT image_path FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Pengguna tidak ditemukan'}), 404
        
        image_path = result[0]
        
        # Hapus data absensi terlebih dahulu
        cursor.execute("DELETE FROM attendance WHERE user_id = %s", (user_id,))
        
        # Kemudian hapus record dari tabel users
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        
        # Hapus file gambar jika ada
        if os.path.exists(image_path):
            os.remove(image_path)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Pengguna dengan ID {user_id} berhasil dihapus'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)