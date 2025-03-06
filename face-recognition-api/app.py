# app.py
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
from werkzeug.utils import secure_filename

from database import db, init_db
from models import User, RecognitionLog
from face_utils import get_face_encoding, recognize_face

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS untuk semua routes

# app.py (lanjutan)
UPLOAD_FOLDER = 'uploads'
KNOWN_FOLDER = os.path.join(UPLOAD_FOLDER, 'known')
UNKNOWN_FOLDER = os.path.join(UPLOAD_FOLDER, 'unknown')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Pastikan direktori untuk upload ada
os.makedirs(KNOWN_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

# Inisialisasi database
init_db(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify({
        'success': True,
        'users': [user.to_dict() for user in users]
    })

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })

@app.route('/api/users', methods=['POST'])
def register_user():
    if 'name' not in request.form:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    # Simpan gambar dengan nama unik
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    image_path = os.path.join(KNOWN_FOLDER, filename)
    file.save(image_path)
    
    # Dapatkan encoding wajah
    face_encoding = get_face_encoding(image_path)
    if face_encoding is None:
        os.remove(image_path)  # Hapus file jika tidak ada wajah terdeteksi
        return jsonify({'success': False, 'error': 'No face detected in the image'}), 400
    
    # Simpan data ke database
    new_user = User(
        name=request.form['name'],
        image_path=image_path,
        face_encoding=face_encoding
    )
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'User registered successfully',
        'user': new_user.to_dict()
    }), 201

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    # Simpan gambar yang dikirim
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    image_path = os.path.join(UNKNOWN_FOLDER, filename)
    file.save(image_path)
    # Ambil semua user dengan encoding wajah
    users = User.query.all()
    
    if not users:
        # Tidak ada wajah yang terdaftar untuk dibandingkan
        log = RecognitionLog(
            user_id=None,
            image_path=image_path,
            recognized=False
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'recognized': False,
            'message': 'No registered faces to compare with',
            'faces': [],
            'image_path': f'/uploads/unknown/{filename}'
        })
    
    # Siapkan data encoding wajah yang diketahui
    known_encodings = []
    known_names = []
    known_ids = []
    
    for user in users:
        known_encodings.append(np.array(json.loads(user.face_encoding)))
        known_names.append(user.name)
        known_ids.append(user.id)
    
    # Lakukan pengenalan wajah
    results, result_path = recognize_face(image_path, known_encodings, known_names)
    
    # Simpan log pengenalan
    for result in results:
        log = RecognitionLog(
            user_id=result['user_id'],
            image_path=image_path,
            recognized=result['recognized']
        )
        db.session.add(log)
    
    db.session.commit()
    
    # Hapus file yang diunggah jika tidak ingin menyimpannya
    # os.remove(image_path)
    
    return jsonify({
        'success': True,
        'recognized': any(result['recognized'] for result in results),
        'faces': results,
        'image_path': f'/uploads/unknown/{os.path.basename(result_path)}'
    })

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    # Hapus file gambar jika ada
    if os.path.exists(user.image_path):
        os.remove(user.image_path)
    
    # Hapus data dari database
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'User {user.name} deleted successfully'
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = RecognitionLog.query.order_by(RecognitionLog.created_at.desc()).all()
    return jsonify({
        'success': True,
        'logs': [log.to_dict() for log in logs]
    })

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Endpoint untuk menampilkan gambar wajah terdaftar
@app.route('/api/user_image/<int:user_id>')
def user_image(user_id):
    user = User.query.get_or_404(user_id)
    return send_from_directory(os.path.dirname(user.image_path), os.path.basename(user.image_path))

# Endpoint tambahan untuk memeriksa koneksi database
@app.route('/api/status', methods=['GET'])
def api_status():
    try:
        # Coba query sederhana
        db.session.execute('SELECT 1')
        return jsonify({
            'success': True,
            'message': 'API is running and connected to the database'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Database connection error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)