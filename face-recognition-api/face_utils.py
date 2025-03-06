# face_utils.py
import face_recognition
import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageFont
import uuid

def get_face_encoding(image_path):
    """Mendapatkan encoding wajah dari gambar"""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        return None
    # Gunakan encoding wajah pertama yang ditemukan
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if face_encodings:
        return json.dumps(face_encodings[0].tolist())
    return None

def recognize_face(image_path, known_face_encodings, known_face_names):
    """Mengenali wajah dari gambar dengan membandingkan dengan wajah yang diketahui"""
    unknown_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(unknown_image)
    
    if not face_locations:
        return [], None
    
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    # Buat gambar hasil
    result_img = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(result_img)
    font = ImageFont.load_default()
    
    results = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Bandingkan dengan wajah yang diketahui
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Temukan kecocokan dengan jarak terkecil
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                user_id = best_match_index + 1  # Sesuaikan dengan ID sebenarnya jika perlu
            else:
                user_id = None
        else:
            user_id = None
        
        # Gambar kotak dan label
        top, right, bottom, left = int(top), int(right), int(bottom), int(left)
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        
        text_bbox = draw.textbbox((left, bottom), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), 
                      fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255), font=font)
        
        results.append({
            'location': (top, right, bottom, left),
            'name': name,
            'user_id': user_id,
            'recognized': name != "Unknown"
        })
    
    # Simpan gambar hasil
    result_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join('uploads', 'unknown', result_filename)
    result_img.save(result_path)
    
    return results, result_path