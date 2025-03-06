-- Membuat database
CREATE DATABASE IF NOT EXISTS face_recognition_db;

-- Menggunakan database
USE face_recognition_db;

-- Membuat tabel users
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    face_encoding LONGTEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);