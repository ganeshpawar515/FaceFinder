# 🎓 Face Detection & Unique Face Extraction from Video/Live Feed(sem-3 mini-project)

## 📌 Project Overview

This project performs **real-time face detection and recognition** using a webcam or video input. It uses **MTCNN** for face detection and **InceptionResnetV1 (FaceNet)** for face embedding extraction. Unique faces (based on embedding distance) are saved to disk with minimal duplication, enabling smart face logging from live or recorded video.

---

## ✨ Features

- Detects faces in real-time or from video files
- Extracts **unique** faces based on embedding distance
- Uses GPU (if available) for faster processing
- Skips low-confidence detections for better accuracy
- Displays bounding boxes on live feed (optional)
- Saves cropped face images in a specified output directory

---

## 📁 Project Structure

