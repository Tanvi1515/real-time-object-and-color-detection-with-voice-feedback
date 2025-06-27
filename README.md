# 🎯 Real-Time Object & Color Detection with Voice Feedback

This project presents a real-time object and color-detection system integrated with voice feedback,designed to enhance human-computer interaction. 
Leveraging the YOLOv5 deep learning model, the system efficiently detects multiple objects in a video stream with high accuracy and low latency. 
Alongside object detection,a color recognition module identifies the dominant color of detected objects.
To provide an interactive experience, thesystem incorporates voice feedback and announces the object name, its precison percentage, its associated color and a short description of the object detected.

## 🚀 Features
- Real-time object detection using YOLO/OpenCV
- Color detection using pixel analysis
- Voice feedback 
- Live video capture via webcam

## 🛠️ Tech Stack
- Programming Language: Python
- Libraries and Framework:
  - OpenCV : for webcam video capture, frame manipulation, and display
  - NumPy : for numerical operations, pixel array reshaping, and color distance calculations
  - PIL (Pillow) : for handling image color analysis and conversion
- Deep Learning Model:
  -YOLOv5 from the ultralytics package – for real-time object detection (pre-trained on COCO  dataset)
  -PyTorch (torch) – underlying framework used by YOLOv5
- Text-to-Speech:
  - gTTS (Google Text-to-Speech) – converts object detection output into speech
  - playsound – for playing the generated audio file
  - tempfile & os – for temporary audio storage and cleanup
- Hardware Dependency:
  -Webcam / Camera – for real-time video input
  -CPU – model runs on the CPU (YOLOv5 is flexible, works without GPU too)
- Model:
  - YOLOv5s (Small) – fast and lightweight model suitable for real-time use cases
  
  

## 🗂️ Project Structure
├── proj2.py # Main logic for object and color detection with voice   
├── video_capture.py # Webcam integration for live feed

## 🎬 How It Works
1. The webcam starts capturing real-time video.
2. `proj2.py` detects objects and their primary color.
3. The system announces the object and its color through voice.

## ✍️ Author
Tanvi Prasad Chavan and Arya Prashant Ghadge
