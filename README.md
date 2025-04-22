# Multi-Object-Tracking-BoxMOT
# IP Camera Object Detection and Tracking

This project captures live video from an IP camera stream and performs real-time **object detection** using a pretrained Faster R-CNN model and **object tracking** with [BoT-SORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_OSNet). The results are displayed and saved to an output video.

---

## 🔧 Features

- Real-time video capture from an IP camera (RTSP stream)
- Object detection using PyTorch's `fasterrcnn_resnet50_fpn`
- Object tracking with BoT-SORT
- Multithreaded pipeline for smoother performance
- Output video saved as `.mp4`

---

## 🧠 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- torchvision
- numpy
- `boxmot` (BoT-SORT implementation)

---

## 📦 Installation

```bash
# Create environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision opencv-python numpy boxmot
