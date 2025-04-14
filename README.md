# 🤟 Real-Time Indian Sign Language Detection with ONNX + OpenCV

A real-time computer vision project for detecting Indian Sign Language (ISL) gestures using a custom-trained Convolutional Neural Network (CNN), exported to ONNX format for efficient inference via OpenCV.

---

## 🚀 Overview

This project aims to detect hand gestures representing the **A-Z and 0-9 signs** in Indian Sign Language using your system's webcam.

- ✅ Real-time inference with OpenCV `dnn` module  
- ✅ Lightweight CNN model trained on a MediaPipe-enhanced dataset  
- ✅ Cross-platform ONNX model for fast deployment  
- ✅ Fully offline — No cloud inference needed

---

## 🧠 Model Details

- **Input**: 128x128 RGB images of hand gestures  
- **Architecture**: Custom CNN trained using TensorFlow  
- **Exported to**: `.onnx` format using `tf2onnx`  
- **Trained on**: [ISL A-Z Hand Gesture Dataset](https://www.kaggle.com/datasets/monishadhariya/indian-sign-language-isl-alphabet)  
- **Total Classes**: 36 (0-9, A-Z)

---

## 📂 Folder Structure

```
├── Model_ISL_Mark1.onnx       # Trained ONNX model
├── ISL OpenCV.py              # Real-time inference script
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/isl-onnx-realtime.git
cd isl-onnx-realtime
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: If you're using a headless server or get `cv2.imshow` errors, consider switching to `matplotlib` or running in a GUI-enabled terminal.

### 3. Run the Detection Script

```bash
python "ISL OpenCV.py"
```

Your webcam will open. Place your hand inside the ROI box and start signing!

---

## 🧾 Dependencies

- `opencv-python`
- `numpy`
- `onnxruntime` *(if you prefer ONNX Runtime over OpenCV `dnn`)*
- `matplotlib` *(optional, for fallback visualization)*

---

## 📸 Example Output

![Sample Screenshot](./assets/sample_prediction.png)  
*Above: The model correctly detecting the ISL gesture for the letter 'D'.*

---

## ✍️ Author

**Robin Robert**  
Final Year AIML Honors, Manipal University  
🎙️ Host of *The Voice* Podcast  
👨‍💻 AI Developer | Cinephile | Curious Mind  

---

## 📌 TODO / Future Work

- Add dynamic gesture recognition (LSTM + continuous signing)
- Deploy on Raspberry Pi or Jetson Nano
- Build a Streamlit web interface for demonstrations

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
