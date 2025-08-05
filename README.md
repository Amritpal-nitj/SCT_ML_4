# SCT_ML_4
TASK4:Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human- computer interaction and gesture-based control systems.
# 🤖 Hand Gesture Recognition using CNN & OpenCV

This project demonstrates a real-time **hand gesture recognition system** using a **Convolutional Neural Network (CNN)** and **OpenCV**. The system is trained to recognize hand gestures like ✌️, 👊, 👍, etc., and classify them into categories.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)


---

## 🔍 Problem Statement

Build a system that recognizes hand gestures from images or webcam video in real-time and classifies them using deep learning.

---

## 🧠 Tech Stack

- Python 3
- TensorFlow / Keras
- OpenCV
- Matplotlib, NumPy, Seaborn
- Jupyter Notebook

---

## 📁 Dataset Structure

Your dataset should be organized as:

```
dataset/
├── 0/   # Gesture: fist
├── 1/   # Gesture: palm
├── 2/   # Gesture: peace
├── 3/   # Gesture: okay
├── 4/   # Gesture: thumbs up
```

Each folder contains images of a specific gesture.

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/Amritpal--nitj/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Install Requirements

```bash
pip install tensorflow opencv-python matplotlib seaborn
```

### 3. Run the Notebook

```bash
jupyter notebook Hand_Gesture_Recognition_Styled.ipynb
```

---

## 🧠 Model Summary

CNN Architecture:
- Conv2D + MaxPooling layers
- Flatten → Dense → Softmax
- Optimizer: Adam
- Loss: Categorical Crossentropy

Training runs for 10+ epochs with accuracy and loss plots.

---

## 📷 Real-Time Prediction (Webcam)

The final notebook cell uses OpenCV to capture video and classify gestures live:

```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prediction = model.predict(...)
cv2.putText(frame, label, ...)
```

Press `Esc` to exit webcam mode.

---

## 📊 Evaluation

- ✅ Accuracy: ~95% (depending on data)
- 📉 Loss & accuracy visualizations
- 🧪 Confusion matrix and classification report

---

## 📌 Use Cases

- Gesture-controlled user interfaces
- Human-computer interaction
- Robotics and automation
- Accessibility tools

---

## 💡 Future Improvements

- Add more gestures or custom user data
- Use background subtraction to reduce noise
- Deploy with Streamlit or Flask
- Export to mobile with TensorFlow Lite

---




