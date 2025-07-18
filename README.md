# 😊 Emotion Detector using CNN

A real-time facial emotion recognition system using a Convolutional Neural Network (CNN) trained on a dataset from [Kaggle](https://www.kaggle.com/vatsalvarshney), and deployed with OpenCV.

---

## 📁 Project Structure

emotion_detector/
│
├── models/
│ └── emotion_model.h5 # Saved CNN model after training
│
├── src/
│ └── cnn_model.py # CNN architecture and training script
│
├── app.py # Real-time emotion detection using webcam
├── README.md # You're here!
└── requirements.txt # Required Python packages

yaml
Copy
Edit

---

## 📦 Dataset

- Dataset used: **Facial Emotion Recognition (FER-2013)** or any similar dataset
- Download it from the [Kaggle profile of Vatsal Varshney](https://www.kaggle.com/vatsalvarshney).
- Unzip and place the dataset in an appropriate folder (`data/` or `dataset/`) as expected by `cnn_model.py`.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector
2. Install Dependencies
Make sure Python 3.8+ is installed. Then install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install tensorflow opencv-python numpy
🧠 Step 1: Train the CNN Model
Run the model training script:

bash
Copy
Edit
python src/cnn_model.py
This will train the model and save it as models/emotion_model.h5.

📷 Step 2: Run the Real-Time Emotion Detection
Once the model is saved, start the webcam-based app:

bash
Copy
Edit
python app.py
A webcam window will pop up.

It will detect faces and display the predicted emotion label in real time.

Press q to exit the app.

🧾 Requirements
Python 3.8+

TensorFlow 2.x

OpenCV

NumPy

Create a requirements.txt file:

text
Copy
Edit
tensorflow
opencv-python
numpy
⚠️ Common Issues
Model not found: Make sure emotion_model.h5 is saved under models/ after training.

Camera not opening: Ensure your webcam is connected and accessible.

Face cascade error: Add this to app.py:

python
Copy
Edit
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
👤 Author
Vatsal Varshney
🔗 Kaggle Profile
