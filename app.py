import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and face detector
model = load_model("face_emotion_model1.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("📸 Real-Time Face Emotion Detector")

# Step 1: Take a picture using webcam
img_data = st.camera_input("Take a photo")

if img_data:
    # Step 2: Convert to image
    image = Image.open(img_data).convert('RGB')
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi)
        emotion = labels[np.argmax(prediction)]

        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_array, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    st.image(img_array, caption="Detected Emotion", use_column_width=True)
