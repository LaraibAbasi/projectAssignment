import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from fer import FER
import cv2

# Load your trained model (if applicable)
model = tf.keras.models.load_model('FER_emotion_detector.pkl')  # Uncomment if using a model

# Initialize the FER emotion detector
emotion_detector = FER()

def predict_emotion(image):
    """Run emotion detection on the uploaded image."""
    # Convert the image to an array
    image_array = np.array(image)
    
    # Detect emotions
    emotion_data = emotion_detector.detect_emotions(image_array)
    
    if emotion_data:
        # Get the most probable emotion
        emotions = emotion_data[0]["emotions"]
        predicted_emotion = max(emotions, key=emotions.get)
        return predicted_emotion, emotions
    else:
        return None, None

# Streamlit UI
st.title("Emotion Detection App")

uploaded_file = st.file_uploader("Upload an image with a facial expression...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting emotion...")

    # Make a prediction on the uploaded image
    predicted_emotion, emotions = predict_emotion(image)
    
    if predicted_emotion:
        # Display the prediction result
        st.write(f"Detected Emotion: {predicted_emotion.capitalize()}")
        
        # Optionally, display the full emotion prediction results
        st.write("Emotion probabilities:", emotions)
    else:
        st.write("No emotion detected.")
