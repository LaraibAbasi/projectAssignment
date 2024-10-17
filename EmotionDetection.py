import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fer import FER

# Load your trained model
model = load_model('FER_emotion_detector.h5')

# Initialize the FER emotion detector
emotion_detector = FER()

def preprocess_image(image):
    """Preprocess the image to match the input format of the model."""
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((48, 48))  # Resize to 48x48 pixels (assuming model was trained on 48x48 images)
    image = np.array(image)
    image = np.interp(image, (0, 255), (0, 1))  # Normalize to 0-1
    image = image.reshape(1, 48, 48, 3)  # Reshape for model input (batch size, height, width, channels)
    return image

def predict(image):
    """Run the model prediction on the preprocessed image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

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
    prediction = predict(image)
    
    # Assuming your model outputs probabilities for the classes
    # Define your emotion labels based on the model's output
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    # Find the predicted class (index of the highest probability)
    predicted_class = np.argmax(prediction[0])
    
    # Display the prediction result
    st.write(f"Detected Emotion: {emotion_labels[predicted_class]}")
    
    # Optionally, display the full prediction array
    st.write("Full prediction probabilities:", {emotion_labels[i]: prediction[0][i] for i in range(len(emotion_labels))})

