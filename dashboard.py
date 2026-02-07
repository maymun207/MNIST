import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.keras')

model = load_model()

st.title("MNIST Digit Classifier (0, 1, 2)")
st.write("Upload an image of a digit to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.resize(img_array, (28, 28)) # Resize if needed, though better to use PIL resize before array conv
    
    # Better resizing with PIL to ensure 28x28
    image_resized = image.resize((28, 28))
    img_array = np.array(image_resized)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    if st.button('Classify'):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.write(f"### Prediction: {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
        
        st.write("Confidence Distribution:")
        st.bar_chart(prediction[0])
