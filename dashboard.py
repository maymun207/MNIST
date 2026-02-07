import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    # v2: Load 0-9 model
    return tf.keras.models.load_model('mnist_model.keras')

model = load_model()

st.title("MNIST Digit Classifier (0-9)")
st.write("Upload an image of a digit to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    # Display image at 25% width (using columns)
    # Display image at 25% width (using columns)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        invert = st.checkbox("Invert Image (Check this if your image is black text on white background)")
        if invert:
             # Invert only if it's grayscale (which it already is)
             from PIL import ImageOps
             image = ImageOps.invert(image)
             st.info("Image inverted.")

    # Better resizing with PIL to ensure 28x28
    image_resized = image.resize((28, 28))
    
    # Show what the model sees
    with col1:
        st.image(image_resized, caption='Model Input (28x28)', width=100)

    img_array = np.array(image_resized)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    if st.button('Classify', type="primary"):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Use columns for metrics
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Prediction", str(predicted_class))
        m_col2.metric("Confidence", f"{confidence:.2%}")
        
        st.subheader("Confidence Distribution")
        st.bar_chart(prediction[0])
