import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.keras')

model = load_model()

st.title("MNIST Digit Classifier (0-9)")
st.write("Draw a digit or upload an image to classify it.")

tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

# --- Tab 1: Draw Digit ---
with tab1:
    st.write("Draw a digit (0-9) below:")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="black",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        if st.button('Classify Drawing', type="primary"):
            # Get the image data from the canvas
            input_image = canvas_result.image_data.astype('uint8')
            
            # Convert to PIL Image and RGBA to Grayscale
            image = Image.fromarray(input_image)
            image_gray = image.convert('L') 
            
            # Resize to 28x28
            image_resized = image_gray.resize((28, 28))
            
            st.write("Model Input:")
            st.image(image_resized, width=100)
            
            # Convert to numpy array and normalize
            img_array = np.array(image_resized)
            img_array = img_array.astype('float32') / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0) 
            
            # Predict
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Prediction", str(predicted_class))
            m_col2.metric("Confidence", f"{confidence:.2%}")
            
            st.bar_chart(prediction[0])

# --- Tab 2: Upload Image ---
with tab2:
    st.write("Upload an image file:")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        
        # Auto-invert logic
        # Convert to numpy array to check mean brightness
        img_array_temp = np.array(image)
        mean_brightness = np.mean(img_array_temp)
        
        # If mean is high (bright), it's likely a white background
        is_light_bg = mean_brightness > 127
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Original Image', width=150)
            st.caption(f"Mean Brightness: {mean_brightness:.1f}")
        
        with col2:
            # Checkbox defaults to True if light background detected
            invert = st.checkbox("Invert Image", value=is_light_bg, help="Check this if the image has a white background (black text).")
            
            if is_light_bg:
                st.info("Detected light background. Auto-inverted.")
            
            if invert:
                 image = ImageOps.invert(image)
                 st.image(image, caption='Inverted Image', width=150)
        
        image_resized = image.resize((28, 28))

        img_array = np.array(image_resized)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button('Classify Upload', type="primary"):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Prediction", str(predicted_class))
            m_col2.metric("Confidence", f"{confidence:.2%}")
            
            st.bar_chart(prediction[0])

