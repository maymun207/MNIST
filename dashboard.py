import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass
import math

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.keras')

model = load_model()

st.title("MNIST Digit Classifier (0-9)")
st.write("Draw a digit or upload an image to classify it.")

tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

def preprocess_digit(image):
    """
    Centers the digit in a 28x28 image using Center of Mass (CoM).
    1. Crop to bounding box.
    2. Resize to 20x20.
    3. Center by CoM in a 28x28 box.
    """
    # 1. Get bounding box of the non-zero (white) pixels
    bbox = image.getbbox()
    if bbox is None:
        return image.resize((28, 28))

    # 2. Crop to the bounding box
    image_cropped = image.crop(bbox)

    # 3. Resize to fit in a 20x20 box (preserve aspect ratio)
    old_width, old_height = image_cropped.size
    new_height, new_width = 20, 20
    
    # Scale to max dimension 20
    if old_width > old_height:
        scale = 20.0 / old_width
        new_height = int(old_height * scale)
    else:
        scale = 20.0 / old_height
        new_width = int(old_width * scale)
    
    image_resized = image_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 4. Center by Mass
    # Create a temporary canvas and paste resized image in the corner to calc CoM
    temp_image = Image.new('L', (28, 28), color=0)
    temp_image.paste(image_resized, (0, 0)) # Paste at top-left
    temp_array = np.array(temp_image)
    
    # Get Center of Mass of the resized image content
    # Note: ndimage.center_of_mass returns (row, col) -> (y, x)
    # We only care about the CoM of the content we just pasted.
    # Since we pasted at (0,0), the CoM of temp_array is the CoM relative to (0,0).
    cy, cx = center_of_mass(temp_array)
    
    if math.isnan(cy) or math.isnan(cx):
        # Fallback to geometric centering if CoM fails (empty image)
        cy, cx = new_height / 2.0, new_width / 2.0

    # We want the CoM (cx, cy) to be at the center of the 28x28 image (14, 14)
    # Target position: (14, 14)
    # Shift needed: 
    # new_center_x = 14
    # shift_x = new_center_x - cx
    # Paste coordinate (top-left) = 0 + shift_x (since we pasted at 0)
    
    shift_x = 14 - cx
    shift_y = 14 - cy
    
    # Create final image
    final_image = Image.new('L', (28, 28), color=0)
    
    # Paste coordinates (cast to int)
    # We are pasting the *resized* image, so its top-left is effectively shifted
    paste_x = int(round(shift_x))
    paste_y = int(round(shift_y))
    
    final_image.paste(image_resized, (paste_x, paste_y))
    return final_image


# --- Tab 1: Draw Digit ---
with tab1:
    st.write("Draw a digit (0-9) below:")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="black",  # Fixed fill color with some opacity
        stroke_width=30,
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
            
            # Apply preprocessing (Center & Crop)
            image_processed = preprocess_digit(image_gray)
            
            st.write("Model Input (Processed):")
            st.image(image_processed, width=100)
            
            # Convert to numpy array and normalize
            img_array = np.array(image_processed)
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

