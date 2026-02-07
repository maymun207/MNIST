import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass
import math

# Load the models
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model('mnist_cnn.keras')
    ann = tf.keras.models.load_model('mnist_model.keras')
    return cnn, ann

model_cnn, model_ann = load_models()

st.title("MNIST Digit Classifier: CNN vs ANN")
st.write("Draw a digit or upload an image to compare model performance.")

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
    cy, cx = center_of_mass(temp_array)
    
    if math.isnan(cy) or math.isnan(cx):
        cy, cx = new_height / 2.0, new_width / 2.0

    # Target position: (14, 14)
    shift_x = 14 - cx
    shift_y = 14 - cy
    
    # Create final image
    final_image = Image.new('L', (28, 28), color=0)
    
    paste_x = int(round(shift_x))
    paste_y = int(round(shift_y))
    
    final_image.paste(image_resized, (paste_x, paste_y))
    return final_image


def display_prediction(header, model, input_data):
    """Helper to display prediction for a model"""
    st.subheader(header)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    c1, c2 = st.columns(2)
    c1.metric("Prediction", str(predicted_class))
    c2.metric("Confidence", f"{confidence:.2%}")
    st.bar_chart(prediction[0])

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
            
            # Prepare inputs
            # CNN needs (1, 28, 28, 1)
            img_cnn = np.expand_dims(img_array, axis=-1)
            img_cnn = np.expand_dims(img_cnn, axis=0)
            
            # ANN needs (1, 28, 28) - Flatten layer handles the rest, but input shape in training was (28,28)
            # Let's check train_mnist.py architecture. Flatten(input_shape=(28, 28)).
            # So input should be (28,28) -> batched (1, 28, 28).
            img_ann = np.expand_dims(img_array, axis=0)

            col_cnn, col_ann = st.columns(2)
            
            with col_cnn:
                display_prediction("CNN Model", model_cnn, img_cnn)
            
            with col_ann:
                display_prediction("ANN Model", model_ann, img_ann)


# --- Tab 2: Upload Image ---
with tab2:
    st.write("Upload an image file:")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        
        # Auto-invert logic
        img_array_temp = np.array(image)
        mean_brightness = np.mean(img_array_temp)
        is_light_bg = mean_brightness > 127
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Original Image', width=150)
            st.caption(f"Mean Brightness: {mean_brightness:.1f}")
        
        with col2:
            invert = st.checkbox("Invert Image", value=is_light_bg, help="Check this if the image has a white background (black text).")
            
            if is_light_bg:
                st.info("Detected light background. Auto-inverted.")
            
            if invert:
                 image = ImageOps.invert(image)
                 st.image(image, caption='Inverted Image', width=150)
        
        image_resized = image.resize((28, 28))

        img_array = np.array(image_resized)
        img_array = img_array.astype('float32') / 255.0
        
        # Prepare inputs
        img_cnn = np.expand_dims(img_array, axis=-1)
        img_cnn = np.expand_dims(img_cnn, axis=0)
        
        img_ann = np.expand_dims(img_array, axis=0)

        if st.button('Classify Upload', type="primary"):
            col_cnn, col_ann = st.columns(2)
            
            with col_cnn:
                display_prediction("CNN Model", model_cnn, img_cnn)
            
            with col_ann:
                display_prediction("ANN Model", model_ann, img_ann)

