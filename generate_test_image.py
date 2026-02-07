import tensorflow as tf
from PIL import Image
import numpy as np

# Load data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Find a '2'
idx = np.where(y_train == 2)[0][0]
img_array = x_train[idx]

# Save as image
img = Image.fromarray(img_array)
img.save('test_digit_2.png')
print("Saved test_digit_2.png")
