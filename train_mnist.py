import tensorflow as tf
import numpy as np
import os

def filter_data(x, y):
    """Filter data to keep only labels 0, 1, and 2."""
    keep = (y == 0) | (y == 1) | (y == 2)
    x, y = x[keep], y[keep]
    return x, y

def main():
    print("Loading data...")
    # Load data
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build model
    print("Building model for digits 0-9...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    print("Starting training...")
    model.fit(x_train, y_train, epochs=5)

    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save model
    print("Saving model to mnist_model.keras...")
    model.save('mnist_model.keras')

if __name__ == '__main__':
    main()
