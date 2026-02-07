import tensorflow as tf
import numpy as np
import os

def main():
    print("Loading data...")
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape for CNN (28, 28, 1)
    print("Reshaping data for CNN...")
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN model
    print("Building CNN model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Train
    print("Starting training...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save model
    print("Saving model to mnist_cnn.keras...")
    model.save('mnist_cnn.keras')

if __name__ == '__main__':
    main()
