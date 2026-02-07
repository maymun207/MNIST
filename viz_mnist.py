import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def main():
    print("Loading model...")
    model = tf.keras.models.load_model('mnist_model.keras')

    print("Loading and filtering data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test_norm = x_test / 255.0

    print("Generating predictions...")
    y_probs = model.predict(x_test_norm)
    y_pred = np.argmax(y_probs, axis=1)

    # Generate Matplotlib grid
    print("Creating visualization...")
    num_samples = 10
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Layout: 2 rows of 5 images, and then a confusion matrix below.
    
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 5)

    # Images
    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5
        ax = fig.add_subplot(gs[row, col])
        
        img = x_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"T:{true_label} P:{pred_label}", color=color)

    # Confusion Matrix
    ax_cm = fig.add_subplot(gs[2, :]) # Take entire bottom row
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=range(10), yticklabels=range(10))
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion Matrix')

    plt.tight_layout()
    print("Saving artifact...")
    plt.savefig('performance_artifact.png')
    print("Done.")

if __name__ == '__main__':
    main()
