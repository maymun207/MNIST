import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def filter_data(x, y):
    """Filter data to keep only labels 0, 1, and 2."""
    keep = (y == 0) | (y == 1) | (y == 2)
    x, y = x[keep], y[keep]
    return x, y

def main():
    print("Loading model...")
    model = tf.keras.models.load_model('mnist_model.keras')

    print("Loading and filtering data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test_orig = x_test.copy() # Keep original for plotting
    x_test, y_test = filter_data(x_test, y_test)
    x_test_norm = x_test / 255.0

    print("Generating predictions...")
    y_probs = model.predict(x_test_norm)
    y_pred = np.argmax(y_probs, axis=1)

    # 1. Generate Matplotlib grid
    print("Creating visualization...")
    num_samples = 10
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig = plt.figure(figsize=(12, 8))
    
    # Grid for images
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(3, 4, i + 1) # 3 rows, 4 cols (first 10)
        img = x_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

    # Confusion Matrix
    ax_cm = fig.add_subplot(3, 4, 11) # Place in the mix or separate? 
    # Actually, let's look at the layout. 3x4=12 slots. 10 images. 2 slots left.
    # I can put the confusion matrix in the last 2 slots or use a different layout.
    # Let's do a GridSpec or just subplots.
    # Let's do 2 rows of 5 images, and then a confusion matrix below.
    
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
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
                xticklabels=[0,1,2], yticklabels=[0,1,2])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion Matrix')

    plt.tight_layout()
    print("Saving artifact...")
    plt.savefig('performance_artifact.png')
    print("Done.")

if __name__ == '__main__':
    main()
