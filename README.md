# MNIST Digit Classification Dashboard

A complete machine learning project for classifying handwritten digits (0-9) using the MNIST dataset. Includes training scripts, evaluation tools, and a live Streamlit dashboard.

## Features

- **Full MNIST Support**: Classifies all digits from 0 to 9.
- **Interactive Dashboard**: Upload images (PNG, JPG, WEBP) and get real-time predictions.
- **Real-World Capability**: Includes an "Invert Colors" feature to handle black-on-white images (common in photos/scans) vs the model's native white-on-black format.
- **Visualization**: Generates confusion matrices and sample predictions.

## Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/maymun207/MNIST.git
    cd MNIST
    ```

2. **Create a virtual environment** (recommended):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Dashboard

Launch the web interface to test the model interactively:

```bash
streamlit run dashboard.py
```

Upload an image of a digit. If your image is black ink on white paper, check the **"Invert Image"** box.

### 2. Retrain the Model

To train the neural network from scratch:

```bash
python train_mnist.py
```

This will save the trained model to `mnist_model.keras`.

### 3. Evaluate Performance

Generate a performance report (confusion matrix and sample predictions):

```bash
python viz_mnist.py
```

The output will be saved as `performance_artifact.png`.

## Project Structure

- `dashboard.py`: Streamlit application source code.
- `train_mnist.py`: Training script for the main model (MLP).
- `viz_mnist.py`: Script to generate evaluation visualizations.
- `generate_test_image.py`: Utility to extract test images from the MNIST dataset.
- `mnist_model.keras`: The trained Keras model file.
- `train_CNN_Main.py`: Alternative training script using a CNN architecture (advanced).
