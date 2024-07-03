# Fruit-Classification-using-TensorFlow

## Project Overview

This project aims to classify images of fruits into three categories: apple, banana, and orange, using a convolutional neural network (CNN) implemented with TensorFlow. The model is trained on a dataset consisting of labeled fruit images, split into training, validation, and test sets for evaluation.

## Dataset

The dataset used in this project consists of images of apples, bananas, and oranges, collected from various sources. It is organized into the following directories:

- `fruits/train`: Training images for model training.
- `fruits/validation`: Validation images for model evaluation during training.
- `fruits/test`: Test images for final evaluation of the trained model.

## Model Architecture

The CNN model architecture comprises several convolutional layers followed by max-pooling layers for feature extraction. Here's a summary of the model architecture:

- Input layer: Rescales pixel values to the range [0, 1].
- Convolutional layers (with ReLU activation): Extracts features from input images.
- Max-pooling layers: Reduces spatial dimensions to capture important features.
- Flatten layer: Converts 2D feature maps into a 1D feature vector.
- Dense layers (fully connected): Neural network layers for classification.
- Output layer: Dense layer with softmax activation for multiclass classification.

## Training and Evaluation

The model is trained using the Adam optimizer with sparse categorical cross-entropy loss. Training progresses over multiple epochs, monitoring accuracy and loss metrics on both training and validation datasets.

After training, the model is evaluated on the test dataset to assess its performance on unseen data. Evaluation metrics include accuracy, providing insights into the model's ability to generalize to new fruit images.

## TensorFlow Lite Conversion

To deploy the model on resource-constrained devices, TensorFlow Lite conversion is performed. The converted TensorFlow Lite (`.tflite`) model retains the trained model's inference capabilities while optimizing for efficiency, suitable for deployment in mobile applications and embedded systems.

## Usage

### Requirements

- Python 3.x
- TensorFlow
- Matplotlib

### Steps to Run

1. Clone the repository and navigate to the project directory.
2. Prepare the dataset by organizing images into `fruits/train`, `fruits/validation`, and `fruits/test`.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run `python fruit_classification.py` to train and evaluate the model.
5. Convert the trained model to TensorFlow Lite using `python convert_to_tflite.py`.

### Example Output



## Conclusion

This project demonstrates the application of deep learning techniques for multiclass classification of fruits using TensorFlow. By leveraging CNNs and TensorFlow Lite, the model achieves efficient inference, making it suitable for deployment in real-world applications.

---

This README file provides a structured overview of your project, detailing its components, methodology, and outcomes. Adjust the technical details and structure as per your specific implementation and preferences.
