Digit Recognition with MNIST (CNN & Data Augmentation)

📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. It also incorporates data augmentation techniques to improve model generalization and robustness.

📂 Dataset

MNIST is a dataset of 70,000 grayscale images (28x28 pixels) of handwritten digits (0-9).

It consists of 60,000 training images and 10,000 test images.

The dataset is available in torchvision.datasets.MNIST.

📜 Model Architecture

The CNN model consists of the following layers:

Conv2D(32 filters, 3x3 kernel) + ReLU + MaxPooling(2x2)

Conv2D(64 filters, 3x3 kernel) + ReLU + MaxPooling(2x2)

Flatten layer

Fully Connected Layer (128 neurons) + ReLU

Fully Connected Layer (10 neurons, Softmax for classification)

🔄 Data Augmentation

To enhance model performance, we apply:

Random Rotation: Rotates images randomly within a given range.

Random Horizontal Flip: Flips images horizontally.

Random Affine Transformations: Applies scaling, translation, and shearing.
Implemented using torchvision.transforms:
    transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

🚀 Training the Model

Uses CrossEntropyLoss and Adam optimizer.

Trained for 10 epochs with batch size 64.

📊 Results

Achieved ~99% accuracy on the test set.

Model performance improves with data augmentation.
