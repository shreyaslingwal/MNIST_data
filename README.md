# Feedforward Neural Network for MNIST Classification
This project implements a feedforward neural network with one hidden layer to classify handwritten digits from the MNIST dataset. The model is built using PyTorch and trained on a GPU-enabled environment.

Model Architecture
The neural network consists of the following layers:

Input Layer: Takes a flattened 28x28 pixel image (784 features).

Hidden Layer: A linear layer with 32 neurons, followed by a ReLU activation function.

Output Layer: A linear layer with 10 neurons, corresponding to the 10 digit classes (0-9).

Dataset
The model is trained and tested on the widely-used MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits. The dataset is preprocessed by converting the images to PyTorch tensors.

Training
The model was trained for a total of 10 epochs with the following details:

Loss Function: Cross-Entropy Loss

Optimizer: Stochastic Gradient Descent (SGD)

Learning Rate:

0.5 for the first 5 epochs

0.1 for the next 5 epochs

Batch Size: 128

Results
After 10 epochs of training, the model achieved the following accuracy on the test set:

Test Accuracy: 96.80%

Training History
Loss vs. Epochs

Accuracy vs. Epochs

Usage
To run this project, you will need to have the following libraries installed:

PyTorch

Torchvision

NumPy

Matplotlib

You can then run the Jupyter Notebook to train the model and evaluate its performance. The notebook is set up to use a GPU if available, which will significantly speed up the training process.
