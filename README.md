# GradScratch

A lightweight, educational implementation of neural networks and automatic differentiation from scratch. Built on top of Andrej Karpathy's micrograd, this project extends the core functionality with modern deep learning features like multiple activation functions, optimizers, and regularization techniques.

## Overview

GradScratch is an educational project that implements:
- Neural network architecture
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (MSE, Cross Entropy)
- Optimizers (SGD with momentum, Adam)
- Dropout regularization
- Batch processing support

## Features

### Activation Functions
- ReLU
- Sigmoid
- Tanh

### Loss Functions
- Mean Squared Error (MSE)
- Cross Entropy

### Optimizers
- Stochastic Gradient Descent (SGD) with momentum
- Adam optimizer

### Regularization
- Dropout

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kishan5111/gradscratch
cd gradscratch
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Run the tests:
```bash
python -m tests.test_engine
python -m tests.test_nn
```

## Usage

### Basic Neural Network
```python
from gradscratch.nn import MLP
from gradscratch.optim import SGD, Adam
from gradscratch.optim import mse_loss, cross_entropy_loss

# Create a neural network
model = MLP(2, [16, 16, 1], dropout_p=0.1)  # 2 input features, 2 hidden layers of 16 neurons, 1 output

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(x)
    loss = mse_loss(y_pred, y_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Credits

This project is built upon Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master) implementation. Extend his work by adding:
- Additional activation functions (Sigmoid, Tanh)
- Loss functions (MSE, Cross Entropy)
- Optimizers (SGD with momentum, Adam)
- Dropout regularization
- Enhanced neural network capabilities

## License

MIT License
