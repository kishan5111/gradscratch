import torch
import numpy as np
from gradscratch.nn import MLP, Layer
from gradscratch.optim import SGD, Adam, mse_loss

def test_mlp_forward():
    # Create a simple MLP
    mlp = MLP(2, [3, 1], dropout_p=0.0)
    
    # Create equivalent PyTorch model
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1)
    )
    
    # Set the same weights
    with torch.no_grad():
        torch_mlp[0].weight.data = torch.tensor([[w.data for w in n.w] for n in mlp.layers[0].neurons], dtype=torch.float32)
        torch_mlp[0].bias.data = torch.tensor([n.b.data for n in mlp.layers[0].neurons], dtype=torch.float32)
        torch_mlp[2].weight.data = torch.tensor([[w.data for w in n.w] for n in mlp.layers[1].neurons], dtype=torch.float32)
        torch_mlp[2].bias.data = torch.tensor([n.b.data for n in mlp.layers[1].neurons], dtype=torch.float32)
    
    # Test forward pass
    x = [1.0, 2.0]
    x_torch = torch.tensor(x, dtype=torch.float32)
    
    y = mlp(x)
    y_torch = torch_mlp(x_torch)
    
    assert np.isclose(y.data, y_torch.item())

def test_mlp_backward():
    # Create a simple MLP
    mlp = MLP(2, [3, 1], dropout_p=0.0)
    
    # Create equivalent PyTorch model
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1)
    )
    
    # Set the same weights
    with torch.no_grad():
        torch_mlp[0].weight.data = torch.tensor([[w.data for w in n.w] for n in mlp.layers[0].neurons], dtype=torch.float32)
        torch_mlp[0].bias.data = torch.tensor([n.b.data for n in mlp.layers[0].neurons], dtype=torch.float32)
        torch_mlp[2].weight.data = torch.tensor([[w.data for w in n.w] for n in mlp.layers[1].neurons], dtype=torch.float32)
        torch_mlp[2].bias.data = torch.tensor([n.b.data for n in mlp.layers[1].neurons], dtype=torch.float32)
    
    # Test backward pass
    x = [1.0, 2.0]
    y_true = 0.5
    
    x_torch = torch.tensor(x, dtype=torch.float32)
    y_true_torch = torch.tensor(y_true, dtype=torch.float32)
    
    # Our implementation
    y_pred = mlp(x)
    loss = (y_pred - y_true) ** 2
    loss.backward()
    
    # PyTorch implementation
    y_pred_torch = torch_mlp(x_torch)
    loss_torch = (y_pred_torch - y_true_torch) ** 2
    loss_torch.backward()
    
    # Compare gradients
    for i, layer in enumerate(mlp.layers):
        for j, neuron in enumerate(layer.neurons):
            for k, w in enumerate(neuron.w):
                torch_grad = torch_mlp[i*2].weight.grad[j, k].item()
                assert np.isclose(w.grad, torch_grad, rtol=1e-4)
            torch_bias_grad = torch_mlp[i*2].bias.grad[j].item()
            assert np.isclose(neuron.b.grad, torch_bias_grad, rtol=1e-4)

def test_optimizer():
    # Create a simple MLP
    mlp = MLP(2, [3, 1], dropout_p=0.0)
    
    # Create equivalent PyTorch model
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1)
    )
    
    # Set the same weights
    with torch.no_grad():
        torch_mlp[0].weight.data = torch.tensor([[w.data for w in n.w] for n in mlp.layers[0].neurons], dtype=torch.float32)
        torch_mlp[0].bias.data = torch.tensor([n.b.data for n in mlp.layers[0].neurons], dtype=torch.float32)
        torch_mlp[2].weight.data = torch.tensor([[w.data for w in n.w] for n in mlp.layers[1].neurons], dtype=torch.float32)
        torch_mlp[2].bias.data = torch.tensor([n.b.data for n in mlp.layers[1].neurons], dtype=torch.float32)
    
    # Create optimizers
    optimizer = SGD(mlp.parameters(), lr=0.01)
    torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=0.01)
    
    # Test one optimization step
    x = [1.0, 2.0]
    y_true = 0.5
    
    x_torch = torch.tensor(x, dtype=torch.float32)
    y_true_torch = torch.tensor(y_true, dtype=torch.float32)
    
    # Our implementation
    y_pred = mlp(x)
    loss = (y_pred - y_true) ** 2
    loss.backward()
    optimizer.step()
    
    # PyTorch implementation
    y_pred_torch = torch_mlp(x_torch)
    loss_torch = (y_pred_torch - y_true_torch) ** 2
    loss_torch.backward()
    torch_optimizer.step()
    
    # Compare updated weights
    for i, layer in enumerate(mlp.layers):
        for j, neuron in enumerate(layer.neurons):
            for k, w in enumerate(neuron.w):
                torch_weight = torch_mlp[i*2].weight.data[j, k].item()
                assert np.isclose(w.data, torch_weight, rtol=1e-4)
            torch_bias = torch_mlp[i*2].bias.data[j].item()
            assert np.isclose(neuron.b.data, torch_bias, rtol=1e-4)

if __name__ == "__main__":
    test_mlp_forward()
    test_mlp_backward()
    test_optimizer()
    print("All neural network tests passed!") 