import torch
import numpy as np
from gradscratch.engine import Value

def test_basic_operations():
    # Test addition
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch + b_torch
    c_torch.backward()
    
    assert np.isclose(c.data, c_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())
    assert np.isclose(b.grad, b_torch.grad.item())

def test_multiplication():
    # Test multiplication
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch * b_torch
    c_torch.backward()
    
    assert np.isclose(c.data, c_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())
    assert np.isclose(b.grad, b_torch.grad.item())

def test_power():
    # Test power operation
    a = Value(2.0)
    c = a ** 3
    c.backward()
    
    a_torch = torch.tensor(2.0, requires_grad=True)
    c_torch = a_torch ** 3
    c_torch.backward()
    
    assert np.isclose(c.data, c_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())

def test_activation_functions():
    # Test ReLU
    a = Value(-1.0)
    b = a.relu()
    b.backward()
    
    a_torch = torch.tensor(-1.0, requires_grad=True)
    b_torch = torch.relu(a_torch)
    b_torch.backward()
    
    assert np.isclose(b.data, b_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())
    
    # Test Sigmoid
    a = Value(0.0)
    b = a.sigmoid()
    b.backward()
    
    a_torch = torch.tensor(0.0, requires_grad=True)
    b_torch = torch.sigmoid(a_torch)
    b_torch.backward()
    
    assert np.isclose(b.data, b_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())
    
    # Test Tanh
    a = Value(0.0)
    b = a.tanh()
    b.backward()
    
    a_torch = torch.tensor(0.0, requires_grad=True)
    b_torch = torch.tanh(a_torch)
    b_torch.backward()
    
    assert np.isclose(b.data, b_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())

def test_complex_expression():
    # Test a more complex expression
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    d = (a * b + c).relu()
    d.backward()
    
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = torch.tensor(4.0, requires_grad=True)
    d_torch = torch.relu(a_torch * b_torch + c_torch)
    d_torch.backward()
    
    assert np.isclose(d.data, d_torch.item())
    assert np.isclose(a.grad, a_torch.grad.item())
    assert np.isclose(b.grad, b_torch.grad.item())
    assert np.isclose(c.grad, c_torch.grad.item())

if __name__ == "__main__":
    test_basic_operations()
    test_multiplication()
    test_power()
    test_activation_functions()
    test_complex_expression()
    print("All tests passed!") 