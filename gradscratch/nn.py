import random
import numpy as np
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class Dropout(Module):

    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training:
            return x
        mask = [Value(random.random() > self.p) for _ in range(len(x))]
        return [xi * mi / (1 - self.p) for xi, mi in zip(x, mask)]

class Neuron(Module):

    def __init__(self, nin, nonlin=True, activation='relu'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.activation = activation

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if not self.nonlin:
            return act
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'tanh':
            return act.tanh()
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize()}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, dropout_p=0.0):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.dropout:
                x = self.dropout(x)
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"