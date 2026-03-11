#!/usr/bin/env python3
"""neural_net2 — Feedforward neural network with backprop from scratch. Zero deps."""
import math, random

def sigmoid(x): return 1 / (1 + math.exp(-max(-500, min(500, x))))
def sigmoid_d(x): s = sigmoid(x); return s * (1 - s)
def relu(x): return max(0, x)
def relu_d(x): return 1.0 if x > 0 else 0.0

class Layer:
    def __init__(self, n_in, n_out, activation='sigmoid'):
        self.weights = [[random.gauss(0, (2/n_in)**0.5) for _ in range(n_in)] for _ in range(n_out)]
        self.biases = [0.0] * n_out
        self.act = sigmoid if activation == 'sigmoid' else relu
        self.act_d = sigmoid_d if activation == 'sigmoid' else relu_d
        self.z = self.a = self.input = None

    def forward(self, x):
        self.input = x
        self.z = [sum(w*xi for w, xi in zip(ws, x)) + b for ws, b in zip(self.weights, self.biases)]
        self.a = [self.act(zi) for zi in self.z]
        return self.a

class NeuralNet:
    def __init__(self, *layer_sizes, lr=0.1):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x, target):
        output = self.forward(x)
        # Output layer deltas
        deltas = [(o - t) * layer.act_d(z) for o, t, z, layer in
                  zip(output, target, self.layers[-1].z, [self.layers[-1]]*len(output))]
        all_deltas = [deltas]
        # Backprop
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            deltas = [layer.act_d(layer.z[j]) *
                      sum(next_layer.weights[k][j] * all_deltas[-1][k] for k in range(len(all_deltas[-1])))
                      for j in range(len(layer.z))]
            all_deltas.append(deltas)
        all_deltas.reverse()
        # Update
        for layer, deltas in zip(self.layers, all_deltas):
            for j in range(len(layer.weights)):
                for k in range(len(layer.weights[j])):
                    layer.weights[j][k] -= self.lr * deltas[j] * layer.input[k]
                layer.biases[j] -= self.lr * deltas[j]
        return sum((o - t)**2 for o, t in zip(output, target)) / len(target)

def main():
    random.seed(42)
    nn = NeuralNet(2, 4, 1, lr=1.0)
    xor_data = [([0,0],[0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[0])]
    print("Training XOR network (2-4-1):")
    for epoch in range(2000):
        loss = sum(nn.train(x, y) for x, y in xor_data) / 4
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:>5}: loss={loss:.6f}")
    print("\nResults:")
    for x, y in xor_data:
        pred = nn.forward(x)
        print(f"  {x} -> {pred[0]:.4f} (expected {y[0]})")

if __name__ == "__main__":
    main()
