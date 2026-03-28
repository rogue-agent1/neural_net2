#!/usr/bin/env python3
"""neural_net2 - Feedforward neural network (XOR demo)."""
import sys, random, math
def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
def sigmoid_d(x): return x*(1-x)
class NeuralNet:
    def __init__(self, layers):
        self.weights = []
        for i in range(len(layers)-1):
            w = [[random.gauss(0,1) for _ in range(layers[i]+1)] for _ in range(layers[i+1])]
            self.weights.append(w)
    def forward(self, x):
        self.activations = [x]
        for layer_w in self.weights:
            inp = self.activations[-1] + [1.0]
            out = [sigmoid(sum(w*i for w,i in zip(neuron, inp))) for neuron in layer_w]
            self.activations.append(out)
        return self.activations[-1]
    def train(self, X, Y, lr=0.5, epochs=5000):
        for epoch in range(epochs):
            total_err = 0
            for x, y in zip(X, Y):
                out = self.forward(x)
                errors = [y[i]-out[i] for i in range(len(y))]
                total_err += sum(e**2 for e in errors)
                deltas = [errors[i]*sigmoid_d(out[i]) for i in range(len(out))]
                for l in range(len(self.weights)-1, -1, -1):
                    inp = self.activations[l] + [1.0]
                    if l > 0:
                        new_deltas = []
                        for j in range(len(self.activations[l])):
                            err = sum(self.weights[l][k][j]*deltas[k] for k in range(len(deltas)))
                            new_deltas.append(err*sigmoid_d(self.activations[l][j]))
                    for k in range(len(deltas)):
                        for j in range(len(inp)):
                            self.weights[l][k][j] += lr*deltas[k]*inp[j]
                    if l > 0: deltas = new_deltas
            if epoch%1000==0: print(f"Epoch {epoch}: error={total_err:.6f}")
if __name__=="__main__":
    nn = NeuralNet([2, 4, 1])
    X = [[0,0],[0,1],[1,0],[1,1]]; Y = [[0],[1],[1],[0]]
    nn.train(X, Y, lr=1.0, epochs=10000)
    print("XOR results:")
    for x in X: print(f"  {x} -> {nn.forward(x)[0]:.4f}")
