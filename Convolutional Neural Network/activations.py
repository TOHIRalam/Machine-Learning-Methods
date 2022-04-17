# An activation function maps a node's input to its corresponding output. Activation layer takes some input neurons and simply passes them through an activation function. 
from layer import Layer 
import numpy as np

class Activation(Layer): 
    # activation and it's derivative activation_prime both are functions 
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime 

    # Applies the activation to it's input 
    def forward(self, input): 
        self.input = input 
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate): 
        return np.multiply(output_gradient, self.activation_prime(self.input))