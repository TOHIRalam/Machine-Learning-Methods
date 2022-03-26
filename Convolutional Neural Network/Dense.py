# A dense layer is alyer that is deeply connected with its preceding layer. Each connection represents weight w[j][i]
# Every output value is computed as the sum of all the inputs multiplied by the weights connecting them to the specific output. 
# We additionally use a last term called bias which is also a trainable parameter, this is basically forward propagation for the dense layer. 

# Y(ouput) = W(weights, e.g. w[j][1], w[j][2] ... w[j][i]) * x(input, e.g. x[1], x[2] ... x[i]) + b(bias, e.g. b[1], b[2] ... b[j])

from turtle import forward
from layer import Layer 
import numpy as np 

class Dense(Layer): 
    # input_size: Numbers of neurons in the input, output_size: numbers of neurons in the output 
    def __init__(self, input_size, output_size):     
        # Initializes weights randomly and biases randomly                
        self.weights = np.random.randn(output_size, input_size)     
        self.bias = np.random.randn(output_size, 1)                

    # Using numpys dot product function this method computes Y = W * X + B 
    def forward(self, input):
        self.input = input 
        return np.dot(self.weights, self.input) + self.bias 

    def backward(self, output_gradient, learning_rate):
        # output_gradient: The derivative of the error with the respect to the biases 
        # Calculate the derivative of the error with the respect to the weights 
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Update the parameters with gradient descent 
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        # Return the derivative of the error with respect to the input.
        return np.dot(self.weights.T, output_gradient)