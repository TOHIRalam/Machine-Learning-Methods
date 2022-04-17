# CNN is a feed forward neural network that is generally used to analyze visual images by processing data with grid like topology.
# A CNN or Convolutional Neural Network is also known as ConvNet. Convolutional operation forms the basis of any CNN. 

# Activation Map: It is also known as feature map. A simple technique to get the discriminative image regions used by a CNN to 
# a specific class of the image. In other words, it lets us see which regions of the image is relevant to this class. 

# Activation Function: An Activation Function decides whether a neuron should be activated or not. This means that it will decide 
# whether the neuron's input to the network is important or not in the process of prediction using simpler mathematical operations.

################################################# Convolutional Neural Network Architecture ###################################################

# 1. Input Layer: It accepts the pixels of the image as input in the form of arrays. 
# 2. Convolutional Layer: This layer uses a matrix filter or kernel and performs convolutional operation to detect patterns in the image. 
# 3. ReLU(Rectified Linear Layer): ReLU activation function is applied to the convolution layer to get the rectified feature map. 
# 4. Pooling Layer: It uses multiple filters to detect edges, corners, eyes, feathers etc.
# 5. Fully connected layer: It finally identifies the object in the image.  

########################################################### Convolutional Layer ###############################################################
# The term 'Convolution' means the operation of multiplying pixel values by weight and summing them up
# A convolutional layer takes three dimensional blocks of data as inputs. The layer has trainable parameters, amongst them there are kernels. 
# Each kernels is also a three dimensional block that extends the full depth of the input. This layer may have multiple kernels all of which 
# has to extend the full depth of the input. With each kernel we associate a bias matrix that will have the same shape as the output. Finally, 
# this layer will have three dimensional block of data as its output. The depth of the output is same as the number of kernels we have. 
  
################################################### Convolutional Layer Implementation ########################################################
# Take each matrix in the first kernel and compute the cross-correlation with the input data. 
# Sum the three result and add up the first bias which will produce the output. 

import numpy as np 
from scipy import signal 
from layer import Layer

class Convolutional(Layer): 
    # input_shape: The tuple containning the height, width and depth of the input
    # kernel_size: Size of each matrix inside each kernel
    # depth: How many kernels we want means depth of the output 
    def __init__(self, input_shape, kernel_size, depth): 
        input_depth, input_height, input_width = input_shape
        self.depth = depth 
        self.input_shape = input_shape 
        self.input_depth = input_depth 
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return 