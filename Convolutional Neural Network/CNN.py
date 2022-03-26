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

class Convolutional(Layer): 
    