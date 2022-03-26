
class Layer: 
    def __init__(self):
        self.input = None
        self.output = None 
    
    # Returns ouput 
    def forward(self, input): 
        pass 

    # Update parameters and return gradient 
    def backward(self, output_gradient, learning_rate): 
        pass 