import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, genome):
        idx = 0

        self.w1 = genome[idx:idx + input_size * hidden1_size].reshape((input_size, hidden1_size))
        idx += input_size * hidden1_size

        self.b1 = genome[idx:idx + hidden1_size]
        idx += hidden1_size

        self.w2 = genome[idx:idx + hidden1_size * hidden2_size].reshape((hidden1_size, hidden2_size))
        idx += hidden1_size * hidden2_size

        self.b2 = genome[idx:idx + hidden2_size]
        idx += hidden2_size

        self.w3 = genome[idx:idx + hidden2_size * output_size].reshape((hidden2_size, output_size))
        idx += hidden2_size * output_size

        self.b3 = genome[idx:idx + output_size]


    def forward(self, x):
        x = np.array(x)

        x = np.tanh(np.dot(x, self.w1) + self.b1)
        x = np.tanh(np.dot(x, self.w2) + self.b2)
        
        output = np.dot(x, self.w3) + self.b3

        return output
