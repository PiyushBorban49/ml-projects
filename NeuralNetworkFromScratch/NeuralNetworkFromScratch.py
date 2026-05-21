import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from sympy.stats.rv import probability

nnfs.init()
X,y = spiral_data(samples=100,classes=3)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,input):
        self.input = input
        self.output = np.dot(input,self.weights)+self.biases

    def backward(self,output_gradiant):
        self.dweights = np.dot(self.input.T,output_gradiant)
        self.dbiases = np.sum(output_gradiant,axis=0,keepdims=True)
        self.dinputs = np.dot(output_gradiant,self.weights.T)

class Activation_ReLu:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,output_gradiant):
        self.dinputs = output_gradiant.copy()
        self.dinputs[self.inputs<=0] = 0

class Activation_SoftMax:
    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilites = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilites

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_true,y_pred):
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped *y_true,axis=1)

        negative_log_liklihoods = -np.log(correct_confidences)
        return negative_log_liklihoods

    def backward(self,output_gradiant,y_true):
        samples = len(output_gradiant)
        labels = len(output_gradiant[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/output_gradiant
        self.dinputs = self.dinputs/samples

class Activation_SoftMax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,output_gradiant,y_true):
        samples = len(output_gradiant)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = output_gradiant.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs = self.dinputs/samples



plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()
