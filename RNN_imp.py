# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:40:15 2019

@author: Elliot
"""
import torch
from torch import nn

import numpy as np

'''layer_n = 3

stride = [1,2,3]
size = [1,3,5]

params = []
params.append(stride)
params.append(size)'''

#params is made up of legal combinations

#encode the params
def param_rep(params):
    rep = torch.zeros(len(params), 1, 3)
    for index, param in enumerate(params):
        #if param is valid then set 1
        for pos, val in enumerate(param):
            rep[index][0][pos] = 1
    return rep

class RNN_net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_net, self).__init__()
        #declare the hidden size for the network
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) #input to hidden layer
        self.i2o = nn.Linear(input_size + hidden_size, output_size) #input to output layer
        self.softmax = nn.LogSoftmax(dim = 1) #softmax for classification
    
    def forward(self, input_, hidden):
        combined = torch.cat((input_, hidden), 1) #concatenate tensors on column wise
        hidden = self.i2h(combined) #generate hidden representation
        output = self.i2o(combined) #generate output representation
        output = self.softmax(output) #get the softmax label
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

'''n_hidden = 128

#create a object of the class
n_params = len(stride)

#inputs are the potential param values, output is the chosen param value
net = RNN_net(n_params, n_hidden, n_params)

output = 0

hidden = net.init_hidden()
for layer in range(layer_n):
    for p in range(len(params)):
        rep = param_rep(params)
        print(rep)
        print("INPUT SIZE: ", str(len(rep)), str(len(hidden)))
        
        
        output, hidden = net(rep[p], hidden) #passing in the possible inputs
        index = torch.argmax(output)
        print("out index = ", index)'''
        
'''def ReinforcementLearningSearch():
    print("Using Randomized search")
    layers_params = list()
    
    layer_n = 3

    stride = [1,2,3]
    size = [1,3,5]
    
    params = []
    params.append(stride)
    params.append(size)
    
    
    n_hidden = 128

    #create a object of the class
    n_params = len(stride)
    
    #inputs are the potential param values, output is the chosen param value
    net = RNN_net(n_params, n_hidden, n_params)
    
    output = 0
    
    hidden = net.init_hidden()
    for layer in range(layer_n):
        for p in range(len(params)):
            rep = param_rep(params)
            print(rep)
            print("INPUT SIZE: ", str(len(rep)), str(len(hidden)))
            
            
            output, hidden = net(rep[p], hidden) #passing in the possible inputs
            index = torch.argmax(output)
            print("out index = ", index)
            
    
    return layers_params'''
    