# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:13:26 2019

@author: Elliot
"""

from pathlib import Path #use paths to store data
import requests #requests to ask to download info

import pickle #To serialise/deserialise dataset
import gzip #data zipped initially

import matplotlib.pyplot as plt
from matplotlib import pyplot #display the data images
import numpy as np #use numpy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Categorical
import random
from random import randrange
import math #various math operations

import torch.nn.functional as F #don't define our own loss functions
from torch import nn #neural net stuff
from torch import optim
from torch.autograd import Variable

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader #easily iterate through dataset

import pdb #debugging
import datetime #so we can name dumpfile
import re #regex to format filenames

#if we can use gpu, set it as the device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Cifar10_DataSet:
    def __init__(self):
        self.name = "CIFAR_10"
        self.colour_channel_num = 3
        self.image_size = 32
        self.batch_size = 100
        self.low_fidelity_iterations = 200
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    def GetDataSets(self): 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root="./" + str(PATH), train=True,
                                        download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="./" + str(PATH), train=False,
                                       download=True, transform=transform)
        
        return trainset, testset
    
class Mnist_DataSet:
    def __init__(self):
        self.name = "mnist"
        self.colour_channel_num = 1
        self.image_size = 28
        self.batch_size = 64
        self.low_fidelity_iterations = 50
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        
    def GetDataSets(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.MNIST(root="./" + str(PATH), train=True,
                                        download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root="./" + str(PATH), train=False,
                                       download=True, transform=transforms.ToTensor())
        
        return trainset, testset

#mnist
#dataset = Mnist_DataSet()
#CIFAR10
dataset = Cifar10_DataSet()


bs = dataset.batch_size  # batch size
#mnist or CIFAR_10
DATASET_NAME = dataset.name

DATA_PATH = Path("data")
PATH = DATA_PATH / DATASET_NAME

#create the needed directory
PATH.mkdir(parents=True, exist_ok=True)


def get_data(train_ds, valid_ds, bs):    
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

#retrieve the actual training and testing dataloaders from the datasets
train_ds, valid_ds = dataset.GetDataSets()
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def showDataSample(images_to_show_num):
    # get random training images to look at
    dataiter = iter(train_dl)
    images, labels = dataiter.next()
    # show the images
    imshow(torchvision.utils.make_grid(images[:images_to_show_num]))
    # print labels (only first 4)
    print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(images_to_show_num)))
 
showDataSample(4)

def createDumpFile():
    #use time but remove illegal characters
    time = str(datetime.datetime.now())
    time = re.sub("[\s\.:-]",'_',time)
    DumpFileName = "NetDump" + time + ".txt"
    
    DUMP_PATH = Path("net_dumps")
    DUMP_PATH.mkdir(parents=True, exist_ok=True)
    #make the path if it doesn't exist
    dumpFile = open((DUMP_PATH / DumpFileName).as_posix(), "w")
    dumpFile.write("Successful nets:\n")
    
    return dumpFile

#create a dumpfile
dumpFile = createDumpFile()

loss_func = F.cross_entropy
lr = 0.1
epochs = 2
max_layers = 4


COLOUR_CHANNEL_NUM = dataset.colour_channel_num
INPUT_IMAGE_SIZE = dataset.image_size

#Grid_Search
#SEARCH_STRAT = "Grid_Search"
#Random_Search
#SEARCH_STRAT = "Random_Search"
#Reinforcement
SEARCH_STRAT = "RL_Search"
#Naive
#PERFORMANCE_PREDICTOR = "Naive"
#low fidelity
PERFORMANCE_PREDICTOR = "Low_Fidelity"

LOW_FIDELITY_ITERATIONS = dataset.low_fidelity_iterations

#define hyper-parameter ranges
layer_types = ["Convolution", "Pooling_Max", "Pooling_Avg"]
hp_kernel_size = [1,3,5]#,7]
hp_padding_size = [0,1,3]
hp_filter_num = [16,24,36]#,48]
hp_stride_size = [1,2,3]

#keep track of the accuracy for every iteration of all produced architectures
architecture_list = dict()
architecture_list["iteration_list"] = []
architecture_list["accuracy_list"] = []

class StateSpace:
    
    def __init__(self):
        self.states = dict()
        self.state_count_ = 0

    def add_state(self, name, values):
        
        index_map = {}  #gets value from index
        value_map = {}  #gets index from value
        
        for i, val in enumerate(values):
            index_map[i] = val
            value_map[val] = i
            
        stateData = {
            'id'        : self.state_count_,
            'name'      : name,
            'values'    : values,
            'size'      : len(values),
            'index_map' : index_map,
            'value_map' : value_map
        }
        
        self.states[self.state_count_] = stateData
        
        self.state_count_ += 1
        
        return self.state_count_
    
    def encode(self, state_id, val):
        #get the chosen state
        state = self.states[state_id]
        state_size = state['size']
        value_map = state['value_map']
        #get index of the value to encode
        val_idx = value_map[val]
        
        #one hot encode
        one_hot_enc = np.zeros((1, state_size), dtype=np.float32)
        one_hot_enc[np.arange(1), val_idx] = val_idx + 1
        return one_hot_enc
    
    def get_state_val(self, state_id, state_val_id):
        #fix state_id from multiple layers
        state_id %= self.state_count_
        
        state = self.states[state_id]
        state_val = state['index_map'][state_val_id]
        
        return state_val
    
    def encode_random_states(self, layer_num):
        states = []
        for layer in range(layer_num):
            for state_id in range( self.state_count_):
                #get relevent state
                state =  self.states[state_id]
                size = state['size']
    
                #get random it from possibilities
                state_sel_idx = randrange(size)
                state_sel = state['index_map'][state_sel_idx]
                #now encode the selection
                state = self.encode(state_id, state_sel)
                states.append(state)
                
        return states
    
    def decode(self, enc_states):
        state_values = []
        for id, state_enc in enumerate(enc_states):
            state_val_idx = np.argmax(state_enc, axis=-1)[0]
            value = self.get_state_val(id, state_val_idx)
            state_values.append(value)

        return state_values
    
    def GenerateRandomLayerParams(self, layer_num, must_be_valid=False):
        while True:
            randomEncodedStates = self.encode_random_states(layer_num)
            
            #if must be valid then don't exit till it's valid
            if (not must_be_valid) or IsLayerParamsValid(self.GenerateLayerParams(randomEncodedStates)):
                return randomEncodedStates
        
    def GenerateLayerParams(self, state_index_values):
        print("Generate layer params")
        
        #if use_plain_decode:
        state_values = self.decode(state_index_values)
        #else:
         #   state_values = self.decodeFromIndex(state_index_values)
        
        print("state values = ", state_values) 
        layers_params = list()
        layer_params = dict()
        #keep track of the output channels of each layer (input channels to nect)
        previous_output_channels = COLOUR_CHANNEL_NUM
        #keep track of input size of each layer (output size of previous)
        input_size = INPUT_IMAGE_SIZE
        
        for i, val in enumerate(state_values):
            layer_id = int(i/self.state_count_)
            state = self[i]
            layer_params[state["name"]] = val
            #when we have read all states for a layer append to the list
            val = (i+1) % self.state_count_
            if val == 0:    #add in the "input_channel_num" as it isn't a state (depends on the output)
                #print("FinishLayer: ", i, " ", self.state_count_-1)
                layer_params["input_channel_num"] = previous_output_channels
                layer_params["input_size"] = input_size
                #add output size so we can set up FC correctly for final layer
                input_size = ((input_size - layer_params["kernel_size"]+2*layer_params["padding_size"]) / layer_params["stride_size"])+1
                layer_params["output_size"] = input_size
                
                #add this layer param to overall list
                layers_params.append(layer_params)
                previous_output_channels = layer_params["output_channel"]
                
                print("layer " + str(layer_id) + ": Input_size: " + str(layer_params["input_size"]) + " output__size: " + str(layer_params["output_size"]) + " : " + layer_params["layer_type"] + " : " + str(layer_params["input_channel_num"]) + " : " + str(layer_params["output_channel"]) + " kernel: " + str(layer_params["kernel_size"]) + " padd: " + str(layer_params["padding_size"]) + " stride: " + str(layer_params["stride_size"]))
                # create a new layer_params for next
                layer_params = dict() 
        
        return layers_params
    
    def dump_states(self):
        print("dumping StateSpace Content:")
        #print(self.states)
        for state in self.states:
            print(self.states[state])
            
    def __getitem__(self, id):
        return self.states[id % self.state_count_]
    
class ReinforcementSearchObj:
    def __init__(self):
        print("init reinforce search")
        self.stateSpace = StateSpace()
        
        #define the states
        self.stateSpace.add_state('layer_type', layer_types)
        self.stateSpace.add_state('kernel_size', hp_kernel_size)
        self.stateSpace.add_state('padding_size', hp_padding_size)
        self.stateSpace.add_state('stride_size', hp_stride_size)
        self.stateSpace.add_state('output_channel', hp_filter_num)
        #stateSpace.dump_states()
        
        self.controller = Controller(self.stateSpace)
        
        #initial state to use to create others from (use a valid network)
        self.state = self.stateSpace.GenerateRandomLayerParams(max_layers, must_be_valid=True)
            
        print("Found a valid initial net", self.stateSpace.decode(self.state), "\n")
        
    def SelectNextNetwork(self):
        print("Using reinforcement search")
        
        layers_params = list()
        
        #test encoding a value
        #print("Encode test = ", stateSpace.encode(1, 5))
        #test encoding a random net
        
        #print("initial state = ", self.state)
        #print("Decode test = ", self.stateSpace.decode(self.state))
        
        #get the next action from the controller pass in previous net (state)
        actions = self.controller.select_action(self.state)  # get an action for the previous state
        
        #print("Got the actions")
        #print(actions)
        
        #state is now set to first action
        self.state = actions
    
        #generate the architecture layer params from the actions
        #decode the produced states
        layers_params = self.stateSpace.GenerateLayerParams(actions)

        #check the layer params are valid otherwise no point building the net
        if not IsLayerParamsValid(layers_params):
            return False
    
        return layers_params
    
    def SetReward(self, reward):
        #add to our list of rewards
        self.controller.rewards.append(reward)
        #update the controller to make better changes in future
        self.controller.update_policy(reward)
        
        return 0

class Agent(nn.Module):

    def __init__(self, input_size, hidden_size, num_steps):
        super(Agent, self).__init__()
        #declare the hidden size for the network
        '''self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) #input to hidden layer
        self.i2o = nn.Linear(input_size + hidden_size, output_size) #input to output layer
        self.softmax = nn.LogSoftmax(dim = 1) #softmax for classification'''

        #self.lstm = nn.LSTMCell(input_size=3, hidden_size=hidden_size)#, num_layers=1, batch_first=True)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_size = hidden_size
        
        self.hidden = self.init_hidden()
        #self.h_t, self.c_t = self.hidden
    
    def forward(self, input, hidden):
        #outputs = []
        #self.h_t, self.c_t = self.hidden
        self.hidden = self.init_hidden()
        #for i in range(self.num_steps):
           # input_data = self.embedding(step_data)
        #input = input.view(1, 1, -1)
        #print("Feeding into rnn\n", input, "\n", hidden)#, "\nhidden:\n", hidden)
        #self.h_t, self.c_t = self.lstm(input, (self.h_t, self.c_t))
        
        output, hidden = self.rnn(input, hidden)
        
        #print("After lstm call:\n", self.h_t, " \n", self.c_t)
        output = self.softmax(output) 
        #output = self.fc(self.h_t)

        #output = h_t
           # Add drop out
           # h_t = self.drop(h_t)
           #output = self.decoder(h_t)
           #input = output
        #outputs += [output]

        #outputs = torch.stack(outputs).squeeze(1)
        
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(1, 1, self.hidden_size)
        #c_t = torch.zeros(1, self.hidden_size)

        return hidden
        #return (h_t, c_t)


class Controller:
    def __init__(self, stateSpace):
        print("create controller")
        
        self.stateSpace = stateSpace
        self.agent = Agent(3, 3, 3*5)
        self.rewards = []
        print("Agent:\n", self.agent)
        
        #set up the optimiser that will be adjusted as we get results
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.01)#optim.SGD(params=self.agent.parameters(), lr=lr,momentum=0.9)
        self.policy_history = Variable(torch.Tensor()) 
        
    def policy_network(self, state, max_layers):
        #state is first input of the previous network
        #other are fed in internally
        
        #print("policy_network input state ", state)
        #the first state will always be the same as previous and work out thenext ones
        currentState = state[0]
        
        actions = []
        
        #hardcode tha value of convolution
        actionVal = np.zeros((1, 3), dtype=np.float32)
        actionVal[np.arange(1), 0] = 1
        #add the first value
        actions.append(actionVal)
        
        #reset the hidden layer
        hidden = self.agent.init_hidden()
        # we provide a flat list of chained input-output to the RNN
        #list of hyper params for each layer
        for i in range((self.stateSpace.state_count_ * max_layers)-1):
            #get the state id independent of layer
            #state_id = i % self.stateSpace.state_count_
            #get the state for the current i
            #state_space = self.stateSpace[i]
            #size = state_space['size']
            
            #print("state i: ", currentState)
            inState_var = torch.from_numpy(currentState)#state[i])#.flatten())
            #print("input to the rnn", inState_var)
            #print("pass input into rnn")
            #print("input dims = ", inState_var.shape)
            
            #change shape
            inState_var = inState_var.view(1,1,-1)
            #print("input dims = ", inState_var.shape)
            
            action_probs, hidden = self.agent(torch.FloatTensor(inState_var), hidden)
            
            #remove blank space
            #action_probs2=action_probs.squeeze().detach().numpy()
            #action_idx = np.random.choice([0,1,2], p=action_probs2)
            #print("output of the rnn", output, len(output))
            
            # Add log probability of our chosen action to our history    
            #action_prob = torch.tensor(action_probs[action_idx])
            
            #print("action probs: ", action_probs)
            c = Categorical(action_probs)
            action = c.sample()
            action_idx = action.squeeze()
            #print("Using categorical: m ", c, " action: ", action, "action val: ", action_idx, "log prob: ",c.log_prob(action), ": dims: ",self.policy_history.dim(), " other dims: ", c.log_prob(action).dim())
            #print("action_prob: ", action_prob)
            
            #create encoding of the chosen action so it can be created into a net
            actionVal = np.zeros((1, 3), dtype=np.float32)
            actionVal[np.arange(1), action_idx] = action_idx + 1
            
            #if self.policy_history.dim() != 0:
             #   self.policy_history = torch.cat([self.policy_history, torch.tensor(c.log_prob(actionCat))])
                #self.policy_history = (c.log_prob(actionCat))
            #else:
            self.policy_history = c.log_prob(action)
            #self.policy_history.backward()
            
            #use retireved action as the input to find the next action
            currentState = actionVal
            
            #print(output)
            actions.append(actionVal)
        
        return actions

    def select_action(self, state):
        print("get action from controller to return the new architecture")
        #initial_state = self.stateSpace[0]
        #size = initial_state['size']

        #if state[0].shape != (1, size):
        #    state = state[0].reshape((1, size)).astype('int32')
        #else:
        #    state = state[0]

        #print("State input to Controller for Action : ", state, " -> ")

        #for l in range(layer_num):
         #   for i in range(state.state_count_):
                
        pred_actions = self.policy_network(state, max_layers)

        return pred_actions 
    
    def update_policy(self, reward):
        print("Learn from the reward: ", reward)
        
        #R = 0
        #policy_loss = []
        #returns = []
        #for all rewards
        #for r in policy.rewards[::-1]:
         #   R = r + args.gamma * R
          #  returns.insert(0, R)
        #returns = torch.tensor(returns)
        #returns = (returns - returns.mean()) / (returns.std() + eps)
        # Calculate loss
        #loss = (torch.sum(torch.mul(self.policy_history, Variable(reward)).mul(-1), -1))
        
        #clear gradients
        self.optimizer.zero_grad()  #clear all stored gradients
        #calculate loss
        loss = -self.policy_history * reward
        loss.backward()
        #update the optimizer
        self.optimizer.step()   #update the rnn parameters based on the gradient
        
        #clear the policy history
        self.policy_history = Variable(torch.Tensor())

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):   #call the stored preprocess function
        return self.func(x)

def preprocess(x):
    return x.view(-1, COLOUR_CHANNEL_NUM, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

#def getOp():
  #  op = random.sample(operations,1)
  #  print(op)
  #  return op
    
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)    #find which index has highest probability for each batch array of probabilities 
    return (preds == yb).float().mean()


#returns the search space of all potential options
def GetValidLayerParams(previous_output_channels, input_size):
    validParams = dict()
    
    #input channels must equal previous output, no choice
    validParams["input_channel_num"] = [previous_output_channels]
    
    #layer types either convolutional or pooling
    if previous_output_channels == COLOUR_CHANNEL_NUM:
        validParams["layer_type"] = ["Convolution"]
    else:
        validParams["layer_type"] = layer_types
    
    conv_comb = []
    #test combinations of kernel_size,padding,stride_size to return what's valid
    for kernel_size in hp_kernel_size:
         for padding_size in hp_padding_size:        
             for stride_size in hp_stride_size:
                 #print("Attempting to test kernel: " + str(kernel_size) + " padd: " + str(padding_size) + " stride " + str(stride_size) + "output= " + str((input_size - kernel_size+2*padding_size)%stride_size))
                 if (((input_size - kernel_size+2*padding_size)%stride_size)) !=0:    #need to consider actual size
                     continue
                 #if (previous_output_channels - kernel_size) % stride_size !=0: #
                    # continue    #input_size - conv_kernel_size)/conv_stride + 1 must be an integer
                 elif padding_size >= (kernel_size/2):
                     continue #pad should be smaller than half of kernel size
                 elif (((input_size - kernel_size+2*padding_size)/stride_size))+1 <=0:#output size must be > 0
                     continue
                 else: 
                     #found an acceptable match group combinations
                     conv_comb.append((kernel_size, padding_size, stride_size))
                     
      
    #check for issues
    if len(conv_comb) == 0:
        print("Issue, no valid conv combinations found for: input_channels= " + str(previous_output_channels))
                     
    #add the potential conv combinations
    validParams["conv_comb"] = conv_comb
    
    validParams["output_channel"] = hp_filter_num
    
    #calcTotal num of valid options
    paramNum = 0
    for param in validParams:
        #print(param + ": " + str(len(validParams[param])))
        if len(validParams[param]) > 0 :
            paramNum += len(validParams[param])
        else:
            paramNum += 1
            
    #print("Valid params nums: " + str(paramNum))
    
    #return all valid params
    return validParams
    
def IsLayerParamsValid(layers_params):        
    for layer_id, layer_param in enumerate(layers_params):    
        input_size = layer_param["input_size"]
        input_channel_num = layer_param["input_channel_num"]
        output_channel = layer_param["output_channel"]
        layer_type = layer_param["layer_type"]
        kernel_size = layer_param["kernel_size"]
        padding_size = layer_param["padding_size"]
        stride_size = layer_param["stride_size"]
        
        #check if it matches with valid params layer
        validParams = GetValidLayerParams(input_channel_num, input_size)
        if layer_param["input_channel_num"] not in validParams["input_channel_num"]:
            return False
        if layer_type not in validParams["layer_type"]:
            return False
        if (kernel_size, padding_size, stride_size) not in validParams["conv_comb"]:
            return False
        #can't change channel size if not a convolutional layer
        if layer_type != "Convolution" and input_channel_num != output_channel:
            print("Failed channel change on non Conv layer check")
            return False
    
    print("net passed: ")
    
    return True
   
class RandomSearchObj:
    def __init__(self):
        print("init random search")
        
    def SelectNextNetwork(self):
        layers_params = list()
    
        #make all the layers
        #initial input channel (colour channels)
        current_ouput_channels = COLOUR_CHANNEL_NUM
        #initialise input size
        input_size = INPUT_IMAGE_SIZE
     
        for i in range(max_layers):
            layer_params, current_ouput_channels, input_size = self.GenerateRandomLayer(i, current_ouput_channels, input_size)
            layers_params.append(layer_params)
        
        return layers_params
    
    def GenerateRandomLayer(self, layer_idx, previous_ouput_channels, input_size):
        chosen_layer_params = dict()
        
        #get all the valid params to randomly select from, dict of param types each with possibilities
        valid_params = GetValidLayerParams(previous_ouput_channels, input_size)
        
        #input_size
        chosen_layer_params["input_size"] = input_size
        #get input channels
        chosen_layer_params["input_channel_num"] = random.choice(valid_params["input_channel_num"])
        #decide on layer type (conv/pooling)
        chosen_layer_params["layer_type"] = random.choice(valid_params["layer_type"])
    
        print("sizes: " + str(len(valid_params["conv_comb"])))
        con_com = random.choice(valid_params["conv_comb"])
        chosen_layer_params["kernel_size"] = con_com[0] #kernel size
        chosen_layer_params["padding_size"] = con_com[1] #padding
        chosen_layer_params["stride_size"] = con_com[2] #stride
        
        #update so future layers now what input channels to use   
        if chosen_layer_params["layer_type"] == "Convolution":
            chosen_layer_params["output_channel"] = random.choice(valid_params["output_channel"])
            previous_ouput_channels =  chosen_layer_params.get("output_channel")
        else: #output channels will be same as they were previously
            chosen_layer_params["output_channel"] = previous_ouput_channels
       
        #update input size to next layer
        input_size = ((input_size - con_com[0]+2*con_com[1]) / con_com[2])+1
        chosen_layer_params["output_size"] = input_size
        
        
        #log whats chosen
        print("layer " + str(layer_idx) + ": " + chosen_layer_params["layer_type"] + " : " + str(chosen_layer_params["input_channel_num"]) + " : " + str(chosen_layer_params["output_channel"]) + " kernel: " + str(chosen_layer_params["kernel_size"]) + " padd: " + str(chosen_layer_params["padding_size"]) + " stride: " + str(chosen_layer_params["stride_size"]) + " Output_size: " + str(chosen_layer_params["output_size"]))
        
        return chosen_layer_params, previous_ouput_channels, input_size

class GridSearchObj:
    def __init__(self):
        print("init grid search")
        
        #keep track of which layer combinations have been selected as we traverse all posibilities
        self.grid_layer_to_inc = 0
    
        self.grid_search_layers_counters = []
        #init the variable counters to 0
        for i in range(max_layers):
            self.grid_search_layers_counters.append(dict())
            self.grid_search_layers_counters[i]["input_channel_num"] = 0
            self.grid_search_layers_counters[i]["layer_type"] = 0
            self.grid_search_layers_counters[i]["conv_comb"] = 0
            self.grid_search_layers_counters[i]["output_channel"] = 0
        
    def SelectNextNetwork(self):
        print("Using Grid search")
    
        layers_params = list()
        
        #make all the layers
        #initial input channel (colour channels)
        current_ouput_channels = COLOUR_CHANNEL_NUM
        #initialise input size
        input_size = INPUT_IMAGE_SIZE
     
        for i in range(max_layers):
            layer_params, current_ouput_channels, input_size = self.GenerateNextGridLayer(i, current_ouput_channels, input_size)
            layers_params.append(layer_params)
        
        return layers_params
    
    def IncrementGridSearchIterator(self, layer_idx, valid_params):
         #don't make changes to layers other than the one to increment
        if self.grid_layer_to_inc != layer_idx:
            return -1
    
        #assume unless told that we reset the layer to inc to 0
        self.grid_layer_to_inc = 0
        
        layer_counters = self.grid_search_layers_counters[layer_idx]
        #attempt io inc in order: input_channel/layer_type/conv_comb
        if (layer_counters["input_channel_num"] +1 ) >= len(valid_params["input_channel_num"]):
            layer_counters["input_channel_num"] = 0
            if (layer_counters["layer_type"] + 1) >= len(valid_params["layer_type"]):
                 layer_counters["layer_type"] = 0
                 if (layer_counters["conv_comb"] + 1) >= len(valid_params["conv_comb"]):
                     layer_counters["conv_comb"] = 0
                     #we have tried all posibilities for this layer so now inc next by 1
                     self.grid_layer_to_inc = layer_idx +1
                 else:
                    layer_counters["conv_comb"] += 1  
            else:
                layer_counters["layer_type"] += 1
        else:
            layer_counters["input_channel_num"] += 1 
            
        #return self.grid_layer_to_inc

    #selects the next option from the valid params
    def GenerateNextGridLayer(self, layer_idx, previous_ouput_channels, input_size):
        chosen_layer_params = dict()
        
        #get the possible combinations to select from
        valid_params = GetValidLayerParams(previous_ouput_channels, input_size)
        
        #how many posibilities for this layer are there
        layer_counters = self.grid_search_layers_counters[layer_idx]
        
        num_of_layer_pos = len(valid_params["layer_type"]) * len(valid_params["conv_comb"]) * len(valid_params["output_channel"])
        print("Number of possibilities for layer = ", num_of_layer_pos)
        print("Counter: ", layer_counters)
        #detect which of each potential to use from iterators to brute force all options
        chosen_layer_params["input_channel_num"] = valid_params["input_channel_num"][layer_counters["input_channel_num"]]
        chosen_layer_params["layer_type"]  = valid_params["layer_type"][layer_counters["layer_type"]]
        con_com = valid_params["conv_comb"][layer_counters["conv_comb"]]
        chosen_layer_params["kernel_size"] = con_com[0] #kernel size
        chosen_layer_params["padding_size"] = con_com[1] #padding
        chosen_layer_params["stride_size"] = con_com[2] #stride
        
        #update so future layers now what input channels to use   
        if chosen_layer_params["layer_type"] == "Convolution":
            chosen_layer_params["output_channel"] = valid_params["output_channel"][layer_counters["output_channel"]]
            previous_ouput_channels =  chosen_layer_params.get("output_channel")
        else: #output channels will be same as they were previously
            chosen_layer_params["output_channel"] = previous_ouput_channels
        
        #update input size to next layer
        input_size = ((input_size - con_com[0]+2*con_com[1]) / con_com[2])+1
        chosen_layer_params["output_size"] = input_size
        
        #log whats chosen
        print("layer " + str(layer_idx) + ": " + chosen_layer_params["layer_type"] + " : " + str(chosen_layer_params["input_channel_num"]) + " : " + str(chosen_layer_params["output_channel"]) + " kernel: " + str(chosen_layer_params["kernel_size"]) + " padd: " + str(chosen_layer_params["padding_size"]) + " stride: " + str(chosen_layer_params["stride_size"]) + " Output_size: " + str(chosen_layer_params["output_size"]))
            
        
        #increase the counters for the future but only if the layer should be inc
        self.IncrementGridSearchIterator(layer_idx, self.grid_layer_to_inc, valid_params)
    
        #print(grid_search_layers_counters[layer_idx])
    
        return chosen_layer_params, previous_ouput_channels, input_size

#from the chosen params for each layer actaully build layers of the network
def BuildNetworkFromParameters(layer_params):
    
    #before building check the net is valkidf
    
    layers = list()
    
    for i,params in enumerate(layer_params):
        print("Building layer " + str(i))
        id = 0
        layer_type = params.get("layer_type")
        input_channel_num = params.get("input_channel_num")
        output_channel_num = params.get("output_channel")
        kernel_size = params.get("kernel_size")
        padding_size = params.get("padding_size")
        stride_size = params.get("stride_size")
        if layer_type == "Convolution":
            id = "Cons"+str(i)
            print("attempt to add: " + str(input_channel_num))
            layers.append((id, nn.Conv2d(input_channel_num, output_channel_num, kernel_size=kernel_size, stride=stride_size, padding=padding_size)))
            layers.append(("ReLu" + str(i), nn.ReLU()))
        elif layer_type == "Pooling_Max":
            id = "Max_Pool"+str(i)
            layers.append((id, nn.MaxPool2d(kernel_size, stride=stride_size, padding=padding_size)))
        else:
            id = "Avg_Pool"+str(i)
            layers.append((id, nn.AvgPool2d(kernel_size, stride=stride_size, padding=padding_size)))
            
        print(id + " " + str(input_channel_num) + " " + str(output_channel_num) + " " + str(kernel_size))
        #layers.append(("ReLu" + str(i), nn.ReLU()))
        
        if i == len(layer_params)-1:
            #transfoem to a list for the fc
            layers.append(("postprocess", Lambda(lambda x: x.view(x.size(0), -1))))
           
            #add on the fc layer so only 10 outputs are possible
            num_input_features =  params.get("output_size") **2 * output_channel_num #size of output * filter
            num_output_classes = 10
            #print("estimated " + str((int)num_input_features))
            print("building FC" + " " + str(num_input_features) + " " + str(num_output_classes))
            layers.append(("FC", nn.Linear(int(num_input_features), num_output_classes)))

    return layers
    

def DrawGraph(scatter, x_data,y_data,x_lim,y_lim, xLabel = "Number of iterations", yLabel = "Accuracy", title = "Accuracy vs Number of iteration"): 
    if scatter:
        #ensure is not a tensor or we get a crash
        y_data = list(map(float, y_data)) 
        
        colors = (0,0,0)
        area = np.pi*3
        plt.scatter(x_data, y_data, s=area, c=colors, alpha=0.5)
    else:
        plt.plot(x_data,y_data,color = "red")
        
    #general methods
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.ylim(0, y_lim)
    plt.xlim(0, x_lim)
    plt.savefig('graph.png')
    plt.show()

def FullyTrainAndTestNetwork(net, opt, netId):
    #Begin Training
    low_fidelity_bool = PERFORMANCE_PREDICTOR == "Low_Fidelity"
    
    print("Begin Training: LowFidelity =", low_fidelity_bool)
    fit(epochs, net, loss_func, opt, train_dl,valid_dl, low_fidelity_bool) #do the training
    print('Finished Training')

    # visualization accuracy over time (learning curve)
    iteration_num = 1000
    if low_fidelity_bool:
        iteration_num = LOW_FIDELITY_ITERATIONS
 
    DrawGraph(False, architecture_list["iteration_list"], architecture_list["accuracy_list"], iteration_num,100, "Number of iterations", "Accuracy", "("+ str(netId) +") Accuracy Over Iterations")
    
    #reset the lists
    architecture_list["iteration_list"].clear()
    architecture_list["accuracy_list"].clear()

    #calculate how well it performs on test set overall
    acc = CalculatePerformance(net)
    loss = 0
    
    return (loss,acc)

def CalculatePerformance(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for xb2,yb2 in valid_dl:
            # Forward propagation
            outputs = net(xb2.to(device))
            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            #print("pred outputs: ", outputs.data)
            #print("predicted: ",predicted)
            #print("Ground: ", yb2)
            # Total number of labels
            total += yb2.size(0)
            correct += (predicted == yb2.to(device)).sum()
        
        print("Correct = ", correct, " / ", float(total))
        accuracy = correct / float(total)
        
        return accuracy

def PredictNetworkPerformance(net, opt, netId):
    print("Predict Performance")
    loss = 0
    acc = 0
    
    #get initial untrained performance
    accuracy = CalculatePerformance(net)          
    print("Iteration " + str(0) + " acc= " + str(accuracy*100))
    #store the iteratioon performance 
    architecture_list["iteration_list"].append(0)
    architecture_list["accuracy_list"].append(accuracy*100)
    
    #choose a method to predict the performance
    if PERFORMANCE_PREDICTOR == "Naive" or PERFORMANCE_PREDICTOR == "Low_Fidelity":
        loss, acc = FullyTrainAndTestNetwork(net, opt, netId)
        
    print('loss: ', loss, 'Acc: ', acc)

    
    return (loss,acc)

def loss_batch(model, loss_func, xb, yb, opt=None):
    
    loss = 0
    
    if opt is not None:
        opt.zero_grad()
        outputs = model(xb)
        #print("\nthe outputs of the model are:", outputs, "\n")
        #print("expected are ", yb)
        loss = loss_func(outputs, yb)
        loss.backward()
        opt.step()
        
        
        
    return loss.item()

def getModel(netLayers):
    net = nn.Sequential()
    net.add_module("preprocess", Lambda(preprocess))

    #add all layers
    for layer in netLayers:
        net.add_module(layer[0],layer[1])        
        
    params = list(net.parameters())
    
    return net, optim.SGD(params=params, lr=lr,momentum=0.9)

def fit(epochs, net, loss_func, opt, train_dl, valid_dl, low_fidelity):
    print("fit")
    iteration_count = 0 #used when storing data
    
    #actual training
    for epoch in range(epochs):
        
        running_loss = 0.0
        net.train() #start training
        for xb,yb in train_dl:
            loss = loss_batch(net, loss_func,xb.to(device),yb.to(device),opt)
            
            running_loss += loss
            
            #about to check against validation goes up to aroumd 750 for mnist
            iteration_count += 1
            plot_performance = True
            if (iteration_count < 100 and (iteration_count%10==0)) or (iteration_count % 50)==0:# and i < 300:
                if plot_performance:
                    accuracy = CalculatePerformance(net)
                    
                    print("Iteration " + str(iteration_count) + " acc= " + str(float(accuracy)*100) + " loss=" + str(running_loss))
                    #store the iteratioon performance 
                    architecture_list["iteration_list"].append(iteration_count)
                    architecture_list["accuracy_list"].append(accuracy*100)
                    
                    #reset the running loss
                    running_loss = 0.0
                else:
                     print("Iteration " + str(iteration_count))
                     
            #if we are doing a low fideliy estimate should we break?
            if low_fidelity and iteration_count > LOW_FIDELITY_ITERATIONS:
                break
            
def main():
    print("Begin")
    
    print("cuda avialnable: " + str(use_cuda)) 
    if use_cuda:
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_cached())
    
    #do the initial file logging
    dumpFile.write("Search strategy: " + str(SEARCH_STRAT) + "\n")
    dumpFile.write("Performance Predictor: " + str(PERFORMANCE_PREDICTOR) + "\n")
    
    #search obj will be defined based on search method selected
    SearchObj = 0
    
    #init the right search type
    if SEARCH_STRAT == "Grid_Search": 
        SearchObj = GridSearchObj()
    elif SEARCH_STRAT == "Random_Search":
        SearchObj = RandomSearchObj()  #randomised search
    elif SEARCH_STRAT == "RL_Search":
        SearchObj = ReinforcementSearchObj()  #reinforcement learning search    
    
    
    #keep looping until satisfied
    valid_nets = 0  #keep track of how many produced nets are actually valid (reinforcement learning)
    best_net = 0    #keep track of best nets
    best_net_acc = 0
    max_nets_to_test = 9999999999   #the maximum possible number of nets to explore (including invalid ones)
    valid_nets_to_test = 30
    net_values = list()
    net_results = list()
    
    
    for nets_tested in range(max_nets_to_test):
        
        #Select the parameters for the network to test we want
        layer_params = SearchObj.SelectNextNetwork()
        
        #layer_params is false if found net is invalid
        if layer_params == False:
            print("Net is Invalid\n")
             #update the reinforce controller
            if SEARCH_STRAT == "RL_Search":
                SearchObj.SetReward(0)
        else:
            #build the layers
            layers = BuildNetworkFromParameters(layer_params)
            #build the final net
            net, opt = getModel(layers)
            net.to(device)
            
            print("\nmade the selected net ID: ", valid_nets)
            print(net)
            
            #check that we havn't done this net already
            if net in net_values:
                print("this net has already been produced so don't test again\n")
            else:
                #Predict how well this network will perform
                loss, acc = PredictNetworkPerformance(net, opt, nets_tested)
                print("Finished prediction\n")
                
                #update the reinforce controller
                if SEARCH_STRAT == "RL_Search":
                    SearchObj.SetReward(acc*100)
                
                #add to results
                net_values.append(net)
                net_results.append(acc)
                
                #print to logs
                dumpFile.write(str(valid_nets) + "\n")
                dumpFile.write(str(net) + "\n")
                dumpFile.write(str(acc) + "\n")
                
                #keep track of best net
                if acc > best_net_acc:
                    best_net_acc = acc
                    best_net = net
                    
                #increment number of valid nets found
                valid_nets += 1
                
                #if we have seen enough valid nets then stop
                if (valid_nets >= valid_nets_to_test):
                    break
    
    print("Best found net")
    print(best_net)
    print(best_net_acc)
    print("\nAll Nets:")
    print(net_results)
    
    #output a graph to show net accuracy produced over time
    #for i, accuracy in enumerate(architecture_list["accuracy_list"]):
    DrawGraph(True, list(range(0,len(net_results))), net_results, len(net_results),1, "Net ID", "Accuracy", "Nets Produced over time")

    #clear gpu cache
    torch.cuda.empty_cache()
    
    #close our logging file
    dumpFile.close()
    
if __name__ == "__main__":
    main()