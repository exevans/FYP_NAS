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

import random
import math #various math operations

import torch.nn.functional as F #don't define our own loss functions
from torch import nn #neural net stuff
from torch import optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pdb #debugging

from torch.utils.tensorboard import SummaryWriter

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():  #don't download if already exists
        content = requests.get(URL + FILENAME).content #get content at the url
        (PATH / FILENAME).open("wb").write(content)  #write the data locally
        
#deserialize 
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")    #deserialize the data

#convert to tensor, numpy arrays are mapped to tensors
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
#print('train data X ',x_train, '\ntrain data Y ', y_train)
print('train shape ',x_train.shape)
print('test shape ',x_valid.shape)
print('y_min: ',y_train.min(), 'y_max: ',y_train.max())

bs = 64  # batch size
loss_func = F.cross_entropy
lr = 0.1
epochs = 2

# create the training set
train_ds = TensorDataset(x_train, y_train)

#for validating our results
valid_ds = TensorDataset(x_valid, y_valid)


def get_data(train_ds, valid_ds, bs):    
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
    
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

#Random_Search
SEARCH_STRAT = "Random_Search"
#Naive
PERFORMANCE_PREDICTOR = "Naive"

#define hyper-parameter ranges
layer_types = ["Convolution", "Pooling"]
hp_kernel_size = [1,3,5,7]
hp_padding_size = [0,1,3]
hp_filter_num = [16,24,36,48]
hp_stride_size = [1,2,3]


class NAS_CNN(nn.Module):
    def __init__(self, net_layers):
        print("val= " + str(net_layers))
        
        super().__init__()
        
        #for layer in net_layers:
            
        #self.lin = nn.Linear(784, 10)
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   #greyscale so only 1 channel 
        #self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        #return self.lin(xb)
        #print(xb.size())
        xb = xb.view(-1, 1, 28, 28) #change the shape of the tensor, -1 is unspecified uses 64 as that's the batch size
        print(xb.size())
        #print('testing a')
       # print(self.conv1(xb)[2])
        #print('testing b')
        #print(F.relu(self.conv1(xb))[2])
        
        xb = F.relu(self.conv1(xb))
        print(xb.size())
        xb = F.relu(self.conv2(xb))
        print(xb.size())
        xb = F.relu(self.conv3(xb))
        print(xb.size())
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):   #call the stored preprocess function
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)

#def getOp():
  #  op = random.sample(operations,1)
  #  print(op)
  #  return op
    
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)    #find which index has highest probability for each batch array of probabilities 
    return (preds == yb).float().mean()

def GenerateLayer(layer_idx, previous_ouput_channels, input_size):
    chosen_layer_params = dict()
    
    #get all the valid params to randomly select from, dict of param types each with possibilities
    valid_params = GetValidLayerParams(previous_ouput_channels, input_size)
    
    #get input channels
    chosen_layer_params["input_channel_num"] = random.choice(valid_params["input_channel_num"])
    #decide on layer type (conv/pooling)
    chosen_layer_params["layer_type"] = random.choice(valid_params["layer_type"])

    print("sizes: " + str(len(valid_params["conv_comb"])))
    con_com = random.choice(valid_params["conv_comb"])
    chosen_layer_params["kernel_size"] = con_com[0] #kernel size
    chosen_layer_params["padding_size"] = con_com[1] #padding
    chosen_layer_params["stride_size"] = con_com[2] #stride
    #chosen_layer_params["output_channel"] = random.choice(valid_params["output_channel"])
    
    #update so future layers now what input channels to use   
    if chosen_layer_params["layer_type"] == "Convolution":
        chosen_layer_params["output_channel"] = random.choice(valid_params["output_channel"])
        previous_ouput_channels =  chosen_layer_params.get("output_channel")
    else: #output channels will be same as they were previously
        chosen_layer_params["output_channel"] = previous_ouput_channels
    #else:   #for pooling need to calulate 
        #(conv_output_size[1] - pool_kernel_size)/pool_stride + 1
        #previous_ouput_channels =  (chosen_layer_params.get("input_channel_num") - chosen_layer_params["kernel_size"])/chosen_layer_params["stride_size"] +1
    
    #update input size to next layer
    input_size = ((input_size - con_com[0]+2*con_com[1]) / con_com[2])+1
    chosen_layer_params["output_size"] = input_size
    
    #log whats chosen
    print("layer " + str(layer_idx) + ": " + chosen_layer_params["layer_type"] + " : " + str(chosen_layer_params["input_channel_num"]) + " : " + str(chosen_layer_params["output_channel"]) + " kernel: " + str(chosen_layer_params["kernel_size"]) + " padd: " + str(chosen_layer_params["padding_size"]) + " stride: " + str(chosen_layer_params["stride_size"]) + " Output_size: " + str(chosen_layer_params["output_size"]))
    
    return chosen_layer_params, previous_ouput_channels, input_size

#returns the search space of all potential options
def GetValidLayerParams(previous_output_channels, input_size):
    validParams = dict()
    
    #input channels must equal previous output, no choice
    validParams["input_channel_num"] = [previous_output_channels]
    
    #layer types either convolutional or pooling
    if previous_output_channels == 1:
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
                 if (previous_output_channels - kernel_size) % stride_size !=0:
                     continue    #input_size - conv_kernel_size)/conv_stride + 1 must be an integer
                 elif padding_size >= (kernel_size/2):
                     continue #pad should be smaller than half of kernel size
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
    


def RandomizedSearch():
    print("Using Randomized search")
    
    layers_params = list()
    
    
    #make all the layers
    #initial input channel (colour channels)
    current_ouput_channels = 1
    #initialise input size
    input_size = 28
    max_layers = 3
    for i in range(max_layers):
        layer_params, current_ouput_channels, input_size = GenerateLayer(i, current_ouput_channels, input_size)
        layers_params.append(layer_params)
    
    return layers_params
    
#what network should we try next
def SelectNextNetwork():
    print("Select from the search space our new net using a search strategy")
    
    layers = list() #create the layers to be generated
    layer_params = dict()
    
    #use correct search strat
    if SEARCH_STRAT == "Random_Search":
        layer_params = RandomizedSearch()  #randomised search
        
    #build from layers
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
        else:
            id = "Pool"+str(i)
            layers.append((id, nn.MaxPool2d(kernel_size, stride=stride_size, padding=padding_size)))
            
        print(id + " " + str(input_channel_num) + " " + str(output_channel_num) + " " + str(kernel_size))
        #layers.append(("ReLu" + str(i), nn.ReLU()))
        
        if i == len(layer_params)-1:
            #transfoem to a list for the fc
            layers.append(("postprocess", Lambda(lambda x: x.view(x.size(0), -1))))
           
            #add on the fc layer so only 10 outputs are possible
            num_input_features =  params.get("output_size") * params.get("output_size")  * output_channel_num #size of output * filter
            num_output_classes = 10
            #print("estimated " + str((int)num_input_features))
            print("building FC" + " " + str(num_input_features) + " " + str(num_output_classes))
            layers.append(("FC", nn.Linear(int(num_input_features), num_output_classes)))
    '''else:
        layers.append(("conv1", nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)))
        layers.append(("ReLu1", nn.ReLU()))
        layers.append(("conv2", nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)))
        layers.append(("ReLu2", nn.ReLU()))
        layers.append(("con3", nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)))
        layers.append(("ReLu3", nn.ReLU()))
        layers.append(("Pool1", nn.AvgPool2d(4)))'''
    
    return layers
    
def FullyTrainAndTestNetwork(net, opt):
    #Begin Training
    print("Begin Training")
    fit(epochs, net, loss_func, opt, train_dl,valid_dl) #do the training
    print('Finished Training')

    #test
    test = 49998
    pred = net(x_train[test].cuda())
    #print(loss_func(pred, y_train[test]))
    print("test predicted ans: " + str(torch.argmax(pred)))
    
    #pyplot.imshow(x_train[test].reshape((28, 28)), cmap="gray")     #display the image, shaped correctly
    #test the network against the test set

    #calculate how well it performs on test set        
    correct = 0
    total = 0
    with torch.no_grad():
        for xb,yb in valid_dl:
            outputs = net(xb.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb.cuda()).sum().item()
    acc = correct/total
    loss = 0
    
    return (loss,acc)

def PredictNetworkPerformance(net, opt):
    print("Predict Performance")
    loss = 0
    acc = 0
    
    if PERFORMANCE_PREDICTOR == "Naive":
        loss, acc = FullyTrainAndTestNetwork(net, opt)
    print('loss: ', loss, 'Acc: ', acc)
    
    return (loss,acc)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def getModel(netLayers):
    net = nn.Sequential()
    net.add_module("preprocess", Lambda(preprocess))

    #add all layers
    for layer in netLayers:
         net.add_module(layer[0],layer[1])
         
    #net.add_module("postprocess", Lambda(lambda x: x.view(x.size(0), -1)))
        
    
    params = list(net.parameters())
    
    return net, optim.SGD(params=params, lr=lr,momentum=0.9)

def fit(epochs, net, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        net.train() #start training
        for xb,yb in train_dl:
            loss_batch(net, loss_func,xb.cuda(),yb.cuda(),opt)
            
        net.eval() #about to check against validation
        #print("DONE TRAINING")
        #with torch.no_grad():   #dont track operations in block
        #    valid_loss = sum(loss_func(net(xb.cuda()), yb.cuda()) for xb, yb in valid_dl)

        #print(epoch, valid_loss / len(valid_dl))

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main():
    print("Begin")
    print("cuda avialnabke: " + str(torch.cuda.is_available())) 
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_cached())
    #get all the training data to use
    #train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    
    #keep looping until satisfied
    #nets_tested = 0
    best_net = 0 #keep track of best nets
    best_net_acc = 0
    max_nets_to_test = 10
    net_results = list()
    
    #writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    # get some random training images
    #dataiter = iter(train_dl)
    #images, labels = dataiter.next()
    # create grid of images
    #img_grid = torchvision.utils.make_grid(images)
    # show images
    # matplotlib_imshow(img_grid, one_channel=True)
    # write to tensorboard
    #writer.add_image('four_fashion_mnist_images', img_grid)
   
    
    for nets_tested in range(max_nets_to_test):
        #Select the parameters for the network to test we want
        net, opt = getModel(SelectNextNetwork())
        #writer.add_graph(net, images)
        #writer.close()
        net.cuda()  #use gpu
        
        
        print("\nmade the selected net")
        print(net)
        
        #Predict how well this network will perform
        loss, acc = PredictNetworkPerformance(net, opt)
        print("Finished prediction\n")
        
        #add to results
        net_results.append(acc)
        
        #keep track of best net
        if acc > best_net_acc:
            best_net_acc = acc
            best_net = net
    
    print("Best found net")
    print(best_net)
    print(best_net_acc)
    print("\nAll Nets:")
    print(net_results)
    

    #clear gpu cache
    torch.cuda.empty_cache()
    
main()