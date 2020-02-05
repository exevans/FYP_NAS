# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:02:52 2019

@author: Elliot
"""

import random

def RandomizedSearch(max_layers):
    print("Using Randomized search")
    
    layers_params = list()
    
    
    #make all the layers
    #initial input channel (colour channels)
    current_ouput_channels = 1
    #initialise input size
    input_size = 28
 
    for i in range(max_layers):
        layer_params, current_ouput_channels, input_size = GenerateLayer(i, current_ouput_channels, input_size)
        layers_params.append(layer_params)
    
    return layers_params

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
