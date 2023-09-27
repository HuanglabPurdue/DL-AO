####################################################################################################
##Script for user adjustable variables
##
##(C) Copyright 2022                The Huang Lab
##
##    All rights reserved           Weldon School of Biomedical Engineering
##                                  Purdue University
##                                  West Lafayette, Indiana
##                                  USA
##
##    Peiyi Zhang, Nov 2022
####################################################################################################

import argparse

#Command line options
def options():

    parser = argparse.ArgumentParser()

    #Training Related:
    parser.add_argument('--learningRate', type=float, default=1e-5) #learning rate
    parser.add_argument('--maxepoch', type=int, default=1000)       #maximum number of training iterations
    parser.add_argument('--savenum', type=int, default=1)           #save model per # iterations
    parser.add_argument('--plot', type=int, default=1)              #1: plot training/testing error; 0: no plot
    parser.add_argument('--modelnum', type=int, default=1)          #start to train from xx model (this is in case training stops in the middle)
    parser.add_argument('--trainSplit', type=float, default=0.99)   #percentage of training data
    parser.add_argument('--batchsize', type=int, default=128)       #batch size
    parser.add_argument('--datasize', type=int, default=100000)     #training and validation data size
    parser.add_argument('--foldernum', type=int, default=1)         #data number that you want to concatenate together (this is in case mat file is too large to save)

    #Testing Related
    parser.add_argument('--checkptname', type=str)                  #model name

    #Training and Testing Related
    parser.add_argument('--weighting', type=int, default=0)         #0: without CRLB weighting; 1: with CRLB weighting
    parser.add_argument('--datapath', type=str)                     #dataset location
    parser.add_argument('--labelsize', type=int, default=28)        #number of modes
    parser.add_argument('--save', type=str)                         #save trained model, error plot and result here
    parser.add_argument('--imHeight', type=int, default=32)         #image height
    parser.add_argument('--imWidth', type=int, default=32)          #image width
    parser.add_argument('--channel', type=int, default=2)           #2:biplane; 1:single plane
    parser.add_argument('--nGPU', type=int, default=4)              #number of GPUs to use

    opt, unknown = parser.parse_known_args()

    return opt
