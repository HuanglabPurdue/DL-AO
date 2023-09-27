####################################################################################################
##Script for training DL-AO network
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
#import packages
import torch
import torch.nn as nn
import torch.nn.parallel
import os
import h5py
import numpy as np
from torch.autograd import Variable
import argparse
from opts import options
from model import Net
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
##########################################
# define functions
def loaddata(opt):
    dataset = torch.Tensor()
    label = torch.Tensor()
    for ff in range(1, opt.foldernum+1):                                                                                # loop for concatenating training data
        if opt.foldernum == 1 and ~os.path.exists(os.path.join(opt.datapath, str(ff), 'data.mat')):
            datadir = opt.datapath
        else:
            datadir = os.path.join(opt.datapath, str(ff))
        print('\nloading from: ', datadir)

        # load PSFs
        mat_contents = h5py.File(os.path.join(datadir, 'data.mat'),'r')                                                 # load training data
        temp = torch.from_numpy(np.array(mat_contents.get('imsp'))).float()                                             # convert data format
        dataset = torch.cat([dataset, temp],0)                                                                          # concatenate data
        if dataset.dim() < 4:                                                                                           # check test data format
#           daraset = dataset.unsqueeze(1)
            print('(Error) require 4D input. Plese modify your input to be: imsz x imsz x nchannel x datasize')         # error message
            exit()

        # load labels
        mat_contents_2 = sio.loadmat(os.path.join(datadir, 'label.mat'))                                                # load ground-truth labels
        temp2 = torch.from_numpy(mat_contents_2['label']).float()                                                       # convert data format
        label = torch.cat([label, temp2], 0)                                                                            # concatenate labels

    opt.datasize = dataset.size(0)                                                                                      # number of PSfs
    N = opt.datasize
    trsize = int(opt.trainSplit * opt.datasize)                                                                         # number of PSfs for training
    valsize = opt.datasize - trsize                                                                                     # number of PSfs for validation
    shuffle = torch.randperm(N)                                                                                         # generate index for shuffling data

    for i in range(N):
        for j in range(opt.channel):
            dataset[shuffle[i]][j] = dataset[shuffle[i]][j]/torch.max(dataset[shuffle[i]][j])                           # normalize sub-regions

    # split into train and validation data
    trdata = dataset[0:trsize]                                                                                          # training data
    valdata = dataset[trsize:N]                                                                                         # validation data
    trlabel = label[0:trsize, :]                                                                                        # ground-truth label for training
    vallabel = label[trsize:N, :]                                                                                       # ground-truth label for validation
    print('training data size:', trdata.size(0), 'x', trdata.size(1), 'x', trdata.size(2), 'x', trdata.size(3))
    print('training label size:', trlabel.size(0), 'x', trlabel.size(1))
    print('validation data size:', valdata.size(0), 'x', valdata.size(1), 'x', valdata.size(2), 'x', valdata.size(3))
    print('validation label size:', vallabel.size(0), 'x', vallabel.size(1))
    return {'trainData': trdata, 'trainLabel':trlabel}, {'valData': valdata, 'valLabel':vallabel}, trsize, valsize

def train(epoch, model, optimizer, opt, data, size):
    trainData = data['trainData']
    trainLabel = data['trainLabel']

    shuffle = torch.randperm(size)                                                                                      # generate index for shuffling training data
    count, totalErr, err = 0, 0, 0
    x = torch.Tensor(opt.batchsize, opt.channel, opt.imWidth, opt.imHeight).cuda()                                      # Tensor for loading sub-regions     
    yt = torch.Tensor(opt.batchsize, opt.labelsize).cuda()                                                              # Tensor for loading ground-truth labels 
    model.train()                                                                                                       # Sets model in training mode

    for i in tqdm(range(0, size, opt.batchsize), ncols = 100, desc = "Training (# of batches):"):                       # loop through batches
        for ii in range(opt.batchsize):                                                                                 # loop sub-regions in one batch
            if (i+ii) > size-1:
                break
            else:
                x[ii] = trainData[shuffle[i+ii]]
                yt[ii] = trainLabel[shuffle[i+ii]]

        optimizer.zero_grad()                                                                                           # set gradients to zero
        xin = Variable(x, requires_grad = True)                                                                         # Wrap input for using automatic differentiation
        ytout = Variable(yt)                                                                                            # Wrap label for using automatic differentiation
        out = model.forward(xin)                                                                                        # forward propagation

        loss = nn.MSELoss()                                                                                             # define loss function
        loss.cuda()
        err = loss(out, ytout)                                                                                          # calculate loss 

        err.backward()                                                                                                  # calculate gradient 
        optimizer.step()                                                                                                # update parameters 
        totalErr = totalErr + err.item()
        count += 1
    trainErr = totalErr/count
    print('training error:', trainErr, ' (A.U.)')
    return trainErr


def validate(epoch, model, opt, data, size):
    valData = data['valData']
    valLabel = data['valLabel']

    shuffle_val = torch.randperm(size)
    count, totalErr, err = 0, 0, 0
    x = torch.Tensor(opt.batchsize, opt.channel, opt.imWidth, opt.imHeight).cuda()                                      # Tensor for loading sub-regions    
    yt = torch.Tensor(opt.batchsize, opt.labelsize).cuda()                                                              # Tensor for loading ground-truth labels
    model.eval()                                                                                                        # Sets model in inference mode

    with torch.no_grad():                                                                                               # deactivate autograd engine
    	for i in tqdm(range(0, size, opt.batchsize), ncols = 100, desc = "Validating (# of batches):"):                 # loop through batches
            for ii in range(opt.batchsize):                                                                             # loop through sub-regions in one batch
                if (i+ii) > size-1:
                    break
                else:
                    x[ii] = valData[shuffle_val[i+ii]]
                    yt[ii] = valLabel[shuffle_val[i+ii]]
            xin = Variable(x)
            ytout = Variable(yt)
            out = model.forward(xin)                                                                                    # forward propagation
            loss = nn.MSELoss()                                                                                         # define loss function
            loss.cuda()
            err = loss(out, ytout)                                                                                      # calculate loss
            totalErr = totalErr + err.item()
            count += 1
    valErr = totalErr/count
    print('validation error:', valErr, ' (A.U.)')
    return valErr

def plot(opt, ep, tErr, vErr):                                                                                          # plot training and validation loss
    fig, ax = plt.subplots(1,1)
    plt.plot(ep, tErr, color = 'black', label='Train error')
    plt.plot(ep, vErr, color = 'red', label='validation error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    ax.set_yscale('log')
    plt.legend()
    plotname = os.path.join(opt.save, 'errorplot.png')
    plt.savefig(plotname)

def main():
    torch.manual_seed(12)                                                                                               # Sets the seed for generating random numbers

    opt = options()                                                                                                     #load arguments
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)                                                                                              # create folder for saving results
    
    trainset, valset, trsize, valsize = loaddata(opt)                                                                   # load data

    files = [f for f in os.listdir(opt.save) if f.endswith(".pth")]#
    if files == []:                                                                                                     # load model
        model = Net(opt.channel,opt.labelsize)
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.load(os.path.join(opt.save, 'model{}.pth'.format(opt.modelnum)))                                  # load pre-trained model
        opt.save = os.path.join(opt.save, str(opt.modelnum))    
        os.mkdir(opt.save)

    optimizer = torch.optim.Adam(model.parameters(), lr = opt.learningRate)                                             # define optimizer

    #train
    tErr, vErr, ep = [], [], []
    for epoch in range(int(opt.modelnum), opt.maxepoch):
        print('\nepoch:', epoch)
        trainErr = train(epoch, model, optimizer, opt, trainset, trsize)                                                # training
        valErr = validate(epoch, model, opt, valset, valsize)                                                           # validation

        ep.append(epoch)                                                                                                # concatenate epoch number
        tErr.append(trainErr)                                                                                           # concatenate training error
        vErr.append(valErr)                                                                                             # concatenate validation error
        plot(opt, ep, tErr, vErr)                                                                                       # plot training and validation error

        errname = os.path.join(opt.save, 'error.log')
        np.savetxt(errname, np.column_stack((tErr, vErr)))                                                              # save training and validation error

        if epoch % opt.savenum == 0:
            modelname =opt.save + '/model{}.pth'.format(epoch)
            torch.save(model, modelname)                                                                                # save trained models


        if epoch == 1:
            arg = [['arguments' + ': ', 'values']]
            for key in opt.__dict__:
                arg.append([key + ': ', opt.__dict__.get(key)])
            with open(os.path.join(opt.save, 'opt.txt'), 'w') as f:
                for item in arg:
                    f.write('%s\n' % item)

if __name__ == '__main__':
    main()                                                                                                              # execute main function
