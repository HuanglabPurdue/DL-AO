####################################################################################################
##Script for testing DL-AO network
##
##(C) Copyright 2022               The Huang Lab
##
##    All rights reserved           Weldon School of Biomedical Engineering
##                                  Purdue University
##                                  West Lafayette, Indiana
##                                  USA
##
##    Peiyi Zhang, Nov 2022
####################################################################################################
#import tapackages
import torch
import torch.nn as nn
import torch.nn.parallel
import os
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import argparse
from model import Net
from opts import options
from tqdm import tqdm
import h5py

#########################################
# define functions
def loaddata(opt):
    datadir = opt.datapath                                                                                     # directory that contain test data
    mat_contents = h5py.File(os.path.join(datadir, 'testdata.mat'),'r')                                        # load test data 
    dataset = torch.from_numpy(np.array(mat_contents.get('imsp'))).float()                                     # convert test data format
    if dataset.dim() < 4:                                                                                      # check test data format
        print('(Error) require 4D input. Plese modify your input to be: imsz x imsz x nchannel x datasize')    # error message
        exit()

    opt.datasize = dataset.size(0)                                                                             # number of sub-regions
    opt.imWidth = dataset.size(2)                                                                              # image width of sub-regions
    opt.imHeight = dataset.size(3)                                                                             # image height of sub-regions

    N = opt.datasize                                                                                           # number of sub-regions
    if N<opt.batchsize:
        opt.batchsize = N                                                                                      # batch size for testing
    
    print('test data size:', dataset.size(0), 'x', dataset.size(1), 'x', dataset.size(2), 'x', dataset.size(3))# print data dimension

    for i in range(N):
        for j in range(opt.channel):
            dataset[i][j]=dataset[i][j]/torch.max(dataset[i][j])                                               # normalize sub-regions

    return dataset

def test(model, opt, data):
    x = torch.Tensor(opt.batchsize, opt.channel, opt.imWidth, opt.imHeight).cuda()                             # Tensor for loading sub-regions                             
    model.eval()                                                                                               # set model in evaluation (inference) mode
    output = torch.Tensor()                                                                                    # Tensor for concatenating output
    with torch.no_grad():                                                                                      # deactivate autograd engine
        for i in tqdm(range(0, opt.datasize, opt.batchsize), ncols = 100, desc = "Testing (# of batches)"):    # loop through batches
            for ii in range(opt.batchsize):                                                                    # loop sub-regions in one batch
                if (i+ii) > opt.datasize - 1:
                    break
                else:
                    x[ii] = data[i+ii]
            xin = Variable(x)                                                                                  # Wrapper for using automatic differentiation (can be removed)
            out = model.forward(xin)                                                                           # forward propagation
            output = torch.cat([output, out.cpu()], 0)                                                         # concatenating outputs

    savepath = opt.save                                                                                        # path for saving results
    if not os.path.exists(savepath):
        os.makedirs(savepath)                                                                                  # create folder for saving results
    sio.savemat(os.path.join(savepath, 'result.mat'), {'aberration': output[0:opt.datasize].numpy()})          # save result
    print('result saved at: ', savepath)


def main():
    torch.manual_seed(12)                                                                                      # Sets the seed for generating random numbers
    opt = options()                                                                                            # Load user adjustable variables
    data = loaddata(opt)                                                                                       # Load testdata
    model = torch.load(opt.checkptname + '.pth')                                                               # Load model
    test(model, opt, data)                                                                                     # Test network

if __name__ == '__main__':
    main()                                                                                                     # Execute main()
