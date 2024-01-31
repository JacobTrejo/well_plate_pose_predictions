from ResNet_Blocks_3D_four_blocks import resnet18
#from CustomDataset2 import CustomImageDataset
#from CustomDataset2 import padding

#from CustomDataset import CustomImageDataset
#from CustomDataset import padding

from CustomDataset2 import CustomImageDataset
from CustomDataset2 import padding

import torch
import numpy as np
import cv2 as cv
import imageio
import pdb

import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

bgsub_im = imageio.imread('../well_plates_bgsub.png')
bgsub_im = np.asarray(bgsub_im)

cutOutFolderPath = '../cut_out_folder/'
cutOutResultsFolder = 'cut_out_results/'

resnetWeightsFolder = '../weights/resnet/'
resnetWeightsFolder = '../'
resnetWeightsFile = 'resnet_pose_best_python_230608_four_blocks.pt'
resnetWeightsFile = 'resnet_pose_best_python_230608_four_blocks.pt'
resnetWeights = resnetWeightsFolder + resnetWeightsFile
resnetWeights = '../../hardcodedWellsIntrinsic/Resnet/resnet_pose_best_python_230608_four_blocks.pt' 


red = [0,0,255]
green = [0,255,0]
blue = [255, 0, 0]

grid = np.load('grid.npy')

imageSizeX, imageSizeY = 960, 720
#modelPath = '../runs/pose/train3/weights/best.pt'
inputsFolder = 'inputs/'
outputsFolder = 'outputs/'

# resnet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnetModel = resnet18(1, 12, activation='leaky_relu').to(device)
resnetModel = nn.DataParallel(resnetModel)
resnetModel.load_state_dict(torch.load( resnetWeights  ))
resnetModel.eval()

n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
    nworkers = n_cuda*12
    pftch_factor = 2
else: 
    print('Cuda is not available. Training without GPUs. This might take long')
    nworkers = 4
    pftch_factor = 2
batch_size = 512*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)





#cut_out_list = []
#for idx, box in enumerate(boxes):
#    box = box.astype(int)
#    cutOut = get_cut_out(im, box)
#    cv.imwrite('cutOut' + str(idx) + '.png', cutOut)
#
#    cut_out_list.append(cutOut)

cut_out_list = []
files = os.listdir(cutOutFolderPath)
for filename in files:
    
    absPath = cutOutFolderPath + filename
    cutOut = imageio.imread( absPath )
    cutOut = np.asarray( cutOut )
    
    if filename == 'cutout37.png':
        cv.imwrite('ogCut.png', cutOut[...,0])
    
    cut_out_list.append( cutOut[...,0])

#cut_out_list = []
#for idx in range(10,19):
#    name = 'test/im_0000' + str(idx) + '.png'
#    cutout = imageio.imread(name)
#    cutout = np.asarray(cutout)
#    cut_out_list.append(cutout)

#transform = transforms.Compose([padding(), transforms.ToTensor(),  transforms.ConvertImageDtype(torch.float)])
#transform = transforms.Compose([padding(), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)  ])
transform = transforms.Compose([padding(), transforms.PILToTensor() ])

#datalist = os.listdir('test')
#data = CustomImageDataset( datalist  , transform = transform)
data = CustomImageDataset(cut_out_list, transform=transform)
loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)

keypointsList = []
for i, im in enumerate(loader):
    im = im.to(device)
    pose_recon = resnetModel(im)
    
    #pose_recon = pose_recon.detach().cpu().numpy()
    #im = np.squeeze(im.detach().cpu().numpy())
   
    pose_recon = pose_recon.detach().cpu().numpy()
    im = np.squeeze(im.cpu().detach().numpy())


    for imIdx in range(im.shape[0]):
        im1 = im[imIdx,...]
        im1 *= 255
        im1 = im1.astype(np.uint8)
        pt1 = pose_recon[imIdx,...]
        
        im2 = np.copy(im1)
        im2 = np.stack((im2, im2, im2), axis = 2)
        pt2 = np.copy(pt1)

        # Drawing them onto the cutouts
        #pt2 = pt1.astype(int)
        #im2[pt2[1,:10], pt2[0,:10]] = green
        #im2[pt2[1,10:], pt2[0,10:]] = red 
        #cv.imwrite( cutOutResultsFolder +  'cutOut' + str(imIdx) + '.png', im2)
        
        fileNum = int(files[imIdx][6 :-4])
        circ = grid[fileNum, ...]
        
        centerX, centerY, radius = circ
        smallY, smallX = centerY - radius, centerX - radius
        pt2[0,:] += smallX
        pt2[1,:] += smallY
        pt2 = pt2.astype(int)
        
        bgsub_im[pt2[1,:10], pt2[0,:10]] = green
        bgsub_im[pt2[1,10:], pt2[0,10:]] = red

        #cv.imwrite( cutOutResultsFolder + files[imIdx], im2)
        
        #print(files[imIdx])
        #pdb.set_trace()
        
        ## sending the pts to pts with respect to the cropped fish
        #nonZero = np.where( im1 > 0  )
        #sY = np.min( nonZero[0] )
        #sX = np.min( nonZero[1] )
        #pt1[0,:] -= sX
        #pt1[1,:] -= sY
        
        #sX, sY, bX, bY = boxes[ imIdx, ...]
        #pt1[0,:] += sX
        #pt1[1,:] += sY
        
        #keypointsList.append(pt1)

        #pt1 = pt1.astype(int)
        #np.save('im1.npy', im1)
        #pdb.set_trace()
        ##print('size ', im1.shape)
        #try:
        #    im1[pt1[1,:], pt1[0,:]] = 255
        #except:
        #    jh = 5
        #try:
        #    cv.imwrite('cutOut_' + str(imIdx)  + '.png',im1.astype(np.uint8) )
        #except:
cv.imwrite('result.png', bgsub_im)



