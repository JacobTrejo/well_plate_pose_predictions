
from programs.ResNet_Blocks_3D_four_blocks import resnet18
#from CustomDataset2 import CustomImageDataset
#from CustomDataset2 import padding

#from CustomDataset import CustomImageDataset
#from CustomDataset import padding

from programs.CustomDataset2 import CustomImageDataset
from programs.CustomDataset2 import padding

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

# INPUTS
videoFileName = 'brafish_video_bgsub.avi'
videoName = videoFileName[:-4]

# this variable determines where the wells are
# each row represents a circle with 3 variables
# they are arranged as centerX, centerY, radius
grid = np.load('inputs/grid.npy')



videoFolder = 'inputs/bgsub_videos/'
videoPath = videoFolder + videoFileName

videoOutputFolder = 'outputs/bgsub_videos/'
videoOutputPath = videoOutputFolder + videoName + '.avi'

#bgsub_im = imageio.imread('../well_plates_bgsub.png')
#bgsub_im = np.asarray(bgsub_im)

resnetWeights = 'weights/backgroundBlurred/resnet_pose_best_python_230608_four_blocks.pt'

red = [0,0,255]
green = [0,255,0]
blue = [255, 0, 0]

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

#if torch.cuda.device_count() > 1:
#  print("Using " + str(n_cuda) + " GPUs!")
#  model = nn.DataParallel(model)

# Getting the cutouts
#for circ in grid:
#    center = (int(circ[0]),int(circ[1]) )
#    radius = int(circ[2])
#
#    sX = center[0] - radius
#    bX = center[0] + radius
#    sY = center[1] - radius
#    bY = center[1] + radius
#    cutOut = result[sY:bY + 1, sX:bX + 1]

def superImpose(image):
    rgb = np.stack((image, image, image), axis = 2)
    
    global grid
    # getting the cutOuts
    cutOutList = []
    for circ in grid:
        center = (int(circ[0]),int(circ[1]) )
        radius = int(circ[2])
    
        sX = center[0] - radius
        bX = center[0] + radius
        sY = center[1] - radius
        bY = center[1] + radius
        cutOut = image[sY:bY + 1, sX:bX + 1]
        cutOutList.append(cutOut)
    
    # Prepping the data to give to resnet
    transform = transforms.Compose([padding(), transforms.PILToTensor() ])
    data = CustomImageDataset(cutOutList, transform=transform)
    loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)

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
            
            noFishThreshold = 10
            if np.max(pt1) < noFishThreshold: continue

            #pt1 = pt1.astype(int)
            #im1[pt1[1,:], pt1[0,:]] =  255
            #cv.imwrite('test.png', im1)
            #exit()
            

            # Fix this part up, should try to get rid of using np.where
            nonZero = np.where( im1 > 0  )
            sY = np.min( nonZero[0] )
            sX = np.min( nonZero[1] )
            pt1[0,:] -= sX
            pt1[1,:] -= sY
            
            circ = grid[imIdx]
            center = (int(circ[0]),int(circ[1]) )
            radius = int(circ[2])

            sX = center[0] - radius
            bX = center[0] + radius
            sY = center[1] - radius
            bY = center[1] + radius
            #sX, sY, bX, bY = boxes[ imIdx, ...]
            pt1[0,:] += sX
            pt1[1,:] += sY
            pt1 = pt1.astype(int)
            rgb[pt1[1,:10], pt1[0,:10]] = green
            rgb[pt1[1,10:], pt1[0,10:]] = red
    
    return rgb


# Initializing video capture object and getting parameters
vidcap = cv.VideoCapture(videoPath)
fps = vidcap.get( cv.CAP_PROP_FPS )
height = vidcap.get( cv.CAP_PROP_FRAME_HEIGHT )
width = vidcap.get( cv.CAP_PROP_FRAME_WIDTH )
# Getting the output ready
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
#output = cv.VideoWriter(videoOutputPath, fourcc  , int(fps) ,(int(width) , int(height)))
print(videoOutputPath)
# Iterating Through the frames of the video and getting the pose from them
success,image = vidcap.read()
full_cc_list = []
count = 0
imagesArray = np.zeros((100, 720, 960))
while success:
    print('computing for frame: ', count)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    imagesArray[count, ...] = image
    #image = superImpose(image)
    
    #keypointsList, ccList = get_pose_from_frame(image)
    #full_cc_list = full_cc_list + ccList
    #rgbIm = np.stack((image, image, image), axis = 2)
    #rbgIm = drawKeypointsList(keypointsList, rgbIm)
    #output.write( image )

    #cv.imwrite('frame0.png', im_with_pose)

    success,image = vidcap.read()
    count += 1
    if count >= 100: break
np.save('vidArray.npy', imagesArray)
vidcap.release()
#output.release()










