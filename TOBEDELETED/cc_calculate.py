from ultralytics import YOLO
from ResNet_Blocks_3D_four_blocks import resnet18
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
from scipy.io import savemat
from evaluation_functions import evaluate_prediction


#   Constants
red = [0,0,255]
green = [0,255,0]
blue = [255, 0, 0]

#   Inputs
imageSizeX, imageSizeY = 960, 720
#modelPath = '../runs/pose/train3/weights/best.pt'
inputsFolder = 'inputs/'
outputsFolder = 'outputs/'

YOLOWeightsFolder = '../weights/YOLO/'
YOLOWeightsFile = 'best.pt'
YOLOWeights = YOLOWeightsFolder + YOLOWeightsFile

resnetWeightsFolder = '../weights/resnet/'
resnetWeightsFile = 'resnet_pose_best_python_230608_four_blocks.pt'
resnetWeights = resnetWeightsFolder + resnetWeightsFile

#   Loading the Models
# Yolo model
model = YOLO(YOLOWeights)
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



def draw_boxes(im, boxArr, color):
    sX, sY, bX, bY = boxArr
    im[sY, sX:bX + 1, :] = color
    im[bY, sX:bX + 1, :] = color
    im[sY:bY + 1, sX, :] = color
    im[sY:bY + 1, bX, :] = color
    return im

def get_cut_out(im, boxArr):
    sX, sY, bX, bY = boxArr
    cutOut = im[sY:bY + 1, sX:bX + 1]
    return cutOut

def get_pose_from_frame(im):
    
    # NOTE: we have to give YOLO and rgb image, even though is was trained on grayscale images?
    rgbIm = np.stack((im, im, im), axis = 2)
    results = model( rgbIm, verbose = False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    keypointsList = results.keypoints.xy.cpu().numpy()

    im = np.stack([ im for _ in range(3)], axis = 2)
    ogim = np.copy(im)
    
    # Cutting out the fish based on the bounding boxes provided by YOLO
    cut_out_list = []
    for idx, box in enumerate(boxes):
        box = box.astype(int)
        cutOut = get_cut_out(im, box)

        cut_out_list.append(cutOut)

    # Placing the data onto a dataloader
    #transform = transforms.Compose([padding(), transforms.ToTensor(),  transforms.ConvertImageDtype(torch.float)])
    transform = transforms.Compose([padding(), transforms.PILToTensor() ])
    #data = CustomImageDataset( datalist  , transform = transform)
    data = CustomImageDataset(cut_out_list, transform=transform)
    loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)
    
    # Passing the data onto Resnet
    ccList = []
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
            #cv.imwrite('real_im.png', im1)
            cc = evaluate_prediction(im1, pt1)
            ccList.append(cc)
            #pdb.set_trace()
            # sending the pts to pts with respect to the cropped fish

            # Fix this part up, should try to get rid of using np.where
            nonZero = np.where( im1 > 0  )
            sY = np.min( nonZero[0] )
            sX = np.min( nonZero[1] )
            pt1[0,:] -= sX
            pt1[1,:] -= sY

            sX, sY, bX, bY = boxes[ imIdx, ...]
            pt1[0,:] += sX
            pt1[1,:] += sY

            keypointsList.append(pt1)


    return keypointsList, ccList

def drawKeypointsList(keypointsList, rgbIm):
    for keypoints in keypointsList:
        rgbIm[ keypoints[1,:10].astype(int), keypoints[0,:10].astype(int),:] = green
        rgbIm[ keypoints[1,10:12].astype(int), keypoints[0,10:12].astype(int),:] = red
    return rgbIm

# Initializing video capture object and getting parameters
vidcap = cv.VideoCapture('brafish_video_bgsub.avi')
fps = vidcap.get( cv.CAP_PROP_FPS )
height = vidcap.get( cv.CAP_PROP_FRAME_HEIGHT )
width = vidcap.get( cv.CAP_PROP_FRAME_WIDTH )
# Getting the output ready
#fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
output = cv.VideoWriter('output.avi', fourcc  , int(fps) ,(int(width) , int(height)))

# Iterating Through the frames of the video and getting the pose from them
success,image = vidcap.read()
full_cc_list = []
count = 0
while success:
    print('computing for frame: ', count)
    
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    #im_with_pose = get_pose_from_frame(image)
    keypointsList, ccList = get_pose_from_frame(image)
    full_cc_list = full_cc_list + ccList
    rgbIm = np.stack((image, image, image), axis = 2)
    rbgIm = drawKeypointsList(keypointsList, rgbIm)
    output.write( rgbIm )

    #cv.imwrite('frame0.png', im_with_pose)
    
    success,image = vidcap.read()
    count += 1
full_cc_list = np.array(full_cc_list)
np.save('full_cc_list_mod_intrinsic.npy', full_cc_list)
vidcap.release()
output.release()

# For writting the Video





