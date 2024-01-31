import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image
#import cv2
import torchvision.transforms as transforms
import pdb
from PIL import Image
import cv2 as cv

class CustomImageDataset(Dataset):
    def __init__(self, img_files_list, transform=None):
        self.img_files_list = img_files_list
        self.data_size = len(img_files_list)
        self.transform = transform
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.img_files_list[idx]
        #image = np.uint8(image[...,0])
        #image = image[...,0] 
        image = Image.fromarray(np.uint8(image))
        #print('Image: ', image.size)
        w, h = image.size
        
        image = self.transform(image)
        image = image.float()
        image /= 255
        image2 = np.squeeze( image.cpu().numpy())
        
        return image


class padding:
    def __call__(self, image):
        w, h = image.size
        w_buffer = 101 - w
        w_left = int(w_buffer/2)
        w_right = w_buffer - w_left
        w_buffer = 101 - h
        w_top = int(w_buffer/2)
        w_bottom = w_buffer - w_top
        padding = (w_left, w_top, w_right, w_bottom)
        pad_transform = transforms.Pad(padding)
        padded_image = pad_transform(image)
        
        return padded_image




