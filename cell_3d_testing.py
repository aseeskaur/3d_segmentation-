#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:44:58 2025

@author: aseeskaur
"""

import numpy as np
import torch.nn as nn
from skimage import io
import torch
import sys
from pathlib import Path
from torch.utils.data import Dataset



images_dir = sys.argv[1]
masks_dir = sys.argv[2]

#images_path = Path("/Users/aseeskaur/Documents/Fluo-C3DH-A549/01")
#masks_path  = Path("/Users/aseeskaur/Documents/Fluo-C3DH-A549/01_GT/SEG")

images_path = Path(images_dir)
masks_path = Path(masks_dir)

image_files = images_path.glob("*.tif")
mask_files  = masks_path.glob("*.tif")


test_image_path   = images_path/"t028.tif"
test_mask_path    = masks_path/"man_seg028.tif"


test_image = io.imread(test_image_path)
test_image_swap = np.swapaxes(test_image, 0, 2)
test_mask = io.imread(test_mask_path)
test_mask_swap = np.swapaxes(test_mask, 0, 2)

test_image_padded = np.pad(test_image_swap, 40, mode = 'constant')
test_mask_padded = np.pad(test_mask_swap, 40, mode = 'constant')

test_image_nor = test_image_padded/test_image_padded.max()

class ToTensorAndNormalize3D:
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, image):
        # Convert the 3D numpy array to a tensor
        image_tensor = torch.from_numpy(image).float()

        if self.normalize:
            # Normalize to [0, 1] for float images
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor/image_tensor.max()  # Rescale to [0, 1]
        
        image_tensor = image_tensor.unsqueeze(0)  # Adding a channel dimension

        return image_tensor


class CellData3d(Dataset):
    def __init__(self, image_list, mask_list, transforms=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask = self.mask_list[index].astype(float)
        
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
            
        
        return image, mask


class Double_Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Double_Conv3D, self).__init__()

        self.conv = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(num_features=out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(num_features=out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class net(nn.Module):
    
    def __init__(self):
        super(net, self).__init__()
        
        # 3D max pooling layers
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(3, 3, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 3), stride=(2, 2, 1))
        
        # 3D convolution blocks
        self.conv1 = Double_Conv3D(1, 64)  
        self.conv2 = Double_Conv3D(64, 128)
        self.conv3 = Double_Conv3D(128, 256)
        self.conv4 = Double_Conv3D(256, 512)
        
        
        self.conv5 = nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        # Forward pass through the network
        
        x1 = self.conv1(image) 
        #print(x1.shape)
        x2 = self.maxpool(x1)  
        #print(x2.shape)
        
        x3 = self.conv2(x2)   
        #print(x3.shape)
        x4 = self.maxpool2(x3) 
        #print(x4.shape)
        
        x5 = self.conv3(x4)     
        #print(x5.shape)
        x6 = self.maxpool3(x5)  
        #print(x6.shape)
        
        x7 = self.conv4(x6)     
        #print(x7.shape)
        x8 = self.maxpool4(x7)  
        #print(x8.shape)

        x9 = self.conv5(x8)
        #print(x9.shape)
       
        return x9


model  = net()
#model_path = "/Users/aseeskaur/Documents/3d/best3d_model.pth"

model_path = "/home/akaur101/data/3d_data/checkpoints/best_model_3d_after_bal_swap_v2.pth"
checkpoint = torch.load(model_path, map_location="cuda")
model = model.to("cuda")
model.load_state_dict(checkpoint["model_state_dict"])
#model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

model.eval()

optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

test_image_tensor = torch.tensor(test_image_nor.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda") 
step_size = 3
# m = 80
# n = 10

X, Y, Z = test_image_nor.shape
print(X,Y,Z)
pred_output = np.zeros((X, Y, Z))  


with torch.no_grad():
    for i in range(40, X - 40 , step_size):
        #print("in i")
        for j in range(40, Y - 40 , step_size):
            #print("in j")
            for k in range(5, Z - 5  , step_size):
                #print("in k")
                
                patch = test_image_tensor[:, :, i-40:i+40, j-40:j+40, k-5:k+5] 
                output_patch = model(patch)  
                output_patch = torch.sigmoid(output_patch).squeeze().cpu().numpy()  
                pred_output[i-1:i+2, j-1:j+2, k-1:k+2] += output_patch

np.save("/home/akaur101/data/3d_data/pred_output_after_bal_swap_v2.npy", pred_output)



