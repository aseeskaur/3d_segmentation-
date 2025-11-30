import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import tifffile
import imagecodecs

from mpl_toolkits.mplot3d import Axes3D


import torch
import random

from pathlib import Path
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from PIL import Image
from typing import List
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Normalize, ToTensor
import sys

#images_path = Path("/Users/aseeskaur/Documents/Fluo-C3DH-A549/01")
#masks_path  = Path("/Users/aseeskaur/Documents/Fluo-C3DH-A549/01_GT/SEG")


#image_files = images_path.glob("*.tif")
#mask_files  = masks_path.glob("*.tif")

images_dir = sys.argv[1]
masks_dir = sys.argv[2]

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

test_image_padded = np.pad(test_image_swap, 
                          pad_width=((40, 40), (40, 40), (5, 5)),
                          mode='constant')
test_mask_padded = np.pad(test_mask_swap, 
                         pad_width=((40, 40), (40, 40), (5, 5)),
                         mode='constant')

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

#trying to get 50 pixels from foreground to see of this is working"
#num = 50

# centroid_df = pd.DataFrame()
# foreground_coords = np.argwhere(test_mask_padded > 0)
# selected_coords = foreground_coords[np.random.choice(len(foreground_coords), num, replace=False)]
# centroid_df = pd.DataFrame(selected_coords, columns=['cord_x', 'cord_y', 'cord_z'])



num = 5000

np.random.seed(30)

# Randomly select voxel positions from unpadded image
total_voxels = test_image_swap.size
selected_indices = np.random.choice(total_voxels, num, replace=False)
coords = np.unravel_index(selected_indices, test_image_swap.shape)

centroid_df = pd.DataFrame({
    'cord_x': coords[0],
    'cord_y': coords[1],
    'cord_z': coords[2]
})



centroid_df['cord_x']= centroid_df['cord_x']+40
centroid_df['cord_y']= centroid_df['cord_y']+40
centroid_df['cord_z']= centroid_df['cord_z']+5



# ===== DEBUG SECTION =====
print(f"\n{'='*60}")
print(f"{'='*60}")
print(f"Total centroids in centroid_df: {len(centroid_df)}")
print(f"\nSample centroids (first 5):")
print(centroid_df.head())

# Check if centroids are on foreground in ground truth
test_values = test_mask_padded[
    centroid_df['cord_x'].values, 
    centroid_df['cord_y'].values, 
    centroid_df['cord_z'].values
]
print(f"\n centroids on foreground (GT): {(test_values > 0).sum()}/{len(centroid_df)}")
print(f"Percentage on foreground: {(test_values > 0).sum()/len(centroid_df)*100:.1f}%")

print(f"{'='*60}\n")
# ===== END DEBUG SECTION =====




def snaps_at_next_cent(next_cents, padded_image, padded_mask):
    image_list = []
    mask_list  = []
    z_cords = []
    y_cords = []
    x_cords = []

    
    for i in next_cents:
        cord_x = i[0]
        cord_y = i[1]
        cord_z = i[2]
        
        snap_image = padded_image[cord_x-40:cord_x+40, cord_y-40:cord_y+40, cord_z-5:cord_z+5]
        snap_mask  = padded_mask[cord_x-1:cord_x+2, cord_y-1:cord_y+2, cord_z-1:cord_z+2]
        
        image_list.append(snap_image)
        mask_list.append(snap_mask)

        x_cords.append(cord_x)
        y_cords.append(cord_y)
        z_cords.append(cord_z)
       
    
    return image_list, mask_list, x_cords, y_cords, z_cords

class CellData3d(Dataset):
    def __init__(self, image_list, mask_list, x_cords, y_cords, z_cords, transforms=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.x_cords = x_cords
        self.y_cords = y_cords
        self.z_cords = z_cords
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask = self.mask_list[index].astype(float)
        x = self.x_cords[index]
        y = self.y_cords[index]
        z = self.z_cords[index]
        
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return image, mask, x, y, z

transform = ToTensorAndNormalize3D(normalize=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model  = net()
model_path = "/home/akaur101/data/3d_data/checkpoints/best_3d_model_fixed_padding.pth"

checkpoint = torch.load(model_path, map_location=device)

# Load only the model weights
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

probability_map = np.zeros(test_image_padded.shape, dtype=np.float32)
count_map = np.zeros(test_image_padded.shape, dtype=np.int32)

S = set()
runn_cent = []
next_S = []

df = centroid_df
all_pred = []
all_indices = []

for i in df.index:
    cord_x = df['cord_x'][i]
    cord_y = df['cord_y'][i]
    cord_z = df['cord_z'][i]
    point = (cord_x, cord_y, cord_z)
    next_S.append([cord_x, cord_y, cord_z])
    S.add(point) 
    
counter = 0

while len(next_S) > 0:
    counter+=1
    print(f"Iteration {counter}, processing {len(next_S)} centroids")
    test_image_list, test_mask_list, x_cords, y_cords, z_cords = snaps_at_next_cent(next_S, test_image_padded, test_mask_padded)

    test_dataset = CellData3d(test_image_list, test_mask_list, x_cords, y_cords, z_cords, transforms=transform)
    test_loader = DataLoader(test_dataset, 50, shuffle=False)

    next_S = []
    
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            images, masks, x_cords, y_cords, z_cords = data
            images = images.to(device = device)
            #print(images.shape)
            masks = masks.to(device = device)
            #print(masks.shape)
            pred = model(images.float())
            #print("Before sigmoid prediction: ", pred)
            final_pred = torch.sigmoid(pred)
            #print("Final Prediction : " ,final_pred)
            

            pred_numpy = final_pred.cpu().numpy()
            
            for b in range(len(x_cords)):
                x_center = x_cords[b].item()
                y_center = y_cords[b].item()
                z_center = z_cords[b].item()
                
                pred_cube = pred_numpy[b, 0]  # 3×3×3 predictions
                
                for i in range(3):
                    for j in range(3):
                        for k_idx in range(3):  # renamed from k to avoid conflict
                            x = x_center - 1 + i
                            y = y_center - 1 + j
                            z = z_center - 1 + k_idx
                            
                            probability_map[x, y, z] += pred_cube[i, j, k_idx]
                            count_map[x, y, z] += 1
            
            pred_temp = torch.where(torch.squeeze(final_pred) < 0.1, 0, 1) #thresholding
            if len(final_pred) == 1:
                pred_temp = torch.where((final_pred) < 0.1, 0, 1)
            
            #print("Pred_temp: ", pred_temp)

            pred_ind = torch.argwhere(pred_temp)

            #print("Pred_ind : ",  pred_ind)


            temp_ind_list = []
            for i in range(len(pred_ind)):
                current = pred_ind[i][0].item()

                x_item = x_cords[current].item()
                pred_ind_x = pred_ind[i][1].item()
                x = x_item - 1 + pred_ind_x

                y_item = y_cords[current].item()
                pred_ind_y = pred_ind[i][2].item()
                y = y_item - 1 + pred_ind_y
                
                z_item = z_cords[current].item()
                pred_ind_z = pred_ind[i][3].item()
                z = z_item - 1 + pred_ind_z

                
        
                #boundry check
                if (40 <= x < test_image_padded.shape[0] -40 and 
                    40 <= y < test_image_padded.shape[1] - 40 and
                    40 <= z < test_image_padded.shape[2] - 5):

                    point = (x,y,z)
                    #print("point: ", point)
                    if point not in S:
                        S.add(point)
                        #print("S: ", S)
                        next_S.append([x,y,z])
                        runn_cent.append([x,y,z])

print(f"\n{'='*60}")
print(f"Region growing finished!")
print(f"{'='*60}")
print(f"Iterations: {counter}")
print(f"Total centroids processed: {len(runn_cent)}")
print(f"Total voxels in set S: {len(S)}")

#Average the accumulated predictions
print("\nAveraging overlapping predictions...")
averaged_prob = np.divide(probability_map, count_map, 
                          where=count_map > 0, 
                          out=np.zeros_like(probability_map))

print(f"Voxels with predictions: {(count_map > 0).sum()}")
print(f"Max overlaps per voxel: {count_map.max()}")

#Remove padding to get back to original image size
print("\nRemoving padding...")
final_prob = averaged_prob[40:-40, 40:-40, 5:-5]

print(f"Final probability map shape: {final_prob.shape}")
print(f"Original image shape: {test_image_swap.shape}")
print(f"Shapes match: {final_prob.shape == test_image_swap.shape}")

threshold = 0.5
binary_mask = (final_prob > threshold).astype(np.uint8)

print(f"\nSegmented voxels (threshold={threshold}): {binary_mask.sum()}")

# Save results
output_dir = Path("/home/akaur101/data/3d_data/3d_segmentation_results")
output_dir.mkdir(exist_ok=True)

print(f"\nSaving results to: {output_dir}")
np.save(output_dir / "3d_probability_map.npy", final_prob)
np.save(output_dir / "3d_binary_mask.npy", binary_mask)
np.save(output_dir / "count_map.npy", count_map[40:-40, 40:-40, 40:-40])
