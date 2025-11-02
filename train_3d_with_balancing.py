#fixing the patch generation
#oct 7, 2025

import numpy as np
import random
import torch.nn as nn
from skimage import io
import sys
import os
from datetime import datetime
import torch
from pathlib import Path
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose    #, Normalize, ToTensor

#images_path = Path("/Users/aseeskaur/Documents/Fluo-C3DH-A549/01")
#masks_path  = Path("/Users/aseeskaur/Documents/Fluo-C3DH-A549/01_GT/SEG")

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = "checkpoints/best_model.pth"  # Path to resume from
RESUME_FROM_CHECKPOINT = os.path.exists(CHECKPOINT_PATH) 

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def save_checkpoint(epoch, model, optimizer, best_val_loss, running_loss_list, is_best=False):
    """Save training checkpoint"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'running_loss_list': running_loss_list,
        'timestamp': timestamp
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}_v2.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model_3d_after_bal_swap_v2.pth')
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved with validation loss: {best_val_loss:.6f}")

    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load training checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    running_loss_list = checkpoint.get('running_loss_list', [])

    print(f"Resuming from epoch {start_epoch}")
    print(f"Best validation loss so far: {best_val_loss:.6f}")

    return start_epoch, best_val_loss, running_loss_list

images_dir = sys.argv[1]
masks_dir = sys.argv[2]
epochs = int(sys.argv[3])


images_path = Path(images_dir)
masks_path = Path(masks_dir)

image_files = images_path.glob("*.tif")
mask_files  = masks_path.glob("*.tif")

train_image_path  = images_path/"t002.tif"
train_mask_path   = masks_path/"man_seg002.tif"

val_image_path = images_path/"t006.tif"
val_mask_path = masks_path/"man_seg006.tif"


train_image = io.imread(train_image_path) 
train_image_swap = np.swapaxes(train_image, 0, 2)
train_mask = io.imread(train_mask_path)
train_mask_swap = np.swapaxes(train_mask, 0, 2)

train_image_padded = np.pad(train_image_swap, 40, mode = 'constant')
train_mask_padded = np.pad(train_mask_swap, 40, mode = 'constant')

val_image = io.imread(val_image_path)
val_image_swap = np.swapaxes(val_image,0,2)
val_mask = io.imread(val_mask_path)
val_mask_swap = np.swapaxes(val_mask, 0, 2)

val_image_padded = np.pad(val_image_swap, 40, mode = 'constant')
val_mask_padded = np.pad(val_mask_swap, 40, mode = 'constant')

snap_mask_tr_sums = []
sum_categories_train = {}

snap_mask_val_sums = []
sum_categories_val = {}

train_image_list = []
train_mask_list  = []

val_image_list = []
val_mask_list = []



m = 80
n = 3
p = 10

pad_bound_x = int(m/2)
pad_bound_y = int(m/2)
pad_bound_z = int(p/2)

snap_image = np.empty([m, m, p])
snap_mask = np.empty([n, n, n])

x = np.shape(train_image_padded)[0]
y = np.shape(train_image_padded)[1]
z = np.shape(train_image_padded)[2]  

for i in range(pad_bound_x, x - pad_bound_x):
    for j in range(pad_bound_y, y - pad_bound_y):
        for k in range(pad_bound_z, z - pad_bound_z):  
            snap_image_tr = train_image_padded[i - pad_bound_x:i + pad_bound_x, j - pad_bound_y:j + pad_bound_y, k - pad_bound_z:k + pad_bound_z]  
            snap_mask_tr = train_mask_padded[i - 1:i + 2, j - 1:j + 2, k - 1:k+2 ] 
            snap_mask_tr_sum = np.sum(snap_mask_tr)
            

            snap_image_val = val_image_padded[i - pad_bound_x:i + pad_bound_x, j - pad_bound_y:j + pad_bound_y, k - pad_bound_z:k + pad_bound_z]  
            snap_mask_val = val_mask_padded[i - 1:i + 2, j - 1:j + 2, k - 1:k+2 ]  
            snap_mask_val_sum = np.sum(snap_mask_val)

            train_image_list.append(snap_image_tr)
            train_mask_list.append(snap_mask_tr)

            if snap_mask_tr_sum in sum_categories_train:
                sum_categories_train[snap_mask_tr_sum]['images'].append(snap_image_tr)
                sum_categories_train[snap_mask_tr_sum]['masks'].append(snap_mask_tr)
            else:
                sum_categories_train[snap_mask_tr_sum] = {'images': [snap_image_tr], 'masks': [snap_mask_tr]}

            
            if snap_mask_val_sum in sum_categories_val:
                sum_categories_val[snap_mask_val_sum]['images'].append(snap_image_val)
                sum_categories_val[snap_mask_val_sum]['masks'].append(snap_mask_val)
            else:
                sum_categories_val[snap_mask_val_sum] = {'images': [snap_image_val], 'masks': [snap_mask_val]}

            val_image_list.append(snap_image_val)
            val_mask_list.append(snap_mask_val)

num_snaps_tr = []
for s in sum_categories_train:
    print("For training image:")
    print("Number of snaps for sum ",s, "are ", len(sum_categories_train[s]['images']))
    num_snaps_tr.append(len(sum_categories_train[s]['images']))


n_train = np.min(num_snaps_tr)
random_images_train = []
random_masks_train  = []

for sum_value in sum_categories_train:
    images_for_category = sum_categories_train[sum_value]['images']
    masks_for_category = sum_categories_train[sum_value]['masks']

    if len(images_for_category) >= n_train:
        random_ind = random.sample(range(len(images_for_category)), n_train)
        random_images_train.extend([images_for_category[i] for i in random_ind])
        random_masks_train.extend([masks_for_category[i] for i in random_ind])

num_snaps_val = []
for s in sum_categories_val:
    print("For validation image: ")
    print("Number of snaps for sum ",s, "are ", len(sum_categories_val[s]['images']))
    num_snaps_val.append(len(sum_categories_val[s]['images']))

n_val = np.min(num_snaps_val)
random_images_val = []
random_masks_val  = []

for sum_value in sum_categories_val:
    images_for_category = sum_categories_val[sum_value]['images']
    masks_for_category = sum_categories_val[sum_value]['masks']

    if len(images_for_category) >= n_val:
        random_ind = random.sample(range(len(images_for_category)), n_val)
        random_images_val.extend([images_for_category[i] for i in random_ind])
        random_masks_val.extend([masks_for_category[i] for i in random_ind])



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


transforms = Compose([
    ToTensorAndNormalize3D(normalize=True) 
    #Normalize(mean=[0.5], std=[0.5])  
])


train_dataset = CellData3d(image_list=random_images_train, 
                           mask_list=random_masks_train, 
                           transforms=transforms)

train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=50)       #shuffle true for training data

val_dataset = CellData3d(image_list=random_images_val, 
                           mask_list=random_masks_val, 
                           transforms=transforms)

val_dataloader = DataLoader(val_dataset,
                              shuffle=False,
                              batch_size=50)  

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
        #self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=3)

        
        # 3D convolution blocks
        self.conv1 = Double_Conv3D(1, 64)  
        self.conv2 = Double_Conv3D(64, 128)
        self.conv3 = Double_Conv3D(128, 256)
        self.conv4 = Double_Conv3D(256, 512)
        self.conv5 = nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        # Forward pass through the network
        
        x1 = self.conv1(image) 
        print(x1.shape)
        x2 = self.maxpool(x1)  
        print(x2.shape)
        
        x3 = self.conv2(x2)   
        print(x3.shape)
        x4 = self.maxpool2(x3) 
        print(x4.shape)
        
        x5 = self.conv3(x4)     
        print(x5.shape)
        x6 = self.maxpool3(x5)  
        print(x6.shape)
        
        x7 = self.conv4(x6)     
        print(x7.shape)
        x8 = self.maxpool4(x7)  
        print(x8.shape)

        x9 = self.conv5(x8)
        print(x9.shape)
    
        return x9



#device = torch.device('cpu')
model  = net()
model  = model.float()
model  = model.to(device=device)


optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)
criterion = torch.nn.BCEWithLogitsLoss()


start_epoch = 0
best_val_loss = float('inf')
best_epoch = -1
train_losses = []
val_losses = []
running_loss_list = []

if RESUME_FROM_CHECKPOINT and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    start_epoch, best_val_loss, running_loss_list = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

print(f"Starting training from epoch {start_epoch}")
for epoch in range(start_epoch, epochs):
    print("Epoch: ", epoch)
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        images, masks = data
        images = images.to(device=device)
        masks = masks.to(device=device)
        
        optimizer.zero_grad()    #empty the gradients
        
        outputs = model(images.float())
        #print(outputs)
        
        
        loss = criterion(outputs, masks.float())
        print(loss)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()

            
    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            images, masks = data
            images = images.to(device=device)
            masks = masks.to(device=device)
            outputs = model(images.float())
            val_loss += criterion(outputs, masks.float()).item()
    
    avg_val_loss = val_loss/ len(val_dataloader)
    val_losses.append(avg_val_loss)    
    

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_epoch = epoch +1 

    checkpoint_path = save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        best_val_loss=best_val_loss,
        running_loss_list=running_loss_list,
        is_best=is_best
)

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}{' (Best!)' if is_best else ''}")
print(f"\nFinished Training - Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
