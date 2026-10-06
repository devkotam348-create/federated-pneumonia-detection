import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader 

# ============== Processing Pipeline ==================
# Resize: CNN needs a fixed input size, but raw x -rays come in varing dimensions
# GrayScale: datasets has mixed color modes(RGB-grayscale) - forcing single channel
# ToTensor : converts PIL Image -> PyTorch tensor, also scales pixel values from 0-255 
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor()
    ])
# ============= joining path 
DATA_DIR = os.path.join('data', 'raw', 'chest_xray')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Image Fol;der auto-labels images based on their subfolder name(NORMAL = 0, PNEUMONIA = 1)
# and applies our transforms pipeline to every image as it's loaded
train_dataset = datasets.ImageFolder(root = TRAIN_DIR, transform = transform) # datasets for train dir 

print(train_dataset.classes) # confirms label order: [ 'NORMAL', PNEUMONIA]
print(len(train_dataset))

test_dataset = datasets.ImageFolder(root = TEST_DIR, transform = transform) # dataset for test dir

print(test_dataset.classes)
print(len(test_dataset))


val_dataset = datasets.ImageFolder(root = VAL_DIR, transform = transform) # datasets for val dir

print(val_dataset.classes)
print(len(val_dataset))

# Dataloader groups dataset into batches for training

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
