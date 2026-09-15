import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader 

# ========= transform the data into 224 by 224 pixel , with grey channel ===========
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

train_dataset = datasets.ImageFolder(root = TRAIN_DIR, transform = transform) # datasets for train folder 

print(train_dataset.classes)
print(len(train_dataset))

test_dataset = datasets.ImageFolder(root = TEST_DIR, transform = transform) # dataset for test folder 

print(test_dataset.classes)
print(len(test_dataset))


val_dataset = datasets.ImageFolder(root = VAL_DIR, transform = transform) # datasets for val folder 

print(val_dataset.classes)
print(len(val_dataset))


train_loader = DataLoader(train_dataset, batch_size = 32, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)


