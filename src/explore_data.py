import os
import matplotlib.pyplot as plt
from PIL import Image
#========== joining the path ===========================
DATA_DIR = os.path.join('data', 'raw', 'chest_xray')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')


def count_images(folder_path):
    '''count the image in the folder and return the total count of images in that folder '''
    return len(os.listdir(folder_path))

# ============ count image per class ===================
print('Train Normal:', count_images(os.path.join(TRAIN_DIR,'NORMAL')))
print('Train Pneumonia:', count_images(os.path.join(TRAIN_DIR,'PNEUMONIA')))
    
# ============ view one sample image form each class ===============   
normal_files = os.listdir(os.path.join(TRAIN_DIR, 'NORMAL'))
first_normal = normal_files[0]
image_path = os.path.join(TRAIN_DIR ,'NORMAL', first_normal)
img_normal = Image.open(image_path)

pneumonia_files = os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA'))
first_pneumonia = pneumonia_files[0]
pneumonia_path = os.path.join(TRAIN_DIR, 'PNEUMONIA', first_pneumonia)
img_pneumonia = Image.open(pneumonia_path)

plt.subplot(1,2,1)
plt.imshow(img_normal)
plt.title('Normal')

plt.subplot(1,2,2)
plt.imshow(img_pneumonia)
plt.title('PNEUMONIA')

print('NORMAL image size:: ', img_normal.size)
print('PNEUMONIA image size:: ', img_pneumonia.size)

plt.show()

normal_size = [] # =============== list of all the sizes of normal pictures
for filename in os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')):
    image_path = os.path.join(TRAIN_DIR,'NORMAL', filename)
    img = Image.open(image_path)
    normal_size.append(img.size)
    
pneumonia_size = [] # ============= list of all the sizes of pneumonia pictures 
for filename in os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA')):
    image_path = os.path.join(TRAIN_DIR, 'PNEUMONIA', filename)
    img = Image.open(image_path)
    pneumonia_size.append(img.size)
    
normal_modes = []
for filename in os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')):
    image_path = os.path.join(TRAIN_DIR, 'NORMAL', filename)
    img = Image.open(image_path)
    normal_modes.append(img.mode)
    
pneumonia_modes = []
for filename in os.listdir(os.path.join(TRAIN_DIR,'PNEUMONIA')):
    image_path = os.path.join(TRAIN_DIR, 'PNEUMONIA', filename)
    img = Image.open(image_path)
    pneumonia_modes.append(img.mode)
    
print('NORMAL SMALEST:: ', min(normal_size))
print('NORMAL LARGEST:: ', max(normal_size))
print('PNEUMONIA SMALLEST:: ', min(pneumonia_size))
print('PNEUMONIA LARGEST:: ', max(pneumonia_size))
print('Test Normal:: ',count_images(os.path.join(TEST_DIR, 'NORMAL')))
print('Test Pneumonia:: ', count_images(os.path.join(TEST_DIR, "PNEUMONIA")))
print('VAl Normal:: ', count_images(os.path.join(VAL_DIR, 'NORMAL')))
print('Val Pneumonia:: ', count_images(os.path.join(VAL_DIR, 'PNEUMONIA')))
print(img_normal.mode)
print(img_pneumonia.mode)
print('NORMAL modes found:: ', set(normal_modes))
print('PNEUMONIA modes found', set(pneumonia_modes))