import os
import random
import shutil

def split_dataset(image_folder, label_folder, train_folder, val_folder, train_ratio=0.8):
    # Get all image files (assuming image files have .jpg extension)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # Shuffle the image files to randomize the split
    random.shuffle(image_files)

    # Split into train and validation sets
    num_train = int(len(image_files) * train_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    # Copy files to train and val folders
    for file in train_files:
        shutil.copy(os.path.join(image_folder, file), os.path.join(train_folder, file))
        shutil.copy(os.path.join(label_folder, file.replace('.jpg', '.txt')), os.path.join(train_folder, file.replace('.jpg', '.txt')))
    
    for file in val_files:
        shutil.copy(os.path.join(image_folder, file), os.path.join(val_folder, file))
        shutil.copy(os.path.join(label_folder, file.replace('.jpg', '.txt')), os.path.join(val_folder, file.replace('.jpg', '.txt')))

# Define the folder paths
image_folder = 'E:/Users/Public/ProjectICS/All_videos/013'  # Folder containing images
label_folder = 'E:/Users/Public/ProjectICS/All_videos/013/labels'  # Folder containing label files
train_folder = 'E:/Users/Public/ProjectICS/All_videos/013/train'  # Folder for training images
val_folder = 'E:/Users/Public/ProjectICS/All_videos/013/val'  # Folder for validation images

# Create the train and val directories
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Create the train and val labels directories
os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)

# Split the dataset
split_dataset(image_folder, label_folder, train_folder, val_folder)
