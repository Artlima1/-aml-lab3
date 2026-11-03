import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO

import torchvision.transforms as transforms

dataset_downloaded_flag = False

train_path, val_path, test_path = (
    '/dataset/tiny-imagenet-200/train',
    '/dataset/tiny-imagenet-200/val',
    '/dataset/tiny-imagenet-200/test',
)

def download_dataset():
    dataset_path = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    response = requests.get(dataset_path)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the downloaded bytes and extract them
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall('/dataset')
        print('Download and extraction complete!')


def prepare_val_folder(val_dir=val_path):
    """Reorganize Tiny ImageNet val/ into ImageFolder format: val/<wnid>/<img>.JPEG.

    The original layout is val/images/*.JPEG + val_annotations.txt (one label per image).
    Safe to call multiple times: does nothing if already reorganized.
    """
    images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.isdir(images_dir):
        return  # already reorganized

    with open(annotations_file) as f:
        for line in f:
            img_name, wnid = line.split('\t')[:2]
            class_dir = os.path.join(val_dir, wnid)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, os.path.join(class_dir, img_name))

    os.rmdir(images_dir)
    print('Validation folder reorganized into class subfolders.')

    
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match AlexNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])