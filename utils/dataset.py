import requests
from zipfile import ZipFile
from io import BytesIO

import torchvision.transforms as transforms

dataset_downloaded_flag = False

def download_dataset():
    if not dataset_downloaded_flag:
        dataset_path = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        response = requests.get(dataset_path)
        # Check if the request was successful
        if response.status_code == 200:
            # Open the downloaded bytes and extract them
            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall('/dataset')
            print('Download and extraction complete!')
            
        dataset_downloaded_flag = True
        
    return (
        '/dataset/tiny-imagenet-200/train'
        '/dataset/tiny-imagenet-200/test'
    )
    
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}