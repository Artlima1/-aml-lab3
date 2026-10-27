from models.TinyImgNetClassifier import TinyImgNetClassifier
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils.loops import train_loop, test_loop
from utils.dataset import download_dataset, transform

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_path, test_path = download_dataset()
    
    tiny_imagenet_dataset_train = ImageFolder(root=train_path, transform=transform["train"])
    tiny_imagenet_dataset_test = ImageFolder(root=test_path, transform=transform["val"])
    dataloader_train = DataLoader(dataset=tiny_imagenet_dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset=tiny_imagenet_dataset_test, batch_size=64, shuffle=True)

    # Create the model
    model = TinyImgNetClassifier().to(device)
    print(model)
    
    learning_rate = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------------")
        train_loop()
    
    
    torch.save(model.state_dict(), "checkpoints/TinyImgNetClassifier_weights.pth")
    
    
if __name__ == "__main__":
    main()