from models.TinyImgNetAlexNet import TinyImgNetAlexNet
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import wandb

from utils.loops import train_loop, test_loop
from utils.dataset import train_path, val_path, transform

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    learning_rate = 0.1
    epochs = 10
    batch_size = 16
    
    wandb.init(
        project="tiny-imagenet",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "SGD",
        }
    )
    
    tiny_imagenet_dataset_train = ImageFolder(root=train_path, transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root=val_path, transform=transform)
    dataloader_train = DataLoader(dataset=tiny_imagenet_dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=tiny_imagenet_dataset_val, batch_size=batch_size, shuffle=True)

    # Create the model
    model = TinyImgNetAlexNet().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=5e-4)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------------")
        train_loop(dataloader_train, model, loss_fn, optim, device)
        test_loop(dataloader_val, model, loss_fn, device)
    
    
    torch.save(model.state_dict(), "checkpoints/TinyImgNetAlexNet_weights.pth")
    wandb.save("checkpoints/TinyImgNetAlexNet_weights.pth")
    
    wandb.finish()
    
    
if __name__ == "__main__":
    main()