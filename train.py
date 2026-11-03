from models.TinyImgNetAlexNet import TinyImgNetAlexNet
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import wandb

from utils.loops import train_loop, test_loop
from utils.dataset import train_path, val_path, transform, prepare_val_folder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    learning_rates = [0.1, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    num_epochs = 5
    
    tiny_imagenet_dataset_train = ImageFolder(root=train_path, transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root=val_path, transform=transform)
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print("============================================================================")
            print(f"INITIATING TRAIN FOR lr={lr} and bs={batch_size}")
            print("============================================================================")
            
            wandb.init(
                project="tiny-imagenet",
                config={
                    "model": "alexnet",
                    "learning_rate": lr,
                    "batch_size": batch_size,
                }
            )
            dataloader_train = DataLoader(dataset=tiny_imagenet_dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            dataloader_val = DataLoader(dataset=tiny_imagenet_dataset_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

            # Create the model
            model = TinyImgNetAlexNet().to(device)
            print(model)
            
            loss_fn = nn.CrossEntropyLoss()
            optim = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)
            
            for t in range(num_epochs):
                print(f"Epoch {t+1}\n------------------------------------------")
                train_loop(dataloader_train, model, loss_fn, optim, device)
            
            test_loop(dataloader_val, model, loss_fn, device)
            
            torch.save(model.state_dict(), f"checkpoints/TinyImgNetAlexNet_weights_{lr}_{batch_size}.pth")
            wandb.save(f"checkpoints/TinyImgNetAlexNet_weights_{lr}_{batch_size}.pth")
            
            wandb.finish()
    
    
if __name__ == "__main__":
    main()