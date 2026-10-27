import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.TinyImgNetClassifier import TinyImgNetClassifier
from utils.dataset import download_dataset, transform
from utils.loops import test_loop


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _, test_path = download_dataset()
    tiny_imagenet_dataset_test = ImageFolder(root=test_path, transform=transform["val"])
    dataloader_test = DataLoader(dataset=tiny_imagenet_dataset_test, batch_size=64, shuffle=True)
    
    
    model = TinyImgNetClassifier().to(device)
    model.load_state_dict(torch.load("checkpoints/TinyImgNetClassifier_weights.pth"))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    print("Evaluating the model on the test dataset...")
    test_loop(dataloader_test, model, loss_fn)

if __name__ == "__main__":
    main()