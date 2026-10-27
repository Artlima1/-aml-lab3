import torch
import wandb

def train_loop(dataloader, model, loss_fn, optmizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        
        train_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            wandb.log({"batch_loss": loss, "batch": batch})
    
    avg_train_loss = train_loss / len(dataloader)
    wandb.log({"avg_train_loss": avg_train_loss})
            
            
def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    accuracy = (100 * correct) / size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})