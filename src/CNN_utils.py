import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader




def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=6).permute(1,2,0))
        break
    plt.show()

def read_files(DATA_DIR):

    #load the train and test data
    dataset = ImageFolder(DATA_DIR,transform = transforms.Compose([
        transforms.Resize((224,224)),transforms.ToTensor()
    ]))



    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    
    val_size = int(0.8 * train_size)
    train_size = train_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle = True, num_workers = 4, pin_memory = True)
    test_dl = DataLoader(test_dataset, batch_size=32, pin_memory = True)
    val_dl = DataLoader(val_dataset, batch_size=32, num_workers = 4, pin_memory = True)


    return train_dl, val_dl, test_dl


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def CNN_evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def CNN_fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = CNN_evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return model,history

        


    



    # img, label = test_dataset[0]
    # print(img.shape,label)

    # show_batch(train_dl)



    # test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    #     transforms.Resize((150,150)),transforms.ToTensor()
    # ]))
