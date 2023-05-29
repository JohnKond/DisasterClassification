import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class torch_transform(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Convert image and label to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label


    

