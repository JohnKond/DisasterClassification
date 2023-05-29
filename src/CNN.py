# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from torch.nn import Module
# from torch.nn import Conv2d
# from torch.nn import Linear
# from torch.nn import MaxPool2d
# from torch.nn import ReLU
# from torch.nn import LogSoftmax
# from torch import flatten

# # Get cpu, gpu or mps device for training.
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")


# CLASSES = ["Earthquake", "Fire","Cyclone","Flood"]

# BATCH_SIZE = 32








# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(56 * 56 * 16, 128)
#         self.fc2 = nn.Linear(128, num_classes)
        
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Define the model
# num_classes = 4  # Number of output classes
# model = SimpleCNN(num_classes)

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)





