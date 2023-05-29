from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from MyDataset import torch_transform
from sklearn.preprocessing import StandardScaler
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split






def split_data(images,labels):
    # Print the shape of the train and test sets
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    # X_train = StandardScaler().fit_transform(images_train)
    # X_test = StandardScaler().fit_transform(images_test)

    print('Shape of dataset before PCA:')
    print('----------------------------')
    print('Train Images:', X_train.shape)
    print('Train Labels:', y_train.shape)
    print('Test Images:', X_test.shape)
    print('Test Labels:', y_test.shape)
    print('\n\n')

    return X_train,X_test, y_train, y_test 


