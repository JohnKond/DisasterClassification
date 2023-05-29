import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from TF_CNN import create_model, train_model

from Preprocess import preprocess, split_data
from PCA import applyPCA, applyPCAFlat
from MyDataset import torch_transform
from plots import plot_variance
import matplotlib.pyplot as plt
# from TransferLearning import train,evaluate
from SVM import SVM_train
from CNN_utils import read_files,accuracy, CNN_evaluate, CNN_fit
from CNN import DisasterClassification

# Configure project path
project_path = "/Users/kondo/Documents/master/ML/Disaster_Image_Detection/"



# Configure data path of original images and data path of 
# transformed images
DATA_DIR = project_path + 'data'
PCA_DATA_DIR = project_path+ 'transformed_data'
PROCESSED_DATA_PATH = project_path + 'preprocessed_data'
NN_PROCESSED_DATA_PATH = project_path + 'nn_preprocessed_data'
MODEL_PARAMS_FILE = project_path + 'cnn_model_params.pth'

def save_processed_data(X_train, y_train, X_test, y_test,path):

    # Create a folder to store the preprocessed data
    # save_folder = 'preprocessed_data'
    print('Saving data to numpy arrays..')
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'X_train.npy'), X_train)
    np.save(os.path.join(path, 'X_test.npy'), X_test)
    np.save(os.path.join(path, 'y_train.npy'), y_train)
    np.save(os.path.join(path, 'y_test.npy'), y_test)



def save_pca_object(pca):
    filename = 'pca_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(pca, file)

def load_pca_object():
    # Load PCA object
    with open('pca_model.pkl', 'rb') as file:
        pca_loaded = pickle.load(file)
    return pca_loaded

def convert_to_data_loaders(X_train, y_train, X_test, y_test):
     # Convert the numpy arrays to PyTorch tensors
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    train_dataset = torch_transform(X_train, y_train)
    test_dataset = torch_transform(X_test, y_test)

    # Create data loaders for batching the data during training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)   
    return train_loader, test_loader

def load_pca_object():
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    return pca

def load_processed_data(path):
    print('Loading data from numpy arrays..')
    X_train = np.load(os.path.join(path, 'X_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    X_test = np.load(os.path.join(path, 'X_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    return X_train, y_train, X_test, y_test



def PCA_pipeline():
    # if data are already processed:
    # TODO put NOT
    if not os.path.exists(PROCESSED_DATA_PATH):
        # preprocess images and split in train and test set
        images, labels  = preprocess(DATA_DIR)

        # reshape images
        images_flat = images.reshape(images.shape[0], -1)


        X_train, y_train, X_test, y_test = split_data(images_flat, labels)
        pca_object, PCA_X_train, PCA_X_test = applyPCA(X_train, X_test, y_train, y_test)
        save_pca_object(pca_object)
        save_processed_data(PCA_X_train, y_train, PCA_X_test, y_test,PROCESSED_DATA_PATH)

    else :
        # load train and test sets from disk in order to save time and space
        PCA_X_train, y_train, PCA_X_test, y_test = load_processed_data(PROCESSED_DATA_PATH)    

    
    plot_variance(load_pca_object())


def save_cnn_model(model):
    torch.save(model.state_dict(), MODEL_PARAMS_FILE)
    
def load_cnn_model():
    model = DisasterClassification(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model





def CNN_train(epochs, train_loader):
    # Define the CNN model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # Replace the classifier with your own

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def CNN_test(model, test_loader):
    # Evaluate on the test set
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)
            predictions.extend(predicted_labels.tolist())

    # Convert predictions to numpy array
    predictions = torch.tensor(predictions).numpy()

    # Print the predictions
    print(predictions)

def NN_pipeline_prev():
    print('-------------------------------')
    print('| Starting Neural Network pipeline    |')
    print('-------------------------------\n\n')

    train_dl, test_dl, val_dl = read_files(DATA_DIR)
    model = DisasterClassification()

    if os.path.exists(MODEL_PARAMS_FILE):
        trained_model = load_cnn_model()

    else :
        # trained_model, history = CNN_fit(
        #     epochs = 50,
        #     lr = 0.001,
        #     model= model,
        #     train_loader= train_dl,
        #     val_loader= val_dl,
        #     opt_func= torch.optim.Adam)

        trained_model = CNN_train(epochs=10,train_loader = train_dl)
        CNN_test(trained_model,test_loader=test_dl)

        print('End of Training..')    
        save_cnn_model(model)
    
    
        
def NN_pipeline():
        print(' -------------------------------------')
        print('| Starting Neural Network pipeline    |')
        print(' -------------------------------------\n\n')
        if os.path.exists(NN_PROCESSED_DATA_PATH):
            X_train, y_train, X_test, y_test = load_processed_data(NN_PROCESSED_DATA_PATH)
        else:
            # preprocess images and split in train and test set
            images, labels  = preprocess(DATA_DIR)
            X_train, y_train, X_test, y_test = split_data(images, labels)
            save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, NN_PROCESSED_DATA_PATH)


        # create validation data for our model
        X_train, y_train, X_val, y_val = split_data(X_train, y_train)


        # print(X_train.shape)
        # print(X_val.shape)
        # print(X_test.shape)

        model = create_model()
        trained_model = train_model(model, X_train, y_train, X_val, y_val)
        
        
        
        
        

        



    

def main():

    # run PCA_pipeline()
    # PCA_pipeline()

    

    # run Neural Network pipeline
    NN_pipeline()


    
    # create data loader for PyTorch model
    # train_loader, test_loader = convert_to_data_loaders(PCA_X_train, y_train, PCA_X_test, y_test)

    # for images, labels in train_loader:
        # print(images.shape)
        # print(labels.shape)
    # train model (transfer learning)
    # train(train_loader)


    # SVM_train(PCA_X_train,y_train, PCA_X_test, y_test)







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


