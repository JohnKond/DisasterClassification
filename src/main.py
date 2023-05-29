import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from Preprocess import preprocess
from PCA import applyPCA, applyPCAFlat
from MyDataset import torch_transform
from plots import plot_images, plot_variance
from TransferLearning import train,evaluate
from SVM import SVM_train
from CNN import NN_pipeline

# Configure project path
project_path = "/Users/kondo/Documents/master/ML/Disaster_Image_Detection/"



# Configure data path of original images and data path of 
# transformed images
DATA_DIR = project_path + 'data'
PCA_DATA_DIR = project_path+ 'transformed_data'
PROCESSED_DATA_PATH = project_path + 'preprocessed_data'
NN_PROCESSED_DATA_PATH = project_path + 'nn_preprocessed_data'

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


def NN_pipeline():
    # if data are already processed:
    # TODO put NOT
    if not os.path.exists(NN_PROCESSED_DATA_PATH):
        # preprocess images and split in train and test set
        images, labels  = preprocess(DATA_DIR)



        X_train, y_train, X_test, y_test = split_data(images, labels)
        
        save_processed_data(X_train, y_train, X_test, y_test, NN_PROCESSED_DATA_PATH)

    else :
        # load train and test sets from disk in order to save time and space
        X_train, y_train, X_test, y_test = load_processed_data(NN_PROCESSED_DATA_PATH)
    
    

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


