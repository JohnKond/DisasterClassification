import os
import torch
import pickle
import numpy as np

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



def save_cnn_model(model, PATH):
    PATH = MODEL_PARAMS_FILE
    torch.save(model.state_dict(), PATH)
    

def load_cnn_model():
    # model = DisasterClassification(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    # return model
    return


def load_processed_data(DIR):
    print('Loading data from numpy arrays..')
    X_train = np.load(os.path.join(DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DIR, 'y_test.npy'))
    return X_train, y_train, X_test, y_test