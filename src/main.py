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

from MyDataset import torch_transform
from plots import plot_variance, plot_distribution, plot_hist_box_plot, plot_initial_images,pie_chart_images
import matplotlib.pyplot as plt
from SVM import ML_pipeline
from CNN_utils import read_files,accuracy, CNN_evaluate, CNN_fit
from CNN import DisasterClassification
import tensorflow as tf
 
from save_load_utils import save_processed_data, save_pca_object, load_pca_object, save_cnn_model, load_cnn_model, load_processed_data

# for pre-processing 
from Preprocess import preprocess, split_data

# for PCA
from PCA import applyPCA, applyPCAFlat

# for my CNN network
from TF_CNN import CNN_create_model, CNN_train_model

# for tranfer learning CNN network
from TranferLearning import TL_read_data, TL_create_model, TL_compile_model, TL_train_model,TL_test_model


# Configure project path
project_path = "/Users/kondo/Documents/master/ML/DisasterClassification/"



# Paths of respective folders 
DATA_DIR = project_path + 'data'
PCA_DATA_DIR = project_path+ 'transformed_data'
PROCESSED_DATA_DIR = project_path + 'preprocessed_data'
NN_PROCESSED_DATA_DIR = project_path + 'nn_preprocessed_data'
DL_MODEL_PARAMS_FILE = project_path + 'cnn_model_params.pth'

# data dir and file for trained models
TRAINED_MODELS_DIR = project_path + 'trained_models/'
TL_MODEL_FILE = TRAINED_MODELS_DIR + 'TL_model.h5'
RF_MODEL_FILE = TRAINED_MODELS_DIR + 'RF_model.pkl'
SVM_MODEL_FILE = TRAINED_MODELS_DIR + 'SVM_model.pkl'



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


def PCA_pipeline():


    print(' ----------------------------------------')
    print('| Starting Transfer Learning pipeline    |')
    print(' ----------------------------------------\n\n')

    # if data are already processed, just import them
    if not os.path.exists(PROCESSED_DATA_DIR):
        # preprocess images and split in train and test set
        images, labels  = preprocess(DATA_DIR)

        # reshape images
        images_flat = images.reshape(images.shape[0], -1)

        # train-test split 
        X_train, y_train, X_test, y_test = split_data(images_flat, labels)
        
        # apply PCA to the dataset
        pca_object, PCA_X_train, PCA_X_test = applyPCA(X_train, X_test, y_train, y_test)
        
        # plot explained variance 
        plot_variance(pca_object)

        # save pca object and processed data for future usage
        save_pca_object(pca_object)
        save_processed_data(PCA_X_train, y_train, PCA_X_test, y_test,PROCESSED_DATA_DIR)

    else :
        # load train and test sets from disk in order to save time and space
        PCA_X_train, y_train, PCA_X_test, y_test = load_processed_data(PROCESSED_DATA_DIR)    

    # run ML pipeline : train and test SVM and RF model
    ML_pipeline(PCA_X_train, y_train, PCA_X_test, y_test,RF_MODEL_FILE, SVM_MODEL_FILE)



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

def myCNN_pipeline_prev():
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
    
    
        
def myCNN_pipeline():
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

        # normalize
        X_train = X_train / 255
        X_val = X_val / 255
        X_val = X_test / 255

        # create model architecture
        model = CNN_create_model()
        # train model
        trained_model = CNN_train_model(model, X_train, y_train, X_val, y_val)
        
        
        
        
        

def transferLearningPipeline():
        print(' ----------------------------------------')
        print('| Starting Transfer Learning pipeline    |')
        print(' ----------------------------------------\n\n')

        # read data
        train_dataset, validation_dataset, test_dataset = TL_read_data(DATA_DIR)

        # If model already exists load, else create new model and train
        if os.path.exists(TL_MODEL_FILE):
            #import model
            trained_model = tf.keras.models.load_model(TL_MODEL_FILE)
        else:

            # create model
            model = TL_create_model(train_dataset)


            # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            loss = tf.keras.losses.CategoricalCrossentropy()

            # compile model
            compiled_model = TL_compile_model(model=model,
                        base_learning_rate=0.0001,
                        loss=loss,
                        metrics=['accuracy'])
            
            
            # train model
            trained_model = TL_train_model(compiled_model, train_dataset, validation_dataset)
            # save trained model
            trained_model.save(TL_MODEL_FILE)
            
            # test the model
            TL_test_model(trained_model,test_dataset)
        


def data_analysis():
    plot_distribution(DATA_DIR)
    plot_initial_images(DATA_DIR)
    pie_chart_images(DATA_DIR)
    return

def main():

    # perform data analysis
    data_analysis()

    # run PCA_pipeline
    PCA_pipeline()

    

    # run pipeline with my custom CNN pipeline
    myCNN_pipeline()


    # run transfer learning pipeline, using pretrained model MobileNetV2
    transferLearningPipeline()
        


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


