import os
import torch
import pickle
import tensorflow as tf
import numpy as np

from plots import plot_variance, plot_distribution, plot_hist_box_plot, plot_initial_images,pie_chart_images
import matplotlib.pyplot as plt
from SVM import ML_pipeline

 
from save_load_utils import save_processed_data, save_pca_object, load_pca_object, save_cnn_model, load_cnn_model, load_processed_data

# for pre-processing 
from Preprocess import preprocess, split_data

# for PCA
from PCA import applyPCA, applyPCAFlat

# for my CNN network
import MyCNN

# for tranfer learning CNN network
from TranferLearning import TL_read_data, TL_create_model, TL_compile_model, TL_train_model,TL_test_model

# CNN utils functions for 2nd and 3rd experiment
from CNN_utils import CNN_read_data, CNN_compile_model, CNN_test_model, CNN_train_model



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
RF_MODEL_FILE = TRAINED_MODELS_DIR + 'RF_model.pkl'
SVM_MODEL_FILE = TRAINED_MODELS_DIR + 'SVM_model.pkl'
TL_MODEL_FILE = TRAINED_MODELS_DIR + 'TL_model.h5'
MY_CNN_MODEL_FILE = TRAINED_MODELS_DIR + 'myCNN_model.h5'



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


        
def myCNN_pipeline():
        print(' -------------------------------------------')
        print('| Starting myCNN Neural Network pipeline    |')
        print(' -------------------------------------------\n\n')

        # read data
        train_dataset, validation_dataset, test_dataset = CNN_read_data(DATA_DIR)

        # If model already exists load, else create new model and train
        if os.path.exists(MY_CNN_MODEL_FILE):
            #import model
            trained_model = tf.keras.models.load_model(MY_CNN_MODEL_FILE)
        else:

            # create model
            model = MyCNN.create_model()

            # compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            compiled_model = CNN_compile_model(model=model,
                        optimizer = optimizer,
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
            
            
            # train model
            trained_model = CNN_train_model(compiled_model, train_dataset, validation_dataset)
            # save trained model
            trained_model.save(MY_CNN_MODEL_FILE)
            
            # test the model
        CNN_test_model(trained_model,test_dataset)
        
        
        

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

            # compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            compiled_model = TL_compile_model(model=model,
                        optimizer=optimizer,
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
            
            
            # train model
            trained_model = TL_train_model(compiled_model, train_dataset, validation_dataset)
            # save trained model
            trained_model.save(TL_MODEL_FILE)
            
        # test the model
        # TL_test_model(trained_model,test_dataset)
        CNN_test_model(trained_model,test_dataset)
        


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


