# initializing the pca
from sklearn.decomposition import PCA,IncrementalPCA
from plots import plot_images
import numpy as np
import matplotlib.pyplot as plt
import cv2





def applyPCA(X_train, X_test,y_train, y_test):
    pca =  PCA(n_components=20)
    PCA_X_train = pca.fit_transform(X_train)
    PCA_X_test = pca.transform(X_test)

    print('Shape of dataset after PCA:')
    print('---------------------------')
    print('Train Images:', PCA_X_train.shape)
    print('Train Labels:', y_train.shape)
    print('Test Images:', PCA_X_test.shape)
    print('Test Labels:', y_test.shape)
    print('\n\n')

    return pca,PCA_X_train, PCA_X_test


def applyPCAFlat(images_flat,labels):
    pca = PCA(n_components=20)
    images_pca = pca.fit_transform(images_flat)
    # Display one image from each category after PCA
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        image_index = label_indices[0]
        image_pca = images_pca[image_index]

        # Reshape the PCA-transformed image back to its original shape
        image_reconstructed = pca.inverse_transform(image_pca)
        image_reconstructed = image_reconstructed.reshape(images_flat.shape[0])

        image_reconstructed = cv2.convertScaleAbs(image_reconstructed)
        # Display the reconstructed image after PCA
        plt.imshow(cv2.cvtColor(image_reconstructed, cv2.COLOR_BGR2RGB))
        # plt.imshow(image_reconstructed)
        plt.title(label)
        plt.axis('off')
        plt.show()



def plot_images_before_after():
        # Display the original and reconstructed images for each category
    categories = np.unique(labels)

    fig, axs = plt.subplots(len(categories), 2, figsize=(10, 10))

    for i, category in enumerate(categories):
        # Find the first image of the current category in the training set
        train_idx = np.where(y_train == category)[0][0]
        train_img = X_train[train_idx]

        # Find the first image of the current category in the test set
        test_idx = np.where(y_test == category)[0][0]
        test_img = X_test[test_idx]

        # Find the reconstructed image of the current category in the training set
        train_reconstructed_img = X_train_reconstructed[train_idx].reshape(X_train.shape[1:])

        # Find the reconstructed image of the current category in the test set
        test_reconstructed_img = X_test_reconstructed[test_idx].reshape(X_test.shape[1:])

        # Display the images
        axs[i, 0].imshow(train_img)
        axs[i, 0].set_title(f'Train Image ({category})')
        axs[i, 1].imshow(train_reconstructed_img)
        axs[i, 1].set_title(f'Train Reconstructed Image ({category})')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()