import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)


def plot_images(pca,PCA_X_train):
    pca_image = PCA_X_train[1]
    # pca_image_resized = cv2.resize(pca_image, IMAGE_SIZE)
    pca_image_recon = pca.inverse_transform(pca_image)
    image_reconstructed = pca_image_recon.reshape(IMAGE_SIZE)
    plt.imshow(image_reconstructed)
    plt.show()
    return None


def plot_variance(pca):
    # Get the explained variance ratios
    explained_variances = pca.explained_variance_ratio_

    # Compute the cumulative explained variance
    cumulative_variances = np.cumsum(explained_variances)

    # Plot the variance explained
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variances) + 1), cumulative_variances, '-o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Variance Explained')
    plt.grid(True)
    plt.show()