import cv2
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