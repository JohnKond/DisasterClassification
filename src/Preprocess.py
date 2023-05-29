import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from MyDataset import torch_transform
import matplotlib.pyplot as plt

# to supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# size to be resized
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224,224,3)

'''
def augmentation(train_images, train_labels):
    # Data augmentation (optional)
    # Apply data augmentation techniques to augment the training dataset
    # Create an ImageDataGenerator instance for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,  # Randomly rotate images by 10 degrees
        width_shift_range=0.1,  # Randomly shift images horizontally by 10% of the width
        height_shift_range=0.1,  # Randomly shift images vertically by 10% of the height
        zoom_range=0.2,  # Randomly zoom images by 20%
        horizontal_flip=True  # Randomly flip images horizontally
    )

    # Fit the ImageDataGenerator on the training data
    datagen.fit(train_images)

    # Generate augmented images for training
    augmented_images = datagen.flow(train_images, train_labels, batch_size=32)
    return augmented_images
'''

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

    return X_train, y_train, X_test, y_test 


def resize_image(image):
    return cv2.resize(image, IMAGE_SIZE)

# Function to preprocess an image (resize and flatten)
def preprocess_image(image):
    resized_image = resize_image(image)
    return resized_image



def adjust_image(image):
    # Adjust the range of the image to 0-255
    image = image - np.min(image)
    image = image / np.max(image) * 255
    image = image.astype(np.uint8)
    return image



def plot_images_before_after(images, images_before,labels=['Flood','Fire','Earthquake','Cyclone']):
    
    i = 0
    pca = PCA(n_components=10)
    fig, axs = plt.subplots(2,len(labels))
    fig.suptitle('Images before PCA')

    original_shape = images_before[0].shape
    images_flat = images.reshape(images.shape[0], -1)
    images_pca = pca.fit_transform(images_flat)
    # images_recon = pca.inverse_transform(images_pca)
    # images_recon = images_recon.reshape(original_shape)
    
    indices = [0,8,16,24]
    for img in images_before:


            

        axs[0,i].imshow(img)
        axs[0,i].set_title(f'{labels[i]}')

        image_reconstructed = pca.inverse_transform(images_pca[indices[i]])
        image_reconstructed= image_reconstructed.reshape(original_shape)
        image_reconstructed = cv2.convertScaleAbs(image_reconstructed)

        axs[1,i].imshow(image_reconstructed)
        axs[1,i].set_title(f'{labels[i]}')

        i += 1
    
    plt.tight_layout()
    plt.show()

    # pca = PCA(n_components=4)
    # images_flat = images_flat.reshape(images_flat.shape[0], -1)
    # images_pca = pca.fit_transform(images_flat)








def preprocess(DATA_DIR):

    # counter for file names
    # images = np.empty((0,IMAGE_SIZE[0]*IMAGE_SIZE[1]), np.float32)
    images = []

    # labels = np.empty((0,), dtype=np.int32)
    labels = []

    # list to contain one image from each category before PCA
    images_example_before = []

    # Iterate through each category folder
    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        
        # flag for isolating first image for each category
        first = 1
        
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                # Load image and resize
                img_path = os.path.join(category_path, filename)
                
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_image = preprocess_image(img)

                if first:
                    images_example_before.append(processed_image)
                    # plt.imshow(processed_image)
                    # plt.show()
                    first = 0

                # images = np.append(images, np.expand_dims(processed_image.flatten(), axis=0), axis=0)
                images.append(processed_image)
                labels.append(np.array(category))


    images = np.array(images)
    labels = np.array(labels)

    # plot image distribution chart
    # plot_image_distributions(labels)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform the labels
    labels = label_encoder.fit_transform(labels)
    labels = np.reshape(labels, (len(labels), 1))

    # just for plot example
    # images_example_before = np.array(images_example_before)
    # plot_images_before_after(images, images_example_before)

    # flatten the images

    return images, labels