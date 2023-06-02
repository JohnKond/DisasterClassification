import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_SIZE = (224, 224)






def plot_initial_images(DATA_DIR):
    CATEGORIES = ["Wildfire", "Earthquake", "Cyclone", "Flood"]

    fig, axes = plt.subplots(nrows=len(CATEGORIES), ncols=3, figsize=(10, 10))

    for i, category in enumerate(CATEGORIES):
        category_dir = os.path.join(DATA_DIR, category)
        image_files = [os.path.join(category_dir, file) for file in os.listdir(category_dir) if file.endswith(".jpg") or file.endswith(".png")]
        image_files = image_files[7:10]  # Select first three images from each category

        for j, image_file in enumerate(image_files):
            img = Image.open(image_file)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            axes[i, j].set_title(category)
            
    plt.tight_layout()
    plt.show()



def plot_images(pca,PCA_X_train):
    pca_image = PCA_X_train[1]
    # pca_image_resized = cv2.resize(pca_image, IMAGE_SIZE)
    pca_image_recon = pca.inverse_transform(pca_image)
    image_reconstructed = pca_image_recon.reshape(IMAGE_SIZE)
    plt.imshow(image_reconstructed)
    plt.show()
    return None



def pie_chart_images(DATA_DIR):

    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    class_counts = {}
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))

    labels = list(class_counts.keys())
    sizes = list(class_counts.values())


    fig1, ax1 = plt.subplots()
    patches, texts, autotexts = ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
    for text in texts:
        text.set_color('black')
    for autotext in autotexts:
        autotext.set_color('black')
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.title('Classes Distribution')
    plt.show()


    # plt.figure(figsize=(6, 6))
    # plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    # plt.axis('equal')
    # plt.title('Classes Distribution')
    # plt.show()



def plot_hist_box_plot(DATA_DIR):
    image_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith(".jpg") or file.endswith(".png")]

    image_sizes = []
    for file in image_files:
        img = cv2.imread(file)
        height, width, _ = img.shape
        image_sizes.append((width, height))

    # Plotting the image size distribution
    sizes = list(zip(*image_sizes))

    heights = sizes[0]
    widths = sizes[1]


    plt.figure(figsize=(8, 6))
    plt.hist(sizes[0], bins=30, alpha=0.5, color='red', label='Width')
    plt.hist(sizes[1], bins=30, alpha=0.5, color='blue', label='Height')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.title('Image Size Distribution')
    plt.legend()
    plt.show()

    # Extract the width and height into separate lists
    # widths = [size[0] for size in image_sizes]
    # heights = [size[1] for size in image_sizes]

    # Plot histogram of image widths
    plt.figure(figsize=(10, 6))
    plt.hist(widths, bins=30, color='blue', alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Widths')
    plt.show()

    # Plot histogram of image heights
    plt.figure(figsize=(10, 6))
    plt.hist(heights, bins=30, color='green', alpha=0.5)
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Heights')
    plt.show()

    # Plot boxplot of image sizes
    plt.figure(figsize=(10, 6))
    plt.boxplot(image_sizes)
    plt.xticks([1], ['Image Sizes'])
    plt.ylabel('Size (Width, Height)')
    plt.title('Distribution of Image Sizes')
    plt.show()


def plot_distribution(DATA_DIR):
    # Get the list of classes (subfolders) in the folder
    classes = sorted(os.listdir(DATA_DIR))

    # Count the number of images in each class
    image_counts = [len(os.listdir(os.path.join(DATA_DIR, class_name))) for class_name in classes]

    # Create the bar chart
    plt.bar(classes, image_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Image Class Distribution')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust the spacing to prevent label overlapping
    plt.show()



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