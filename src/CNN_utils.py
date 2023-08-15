import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.metrics import f1_score




# Batch size
BATCH_SIZE = 32

# Image size
IMG_SIZE = (160, 160)


def CNN_prefetch_dataset(train_dataset, validation_dataset, test_dataset):
    # prefetch dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, test_dataset

def CNN_augment_dataset(dataset):
    # augment the dataset
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    return data_augmentation

def CNN_read_data(DATA_DIR):
    # Create the training dataset with 80% of the data
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,  # Split 80% for training
        subset='training',
        seed=123,  # Set a random seed for reproducibility
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        shuffle=True,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Create the validation dataset with 10% of the data
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,  # Split 10% for validation
        subset='validation',
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        seed=123,  # Set the same random seed for consistency
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # create test set
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    # prefetch dataset
    train_dataset, validation_dataset, test_dataset = CNN_prefetch_dataset(train_dataset, validation_dataset, test_dataset)    

    
    return train_dataset, validation_dataset, test_dataset
    

# to do add learning_rate parameter
def CNN_compile_model(model, optimizer, loss, metrics):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    model.summary()
    return model


def CNN_plot_learning_curve(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def CNN_train_model(model, train_dataset, validation_dataset):
    
    initial_epochs = 20
    

    loss0,accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
    
    
    CNN_plot_learning_curve(history)
    return model


def get_predictions(model,test_dataset):
    y_true = []  # true labels for the test set
    y_pred = []  # predicted labels for the test set

    for images, labels in test_dataset:
        # Make predictions using the trained model
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Convert one-hot encoded labels to integers
        true_labels = np.argmax(labels.numpy(), axis=1)
        
        # Append the true and predicted labels
        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)
    return y_true, y_pred



def calculate_ROC(model,test_dataset):
    for images, labels in test_dataset:
        y_scores = model.predict(images)

        # Compute the micro-averaged ROC curve and AUC
        fpr, tpr, _ = metrics.roc_curve(labels.ravel(), y_scores.ravel())
        roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # fpr = {}
    # tpr = {}
    # roc_auc = {}

    # for i in range(4):
    #     fpr[i], tpr[i], _ = metrics.roc_curve(labels[:, i], y_scores[:, i])
    #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    # plt.figure()
    # colors = ['blue', 'red', 'green', 'orange']  # Customize colors for each class

    # for i in range(4):
    #     plt.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

    # plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()





def CNN_test_model(model,test_dataset):


    # calculate_ROC(model, test_dataset)
    y_true, y_pred = get_predictions(model, test_dataset)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_true, y_pred)
    
    
    # CNN_plot_learning_curve(model)

    print("f1-score : {:.2f}".format(f1))
    print("accuracy : {:.2f}".format(accuracy))

    # loss, accuracy = model.evaluate(test_dataset)
    # print("loss: {:.2f}".format(loss))
    # print("accuracy: {:.2f}".format(accuracy))
