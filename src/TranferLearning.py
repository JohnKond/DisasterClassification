import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt



# Batch size
BATCH_SIZE = 32

# Image size
IMG_SIZE = (160, 160)


# define pre-trained model preprocessesing method
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def TL_prefetch_dataset(train_dataset, validation_dataset, test_dataset):
    # prefetch dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, test_dataset

def TL_augment_dataset(dataset):
    # augment the dataset
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    return data_augmentation

def TL_read_data(DATA_DIR):
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
    train_dataset, validation_dataset, test_dataset = TL_prefetch_dataset(train_dataset, validation_dataset, test_dataset)    

    
    return train_dataset, validation_dataset, test_dataset
    
def TL_create_model(train_dataset):
    IMG_SHAPE = IMG_SIZE + (3,)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    # freeze base model
    base_model.trainable = False
    # base_model.summary()

    # add global average layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    # add global average layer
    prediction_layer = layers.Dense(4,activation='softmax')
    prediction_batch = prediction_layer(feature_batch_average)
    

    # for data_augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    # add input layer
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # print(feature_batch.shape)
    return model

def TL_compile_model(model,base_learning_rate, loss, metrics):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=base_learning_rate),
        loss=loss,
        metrics=metrics)
    model.summary()
    return model

def TL_plot_learning_curve(history):
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

def TL_train_model(model, train_dataset, validation_dataset):
    
    initial_epochs = 10

    loss0,accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
    
    


    TL_plot_learning_curve(history)
    return model


def TL_test_model(model,test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print("loss: {:.2f}".format(loss))
    print("accuracy: {:.2f}".format(accuracy))
