import tensorflow as tf
from tensorflow.keras import layers,models


IMG_SIZE = (160,160)

def format(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32)
  image = image/255
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

def CNN_create_model():
    model = models.Sequential()
    # input_shape=(32, 32, 3) --> images of 32x32 size and 3 channels
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) # layers.Conv2D(32, (3, 3) --> 32 filters of size 3x3 each
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # add Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4)) # number of classes
    model.summary()
    return model


def CNN_train_model(model, train_images, train_labels, val_images, val_labels):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(val_images, val_labels))
    

def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=3)
    print(test_acc)
