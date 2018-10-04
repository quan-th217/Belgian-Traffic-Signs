import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def loadJSON(file):
    with open(file) as json_file:
        jsonData = json.load(json_file)
        images = np.array(jsonData['data'])
        labels = jsonData['label']
    return images, labels

def createModel():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(62, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def trainingGraph(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()    
    plt.show()

DATA_PATH = 'E:\GitHub\Belgian Traffic Signs\Data'
train_data_path = os.path.join(DATA_PATH,'train.txt')
images28, labels = loadJSON(train_data_path)

images_train, images_val, labels_train, labels_val = train_test_split(images28, labels,
                                                                      test_size = 0.2,
                                                                      random_state = 0)
MODEL_PATH = 'E:\GitHub\Belgian Traffic Signs\Model'
checkpoint_path = os.path.join(MODEL_PATH,'cp-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1, period=5)
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model = createModel()
history = model.fit(images_train, labels_train,
                    epochs = 200,
                    validation_data=(images_val, labels_val),
                    callbacks = [es_callback,cp_callback])

model_path = os.path.join(MODEL_PATH,'my_model.h5')
model.save(model_path)

trainingGraph(history)