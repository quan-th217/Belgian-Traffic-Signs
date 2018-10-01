import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

ROOT_PATH = 'E:\GitHub\Belgian Traffic Signs'
images28, labels = loadJSON(ROOT_PATH + '\Data\\train.txt')

checkpoint_path = ROOT_PATH + '\Model\cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1, period=5)
model = createModel()
model.fit(images28, labels, epochs=25, callbacks = [cp_callback])
model.save(ROOT_PATH + '\Model\my_model.h5')