import tensorflow as tf
from tensorflow import keras

class SimpleCNN:
    def buildModel():
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(20, kernel_size=(3,3),
                                      activation=tf.nn.relu,
                                      input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(20, kernel_size=(3,3),
                                      activation=tf.nn.relu))
        model.add(keras.layers.Flatten())        
        model.add(keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(keras.layers.Dense(62, activation=tf.nn.softmax))
        
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model