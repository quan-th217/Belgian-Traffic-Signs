import tensorflow as tf
from tensorflow import keras

class SimpleNN:
    def buildModel():
        model = keras.Sequential()
        
        model.add(keras.layers.Flatten(input_shape=(28, 28)))
        model.add(keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(keras.layers.Dense(62, activation=tf.nn.softmax))
        
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model