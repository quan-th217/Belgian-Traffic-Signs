import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import randint

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

def showImages(predictions,labels):    
    plt.figure(figsize=(10, 10))
    for i in range(10):
        k = randint(0,2500)
        prediction = np.argmax(predictions[k])
        truth = labels[k]
        plt.subplot(5, 2,1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
                 fontsize=12, color=color)
        plt.imshow(test_images28[k],  cmap="gray")
    plt.show()

DATA_PATH = 'E:\GitHub\Belgian Traffic Signs\Data'
test_data_path = os.path.join(DATA_PATH,'test.txt')
test_images28, test_labels = loadJSON(test_data_path)

MODEL_PATH = 'E:\GitHub\Belgian Traffic Signs\Model'
model_path = os.path.join(MODEL_PATH,'my_model.h5')
model = createModel()
model.load_weights(model_path)

test_loss, test_acc = model.evaluate(test_images28, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images28)
showImages(predictions,test_labels)