import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint

import json_data as js
from better_cnn import BetterCNN as nn

def showImages(predictions, images, labels):    
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
        plt.imshow(images[k],  cmap="gray")
    plt.show()

# Load and prepare data
DATA_PATH = 'Data'
test_data_path = os.path.join(DATA_PATH,'test.txt')
test_images28, test_labels = js.load(test_data_path)
test_images28 = test_images28/256
test_images28_wChannel = test_images28.reshape(len(test_images28),28,28,1)

# Load model
MODEL_PATH = 'Model'
model_path = os.path.join(MODEL_PATH,'cnn2-cp-0049.ckpt')
model = nn.buildModel()
model.load_weights(model_path)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images28_wChannel, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images28_wChannel)
showImages(predictions, test_images28, test_labels)