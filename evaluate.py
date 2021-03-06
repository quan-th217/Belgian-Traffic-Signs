import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint

import json_data as js
from simple_nn import SimpleNN as nn

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

# Load model
MODEL_PATH = 'Model'
model_path = os.path.join(MODEL_PATH,'cp-0038.ckpt')
model = nn.buildModel()
model.load_weights(model_path)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images28, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images28)
showImages(predictions, test_images28, test_labels)