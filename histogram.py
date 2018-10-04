import os
import json
import numpy as np
import matplotlib.pyplot as plt

def loadJSON(file):
    with open(file) as json_file:
        jsonData = json.load(json_file)
        images = np.array(jsonData['data'])
        labels = jsonData['label']
    return images, labels

DATA_PATH = 'E:\GitHub\Belgian Traffic Signs\Data'
train_data_path = os.path.join(DATA_PATH,'train.txt')
images28, labels = loadJSON(train_data_path)
test_data_path = os.path.join(DATA_PATH,'test.txt')
test_images28, test_labels = loadJSON(test_data_path)

plt.figure(0)
plt.hist(labels,bins=62)
plt.xlabel("Labels")
plt.ylabel("Count")
plt.title('Training data')

plt.figure(1)
plt.hist(test_labels,bins=62)
plt.xlabel("Labels")
plt.ylabel("Count")
plt.title('Testing data')