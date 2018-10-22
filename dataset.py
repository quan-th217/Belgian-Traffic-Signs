import os
import numpy as np
from skimage import data
from skimage import transform
from skimage.color import rgb2gray

import json_data as js

def loadData(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

def convertData(images):
    images28 = [transform.resize(image, (28, 28)) for image in images]
    images28 = rgb2gray(np.array(images28))
    # Convert the pixel values to int to reduce the size of the data files
    images28 = images28*256
    images28 = images28.astype(int)
    return images28

# Load, resize and convert the images to grayscale
DATA_PATH = 'Data'
train_data_directory = os.path.join(DATA_PATH,'Training')
test_data_directory = os.path.join(DATA_PATH,'Testing')

images, labels = loadData(train_data_directory)
test_images, test_labels = loadData(test_data_directory)

images28 = convertData(images)
test_images28 = convertData(test_images)

# Save processed data to JSON files
js.save(os.path.join(DATA_PATH,'train.txt'),images28,labels)
js.save(os.path.join(DATA_PATH,'test.txt'),test_images28,test_labels)