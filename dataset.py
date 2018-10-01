import os
import numpy as np
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import json

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
    return images28

def saveJSON(file,images,labels):
    jsonData = {}
    jsonData['data'] = images.tolist()
    jsonData['label'] = labels
    with open (file,'w') as json_file:
        json.dump(jsonData, json_file, indent=4)

DATA_PATH = 'E:\GitHub\Belgian Traffic Signs\Data'
train_data_directory = os.path.join(DATA_PATH,'Training')
test_data_directory = os.path.join(DATA_PATH,'Testing')

images, labels = loadData(train_data_directory)
test_images, test_labels = loadData(test_data_directory)

images28 = convertData(images)
test_images28 = convertData(test_images)

saveJSON(os.path.join(DATA_PATH,'train.txt'),images28,labels)
saveJSON(os.path.join(DATA_PATH,'test.txt'),test_images28,test_labels)