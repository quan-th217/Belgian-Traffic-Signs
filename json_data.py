import json
import numpy as np

def save(file,images,labels):
    jsonData = {}
    jsonData['data'] = images.tolist()
    jsonData['label'] = labels
    with open (file,'w') as json_file:
        json.dump(jsonData, json_file, indent=4)

def load(file):
    with open(file) as json_file:
        jsonData = json.load(json_file)
        images = np.array(jsonData['data'])
        labels = jsonData['label']
    return images, labels