import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import json_data as js
from simple_cnn import SimpleCNN as nn

def trainingGraph(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()   
    plt.grid()
    plt.xlim(left=0)
    plt.ylim(0.5,1)
    plt.savefig("trainingGraph.png")
    plt.show()

# Load and prepare data
DATA_PATH = 'Data'
train_data_path = os.path.join(DATA_PATH,'train.txt')
images28, labels = js.load(train_data_path)
images28 = images28/256
images28 = images28.reshape(len(images28),28,28,1)

images_train, images_val, labels_train, labels_val = train_test_split(images28, labels,
                                                                      test_size = 0.2,
                                                                      random_state = 0)
# Prepare the callbacks for the model
MODEL_PATH = 'Model'
checkpoint_path = os.path.join(MODEL_PATH,'cnn-cp-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create, train and save the model
model = nn.buildModel()
history = model.fit(images_train, labels_train,
                    epochs = 20,
                    validation_data=(images_val, labels_val),
                    callbacks = [cp_callback])

trainingGraph(history)