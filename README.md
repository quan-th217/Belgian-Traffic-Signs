# BelgianTrafficSigns
A demonstration of using Keras to build a neural network to classify Belgian traffic signs

## Dataset
The dataset can be downloaded from [here](https://btsd.ethz.ch/shareddata/):
- [Training data](https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip)
- [Testing data](https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip)

## Run
1. Run dataset.py to load, resize and convert the images to grayscale, then serialize the data and write it to a JSON file
2. Run train.py to build and train the neural network
3. Run evaluate.py to use the trained model with the test data