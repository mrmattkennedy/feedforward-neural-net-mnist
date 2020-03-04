import pdb
import sys
import math
import time
import argparse
import traceback
import idx2numpy
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense


def init_params():
    parser = argparse.ArgumentParser()

    # hyperparameters setting
    parser.add_argument('--alpha', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0003,
                        help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--n_x', type=int, default=22000,
                        help='number of inputs')
    parser.add_argument('--n_h', type=int, default=100,
                        help='number of hidden units')
    parser.add_argument('--n_o', type=int, default=1,
                        help='number of output units')
    parser.add_argument('--beta', type=float, default=0.98,
                        help='parameter for momentum')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size')
    parser.add_argument('--batches', type=int, default=5,
                        help='batch iterations')
    return parser.parse_args()


def init_data():
    #Get file paths
    rootDir = Path(sys.path[0]).parent
    train_images = str(rootDir) + "\\MNIST data\\train-images.idx3-ubyte"
    train_label = str(rootDir) + "\\MNIST data\\train-labels.idx1-ubyte"
    test_images = str(rootDir) + "\\MNIST data\\t10k-images.idx3-ubyte"
    test_label = str(rootDir) + "\\MNIST data\\t10k-labels.idx1-ubyte"

    #Read in from file to numpy array
    train_image_data = idx2numpy.convert_from_file(train_images)
    train_label_data = idx2numpy.convert_from_file(train_label)
    test_image_data = idx2numpy.convert_from_file(test_images)
    test_label_data = idx2numpy.convert_from_file(test_label)

    #Reshaped the inputs from 3D to 2D
    items, rows, cols = train_image_data.shape
    train_image_data = train_image_data.reshape(items, rows * cols)
    items, rows, cols = test_image_data.shape
    test_image_data = test_image_data.reshape(items, rows * cols)

    return train_image_data, train_label_data, test_image_data, test_label_data

def train():
    train_input, train_target, test_input, test_target = init_data()
    train_target = reshape(train_target)
    test_target = reshape(test_target)
    #pdb.set_trace()
    model = Sequential()
    model.add(Dense(500, input_shape=(784,), activation='sigmoid'))
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_input, train_target, epochs=100, batch_size=60000)
    _, accuracy = model.evaluate(test_input, test_target)

def reshape(target):
    #Get the shape of the output
    rows, cols = target.shape[0], 10

    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((rows, cols))

    #Change index of correct predictions to a 1
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1

    return reshaped_target



start_time = time.time()
opts = init_params()
train()
print("--- %s seconds ---" % (time.time() - start_time))
