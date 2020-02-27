import os
import sys
import idx2numpy
import numpy as np
from pathlib import Path

rootDir = Path(sys.path[0]).parent
train_images = str(rootDir) + "\\MNIST test data\\train-images.idx3-ubyte"
train_label = str(rootDir) + "\\MNIST test data\\train-labels.idx1-ubyte"
test_images = str(rootDir) + "\\MNIST test data\\t10k-images.idx3-ubyte"
test_label = str(rootDir) + "\\MNIST test data\\t10k-labels.idx1-ubyte"

train_image_data = idx2numpy.convert_from_file(train_images)
train_label_data = idx2numpy.convert_from_file(train_label)
test_image_data = idx2numpy.convert_from_file(test_images)
test_label_data = idx2numpy.convert_from_file(test_label)
items, rows, cols = train_image_data.shape
train_image_data = train_image_data.reshape(items, rows * cols)
