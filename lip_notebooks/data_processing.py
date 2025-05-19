import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from deel.lip.layers import (
    SpectralDense,
    SpectralConv2D,
    ScaledL2NormPooling2D,
    FrobeniusDense,
)
from deel.lip.model import Sequential
from deel.lip.activations import GroupSort
from deel.lip.losses import MulticlassHKR, MulticlassKR
from keras.layers import Input, Flatten
from keras.optimizers import Adam
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical
import numpy as np
import keras.ops as K

def prepare_data_MNIST08(x,y):
    # select items from the two selected classes
    mask = (y == 0) + (
         y == 8
    )  # mask to select only items from class_a or class_b
    x = x[mask]
    y = y[mask]
    x = x.astype("float32")
    y = y.astype("float32")
    # convert from range int[0,255] to float32[-1,1]
    x /= 255
    x = x.reshape((-1, 28, 28, 1))
    # change label to binary classification {-1,1}
    y[y == 0] = 1.0
    y[y == 8] = 0.0
    # print(x.shape, y.shape)
    return x, y.reshape((-1,1))

def load_data(dataset):
    if dataset == "FMNIST":
        # load data
        (x_train, y_train_ord), (x_test, y_test_ord) = fashion_mnist.load_data()
        # standardize and reshape the data
        x_train = np.expand_dims(x_train, -1) / 255
        x_test = np.expand_dims(x_test, -1) / 255
        # one hot encode the labels
        y_train = to_categorical(y_train_ord)
        y_test = to_categorical(y_test_ord)
        # channel first
        x_train = np.transpose(x_train,(0,3,1,2))
        x_test = np.transpose(x_test,(0,3,1,2))
    elif dataset == "MNIST":
        # load data
        (x_train, y_train_ord), (x_test, y_test_ord) = mnist.load_data()
        # standardize and reshape the data
        x_train = np.expand_dims(x_train, -1) / 255
        x_test = np.expand_dims(x_test, -1) / 255
        # one hot encode the labels
        y_train = to_categorical(y_train_ord)
        y_test = to_categorical(y_test_ord)
        # channel first
        x_train = np.transpose(x_train,(0,3,1,2))
        x_test = np.transpose(x_test,(0,3,1,2))
    elif dataset == "MNIST08":
        # now we load the dataset
        (x_train, y_train_ord), (x_test, y_test_ord) = mnist.load_data()

        # prepare the data
        x_train, y_train = prepare_data_MNIST08(x_train, y_train_ord)
        x_test, y_test = prepare_data_MNIST08(x_test, y_test_ord)

        y_test_ord = y_test[:,0]
        y_train_ord = y_train[:,0]

        y_test = to_categorical(y_test)
        y_train = to_categorical(y_train)

        x_train = np.transpose(x_train,(0,3,1,2))
        x_test = np.transpose(x_test,(0,3,1,2))
    else:
        print("Please select an existing dataset")
    return x_train, x_test, y_train, y_test, y_test_ord

def select_data_for_radius_evaluation(x_test, y_test_ord, model):
    # strategy: first
    # we select a sample from each class.
    images_list = []
    labels_list = []
    idx_list = []
    # select only a few element from the test set
    # selected = np.random.choice(len(y_test_ord), 500)
    sub_y_test_ord = y_test_ord[:400]
    sub_x_test = x_test[:400]
    # drop misclassified elements
    misclassified_mask = K.equal(
        K.argmax(model.predict(sub_x_test, verbose=0), axis=-1), sub_y_test_ord
    )
    misclassified_mask = misclassified_mask.detach().cpu().numpy()

    idx_global = np.arange(len(sub_x_test))[misclassified_mask]
    sub_x_test = sub_x_test[misclassified_mask]
    sub_y_test_ord = sub_y_test_ord[misclassified_mask]

    for i in range(10):
        # select the 20 firsts elements of the ith label
        label_mask = sub_y_test_ord == i
        x = sub_x_test[label_mask][:20]
        y = sub_y_test_ord[label_mask][:20]
        idx = idx_global[label_mask][:20]
        # convert it to tensor for use with foolbox
        images = K.convert_to_tensor(x.astype("float32"), dtype="float32")
        labels = K.convert_to_tensor(y, dtype="int64")
        # repeat the input 10 times, one per misclassification target
        for j in range(20):
            images_list.append(images[j])
            labels_list.append(labels[j])
            idx_list.append(idx[j])
    images = K.convert_to_tensor(images_list)
    labels = K.convert_to_tensor(labels_list)
    return images, labels, idx_list 

def select_data_for_radius_evaluation_MNIST08(x_test, y_test_ord, model):
    # strategy: first
    # we select a sample from each class.
    images_list = []
    labels_list = []
    idx_list = []
    # select only a few element from the test set
    # selected = np.random.choice(len(y_test_ord), 500)
    sub_y_test_ord = y_test_ord[:400]
    sub_x_test = x_test[:400]
    # drop misclassified elements
    misclassified_mask = K.equal(
        K.argmax(model.predict(sub_x_test, verbose=0), axis=-1), sub_y_test_ord
    )

    misclassified_mask = misclassified_mask.detach().cpu().numpy()

    idx_global = np.arange(len(sub_x_test))[misclassified_mask]
    sub_x_test = sub_x_test[misclassified_mask]
    sub_y_test_ord = sub_y_test_ord[misclassified_mask]
    for i in range(2):
        # select the 20 firsts elements of the ith label
        label_mask = sub_y_test_ord == i
        x = sub_x_test[label_mask][:100]
        y = sub_y_test_ord[label_mask][:100]
        idx = idx_global[label_mask][:100]
        # convert it to tensor for use with foolbox
        images = K.convert_to_tensor(x.astype("float32"), dtype="float32")
        labels = K.convert_to_tensor(y, dtype="int64")
        # repeat the input 10 times, one per misclassification target
        # print(images.shape)
        for j in range(100):
            images_list.append(images[j])
            labels_list.append(labels[j])
            idx_list.append(idx[j])
    images = K.convert_to_tensor(images_list)
    labels = K.convert_to_tensor(labels_list)
    return images, labels, idx_list 