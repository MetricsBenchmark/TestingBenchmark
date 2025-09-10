import time
import os
import sys
import datetime
import numpy as np
import keras
from keras.models import Model
import random
from keras.datasets import mnist
from numpy import arange
import argparse
from keras.applications import vgg19,resnet50

basedir = os.path.dirname(os.path.abspath(__file__))

threshold = 0

def get_mnist(**kwargs):
    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test /= 255
    Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test,Y_test

def get_mnist_train(**kwargs):
    (X_train, Y_train), (_, _) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train /= 255
    Y_train = keras.utils.to_categorical(Y_train, 10)
    return X_train, Y_train

def get_adv_mnist(**kwargs):
    image_path = os.path.join(basedir,'data','adv_image/fgsm_bim_pgd_clean_mnist_image.npy')
    label_path = os.path.join(basedir,'data','adv_image/fgsm_bim_pgd_clean_mnist_label.npy')
    x_test = np.load(image_path)
    # x_test = x_test.astype('float32') / 255.0
    y_test = np.load(label_path)
    #y_test = keras.utils.to_categorical(y_test,num_classes=10)
    return x_test,y_test

def get_label_mnist(**kwargs):
    image_path = os.path.join(basedir,'data','label_shift_data/mnist_label_imgs.npy')
    label_path = os.path.join(basedir,'data','label_shift_data/mnist_label_labels.npy')
    x_test = np.load(image_path)
    # x_test = x_test.astype('float32') / 255.0
    y_test = np.load(label_path)
    #y_test = keras.utils.to_categorical(y_test,num_classes=10)
    return x_test,y_test

def get_corrupted_mnist(**kwargs):
    image_path = os.path.join(basedir,'data','corrupted_image/corrupted_clean_mnist_image.npy')
    label_path = os.path.join(basedir,'data','corrupted_image/corrupted_clean_mnist_label.npy')
    x_test = np.load(image_path)
    y_test = np.load(label_path)
    return x_test,y_test

def get_mnist_emnist(**kwargs):
    image_path = os.path.join(basedir, 'data', 'natural_shift_data/mnist_emnist_mix_imgs.npy')
    label_path = os.path.join(basedir, 'data', 'natural_shift_data/mnist_emnist_mix_labels.npy')
    x_test = np.load(image_path).astype('float32')
    y_test = np.load(label_path)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    # import pdb; pdb.set_trace()
    return x_test, y_test



def get_data(exp_id):
    exp_model_dict = {'lenet1': get_mnist,
                      'lenet4': get_mnist,
                      'lenet5': get_mnist,
                      'adv_mnist':get_adv_mnist,
                      'corrupted_mnist':get_corrupted_mnist,
                      'label_mnist':get_label_mnist,
                      'mnist_emnist':get_mnist_emnist,}
    return exp_model_dict[exp_id](exp_id=exp_id)


def get_model(exp_id):
    basedir = os.path.abspath(os.path.dirname(__file__))

    exp_model_dict = {'lenet1':'model/LeNet-1.h5',
                      'lenet4':'model/LeNet-4.h5',
                      'lenet5':'model/LeNet-5.h5',
                      'adv_mnist':'model/LeNet-5.h5',
                      'corrupted_mnist':'model/LeNet-5.h5',
                      'label_mnist':'model/LeNet-5.h5',
                      'mnist_emnist':'model/LeNet-5.h5',}
    my_model = keras.models.load_model(os.path.join(basedir,exp_model_dict[exp_id]))
    return my_model
