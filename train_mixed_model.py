import gzip
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Add, Lambda, Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import non_neg, max_norm, min_max_norm
from tensorflow.keras.initializers import Constant

from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.ops import nn
import tensorflow as tf
import random
import os
import glob

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.models import load_model
import h5py


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def get_fashion_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def get_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def train_fnn_sigmoid(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation=nn.sigmoid,
                      train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    # elif dataset == 'gtsrb':
    #     x_train, y_train, x_test, y_test = get_GTSRB_dataset()

    batch_size = 128

    print('activation: ', activation)

    model = Sequential()

    model.add(Flatten(input_shape=x_train.shape[1:]))
    for i in range(layer_num):
        model.add(Dense(nodes_per_layer))

        model.add(Lambda(lambda x: nn.sigmoid(x)))
        # model.add(Lambda(lambda x: nn.tanh(x)))
        # model.add(Lambda(lambda x: tf.atan(x)))

    model.add(Dense(10, activation='softmax'))

    # sgd = SGD(lr=0.1, decay=0.1/128, momentum=0.9, nesterov=True)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Traing a {} layer model, saving to {}".format(layer_num + 1, file_name))

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        epochs=num_epochs,
                        shuffle=True)

    # save model to a file
    if file_name != None:
        model.save(file_name + '.h5')

    return {'model': model, 'history': history}


def train_fnn_tanh(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation=nn.sigmoid,
                   train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    # elif dataset == 'gtsrb':
    #     x_train, y_train, x_test, y_test = get_GTSRB_dataset()

    batch_size = 128

    print('activation: ', activation)

    model = Sequential()

    model.add(Flatten(input_shape=x_train.shape[1:]))
    for i in range(layer_num):
        model.add(Dense(nodes_per_layer))

        # model.add(Lambda(lambda x: nn.sigmoid(x)))
        model.add(Lambda(lambda x: nn.tanh(x)))
        # model.add(Lambda(lambda x: tf.atan(x)))

    model.add(Dense(10, activation='softmax'))

    # sgd = SGD(lr=0.1, decay=0.1/128, momentum=0.9, nesterov=True)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Traing a {} layer model, saving to {}".format(layer_num + 1, file_name))

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        epochs=num_epochs,
                        shuffle=True)

    # save model to a file
    if file_name != None:
        model.save(file_name + '.h5')

    return {'model': model, 'history': history}


def train_fnn_atan(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation=nn.sigmoid,
                   train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    # elif dataset == 'gtsrb':
    #     x_train, y_train, x_test, y_test = get_GTSRB_dataset()

    batch_size = 128

    print('activation: ', activation)

    model = Sequential()

    model.add(Flatten(input_shape=x_train.shape[1:]))
    for i in range(layer_num):
        model.add(Dense(nodes_per_layer))

        # model.add(Lambda(lambda x: nn.sigmoid(x)))
        # model.add(Lambda(lambda x: nn.tanh(x)))
        model.add(Lambda(lambda x: tf.atan(x)))

    model.add(Dense(10, activation='softmax'))

    # sgd = SGD(lr=0.1, decay=0.1/128, momentum=0.9, nesterov=True)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Traing a {} layer model, saving to {}".format(layer_num + 1, file_name))

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        epochs=num_epochs,
                        shuffle=True)

    # save model to a file
    if file_name != None:
        model.save(file_name + '.h5')

    return {'model': model, 'history': history}


def train_cnn(file_name, dataset, filters, kernels, num_epochs=5, activation=nn.sigmoid, bn=False,
              train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    # elif dataset == 'gtsrb':
    #     x_train, y_train, x_test, y_test = get_GTSRB_dataset()

    batch_size = 128

    print('activation: ', activation)

    model = Sequential()
    model.add(Convolution2D(filters[0], kernels[0], activation=activation, input_shape=x_train.shape[1:]))
    for f, k in zip(filters[1:], kernels[1:]):
        model.add(Convolution2D(f, k, activation=activation))

    # the output layer, with 10 classes
    model.add(Flatten())
    if dataset == 'gtsrb':
        model.add(Dense(43, activation='softmax'))
    else:
        model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name))

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        epochs=num_epochs,
                        shuffle=True)

    # save model to a file
    if file_name != None:
        model.save(file_name + '.h5')

    return {'model': model, 'history': history}


def train_lenet(file_name, dataset, params, num_epochs=10, activation=nn.sigmoid, batch_size=128, train_temp=1,
                pool=True):
    """
    Standard neural network training procedure. Trains LeNet-5 style model with pooling optional.
    """
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    # elif dataset == 'gtsrb':
    #     x_train, y_train, x_test, y_test = get_GTSRB_dataset()

    img_rows, img_cols, img_channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_shape = (img_rows, img_cols, img_channels)

    model = Sequential()

    model.add(Convolution2D(params[0], (5, 5), activation=activation, input_shape=input_shape, padding='same'))
    if pool:
        model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(params[1], (5, 5), activation=activation))
    if pool:
        model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[2], activation=activation))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name + '.h5')

    return model


def printlog(s):
    print(s, file=open("cifar_cnn_5layer_5_3_sigmoid.txt", "a"), end='')


def print_weights(path_prefix, model_name):
    model = load_model(path_prefix + model_name, custom_objects={'fn': fn, 'tf': tf})
    model.summary()

    layer_num = 0

    for layer in model.layers:
        if type(layer) == Conv2D:
            printlog("layer num: {}\n".format(layer_num))
            layer_num += 1
            w, b = layer.get_weights()

            printlog("layer.name: {}, w.shape: {}\n".format(layer.name, w.shape))
            out_ch = w.shape[3]
            in_ch = w.shape[2]
            height = w.shape[0]
            width = w.shape[1]

            for i in range(out_ch):
                printlog("out_ch: {}\n".format(i))

                for j in range(in_ch):
                    printlog("in_ch: {}\n".format(j))
                    for m in range(height):
                        for n in range(width):
                            printlog("{}, ".format(w[m, n, j, i]))
                        printlog("\n")
                    printlog('---------------------------\n')
        elif (type(layer) == Dense):
            printlog("layer num: {}\n".format(layer_num))
            layer_num += 1
            w, b = layer.get_weights()

            printlog("layer.name: {}, w.shape: {}\n".format(layer.name, w.shape))
            out_ch = w.shape[1]
            in_ch = w.shape[0]

            for i in range(out_ch):
                printlog("out_ch: {}\n".format(i))

                for j in range(in_ch):
                    if (j % 6 == 0):
                        printlog("{} \n".format(w[j, i]))
                    else:
                        printlog("{}, ".format(w[j, i]))

                printlog("\n--------------------------\n")


def fnn():
    path_prefix = "models/models_with_positive_weights/sigmod/"

    # 3*50
    # 0.9998
    train_fnn_sigmoid(file_name=path_prefix + "mnist_ffnn_3x50_with_positive_weights", dataset='mnist',
                      layer_num=3, nodes_per_layer=50, num_epochs=300)
    # 0.9380
    train_fnn_sigmoid(file_name=path_prefix + "fashion_mnist_ffnn_3x50_with_positive_weights", dataset='fashion_mnist',
                      layer_num=3, nodes_per_layer=50, num_epochs=300)
    # 0.5852
    train_fnn_sigmoid(file_name=path_prefix + "cifar10_ffnn_3x50_with_positive_weights", dataset='cifar10',
                      layer_num=3, nodes_per_layer=50, num_epochs=300)

    # 3*100
    # 1
    train_fnn_sigmoid(file_name=path_prefix + "mnist_ffnn_3x100_with_positive_weights", dataset='mnist',
                      layer_num=3, nodes_per_layer=100, num_epochs=300)
    # 0.9618
    train_fnn_sigmoid(file_name=path_prefix + "fashion_mnist_ffnn_3x100_with_positive_weights", dataset='fashion_mnist',
                      layer_num=3, nodes_per_layer=100, num_epochs=300)
    # 0.6903
    train_fnn_sigmoid(file_name=path_prefix + "cifar10_ffnn_3x100_with_positive_weights", dataset='cifar10',
                      layer_num=3, nodes_per_layer=100, num_epochs=300)

    # 5*100
    # 0.9990
    train_fnn_sigmoid(file_name=path_prefix + "mnist_ffnn_5x100_with_positive_weights", dataset='mnist',
                      layer_num=5, nodes_per_layer=100, num_epochs=300)
    # 0.9608
    train_fnn_sigmoid(file_name=path_prefix + "fashion_mnist_ffnn_5x100_with_positive_weights", dataset='fashion_mnist',
                      layer_num=5, nodes_per_layer=100, num_epochs=300)
    # 0.6124
    train_fnn_sigmoid(file_name=path_prefix + "cifar10_ffnn_5x100_with_positive_weights", dataset='cifar10',
                      layer_num=5, nodes_per_layer=100, num_epochs=300)

    # 3*200
    # 1
    train_fnn_sigmoid(file_name=path_prefix + "mnist_ffnn_3x200_with_positive_weights", dataset='mnist',
                      layer_num=3, nodes_per_layer=200, num_epochs=300)
    # 0.9799
    train_fnn_sigmoid(file_name=path_prefix + "fashion_mnist_ffnn_3x200_with_positive_weights", dataset='fashion_mnist',
                      layer_num=3, nodes_per_layer=200, num_epochs=300)
    # 0.8028
    train_fnn_sigmoid(file_name=path_prefix + "cifar10_ffnn_3x200_with_positive_weights", dataset='cifar10',
                      layer_num=3, nodes_per_layer=200, num_epochs=300)

    # 3*400
    # 1
    train_fnn_sigmoid(file_name=path_prefix + "mnist_ffnn_3x400_with_positive_weights", dataset='mnist',
                      layer_num=3, nodes_per_layer=400, num_epochs=300)
    # 0.9867
    train_fnn_sigmoid(file_name=path_prefix + "fashion_mnist_ffnn_3x400_with_positive_weights", dataset='fashion_mnist',
                      layer_num=3, nodes_per_layer=400, num_epochs=300)
    # 0.9287
    train_fnn_sigmoid(file_name=path_prefix + "cifar10_ffnn_3x400_with_positive_weights", dataset='cifar10',
                      layer_num=3, nodes_per_layer=400, num_epochs=300)

    # 3*700
    # 1
    train_fnn_sigmoid(file_name=path_prefix + "mnist_ffnn_3x700_with_positive_weights", dataset='mnist',
                      layer_num=3, nodes_per_layer=700, num_epochs=300)
    # 0.9900
    train_fnn_sigmoid(file_name=path_prefix + "fashion_mnist_ffnn_3x700_with_positive_weights", dataset='fashion_mnist',
                      layer_num=3, nodes_per_layer=700, num_epochs=300)
    # 0.9719
    train_fnn_sigmoid(file_name=path_prefix + "cifar10_ffnn_3x700_with_positive_weights", dataset='cifar10',
                      layer_num=3, nodes_per_layer=700, num_epochs=300)

    ########################################################################################################################
    path_prefix = "models/models_with_positive_weights/tanh/"
    # 0.9998
    train_fnn_tanh(file_name=path_prefix + "mnist_ffnn_3x50_with_positive_weights_tanh", dataset='mnist',
                   layer_num=3, nodes_per_layer=50, num_epochs=300, activation=nn.tanh)
    # 0.9737
    train_fnn_tanh(file_name=path_prefix + "fashion_mnist_ffnn_3x50_with_positive_weights_tanh",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=50, num_epochs=300, activation=nn.tanh)
    # 0.6853
    train_fnn_tanh(file_name=path_prefix + "cifar10_ffnn_3x50_with_positive_weights_tanh", dataset='cifar10',
                   layer_num=3, nodes_per_layer=50, num_epochs=300, activation=nn.tanh)

    # 3*100
    # 1
    train_fnn_tanh(file_name=path_prefix + "mnist_ffnn_3x100_with_positive_weights_tanh", dataset='mnist',
                   layer_num=3, nodes_per_layer=100, num_epochs=300, activation=nn.tanh)
    # 0.9929
    train_fnn_tanh(file_name=path_prefix + "fashion_mnist_ffnn_3x100_with_positive_weights_tanh",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=100, num_epochs=300, activation=nn.tanh)
    # 0.8493
    train_fnn_tanh(file_name=path_prefix + "cifar10_ffnn_3x100_with_positive_weights_tanh", dataset='cifar10',
                   layer_num=3, nodes_per_layer=100, num_epochs=300, activation=nn.tanh)

    # 5*100
    # 1
    train_fnn_tanh(file_name=path_prefix + "mnist_ffnn_5x100_with_positive_weights_tanh", dataset='mnist',
                   layer_num=5, nodes_per_layer=100, num_epochs=300, activation=nn.tanh)
    # 0.9961
    train_fnn_tanh(file_name=path_prefix + "fashion_mnist_ffnn_5x100_with_positive_weights_tanh",
                   dataset='fashion_mnist',
                   layer_num=5, nodes_per_layer=100, num_epochs=300, activation=nn.tanh)
    # 0.9035
    train_fnn_tanh(file_name=path_prefix + "cifar10_ffnn_5x100_with_positive_weights_tanh", dataset='cifar10',
                   layer_num=5, nodes_per_layer=100, num_epochs=300, activation=nn.tanh)

    # 3*200
    # 1
    train_fnn_tanh(file_name=path_prefix + "mnist_ffnn_3x200_with_positive_weights_tanh", dataset='mnist',
                   layer_num=3, nodes_per_layer=200, num_epochs=300, activation=nn.tanh)
    # 0.994
    train_fnn_tanh(file_name=path_prefix + "fashion_mnist_ffnn_3x200_with_positive_weights_tanh",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=200, num_epochs=300, activation=nn.tanh)
    # 0.9926
    train_fnn_tanh(file_name=path_prefix + "cifar10_ffnn_3x200_with_positive_weights_tanh", dataset='cifar10',
                   layer_num=3, nodes_per_layer=200, num_epochs=300, activation=nn.tanh)

    # 3*400
    # 1
    train_fnn_tanh(file_name=path_prefix + "mnist_ffnn_3x400_with_positive_weights_tanh", dataset='mnist',
                   layer_num=3, nodes_per_layer=400, num_epochs=300, activation=nn.tanh)
    # 1
    train_fnn_tanh(file_name=path_prefix + "fashion_mnist_ffnn_3x400_with_positive_weights_tanh",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=400, num_epochs=300, activation=nn.tanh)
    # 0.9925
    train_fnn_tanh(file_name=path_prefix + "cifar10_ffnn_3x400_with_positive_weights_tanh", dataset='cifar10',
                   layer_num=3, nodes_per_layer=400, num_epochs=300, activation=nn.tanh)

    # 3*700
    # 1
    train_fnn_tanh(file_name=path_prefix + "mnist_ffnn_3x700_with_positive_weights_tanh", dataset='mnist',
                   layer_num=3, nodes_per_layer=700, num_epochs=300, activation=nn.tanh)
    # 1
    train_fnn_tanh(file_name=path_prefix + "fashion_mnist_ffnn_3x700_with_positive_weights_tanh",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=700, num_epochs=300, activation=nn.tanh)
    # 1
    train_fnn_tanh(file_name=path_prefix + "cifar10_ffnn_3x700_with_positive_weights_tanh", dataset='cifar10',
                   layer_num=3, nodes_per_layer=700, num_epochs=300, activation=nn.tanh)

    ########################################################################################################################

    path_prefix = "models/models_with_positive_weights/arctan/"
    #  3*50
    # 1
    train_fnn_atan(file_name=path_prefix + "mnist_ffnn_3x50_with_positive_weights_atan", dataset='mnist',
                   layer_num=3, nodes_per_layer=50, num_epochs=300, activation=tf.atan)
    # 0.9664
    train_fnn_atan(file_name=path_prefix + "fashion_mnist_ffnn_3x50_with_positive_weights_atan",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=50, num_epochs=300, activation=tf.atan)
    # 0.6737
    train_fnn_atan(file_name=path_prefix + "cifar10_ffnn_3x50_with_positive_weights_atan", dataset='cifar10',
                   layer_num=3, nodes_per_layer=50, num_epochs=300, activation=tf.atan)

    # 3*100
    # 1
    train_fnn_atan(file_name=path_prefix + "mnist_ffnn_3x100_with_positive_weights_atan", dataset='mnist',
                   layer_num=3, nodes_per_layer=100, num_epochs=300, activation=tf.atan)
    # 0.9893
    train_fnn_atan(file_name=path_prefix + "fashion_mnist_ffnn_3x100_with_positive_weights_atan",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=100, num_epochs=300, activation=tf.atan)
    # 0.8138
    train_fnn_atan(file_name=path_prefix + "cifar10_ffnn_3x100_with_positive_weights_atan", dataset='cifar10',
                   layer_num=3, nodes_per_layer=100, num_epochs=300, activation=tf.atan)

    # 5*100
    # 1
    train_fnn_atan(file_name=path_prefix + "mnist_ffnn_5x100_with_positive_weights_atan", dataset='mnist',
                   layer_num=5, nodes_per_layer=100, num_epochs=300, activation=tf.atan)
    # 0.9935
    train_fnn_atan(file_name=path_prefix + "fashion_mnist_ffnn_5x100_with_positive_weights_atan",
                   dataset='fashion_mnist',
                   layer_num=5, nodes_per_layer=100, num_epochs=300, activation=tf.atan)
    # 0.8680
    train_fnn_atan(file_name=path_prefix + "cifar10_ffnn_5x100_with_positive_weights_atan", dataset='cifar10',
                   layer_num=5, nodes_per_layer=100, num_epochs=300, activation=tf.atan)

    # 3*200
    # 1
    train_fnn_atan(file_name=path_prefix + "mnist_ffnn_3x200_with_positive_weights_atan", dataset='mnist',
                   layer_num=3, nodes_per_layer=200, num_epochs=300, activation=tf.atan)
    # 0.9954
    train_fnn_atan(file_name=path_prefix + "fashion_mnist_ffnn_3x200_with_positive_weights_atan",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=200, num_epochs=300, activation=tf.atan)
    # 0.9745
    train_fnn_atan(file_name=path_prefix + "cifar10_ffnn_3x200_with_positive_weights_atan", dataset='cifar10',
                   layer_num=3, nodes_per_layer=200, num_epochs=300, activation=tf.atan)

    # 3*400
    # 1
    train_fnn_atan(file_name=path_prefix + "mnist_ffnn_3x400_with_positive_weights_atan", dataset='mnist',
                   layer_num=3, nodes_per_layer=400, num_epochs=300, activation=tf.atan)
    # 1
    train_fnn_atan(file_name=path_prefix + "fashion_mnist_ffnn_3x400_with_positive_weights_atan",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=400, num_epochs=300, activation=tf.atan)
    # 1
    train_fnn_atan(file_name=path_prefix + "cifar10_ffnn_3x400_with_positive_weights_atan", dataset='cifar10',
                   layer_num=3, nodes_per_layer=400, num_epochs=300, activation=tf.atan)

    # 3*700
    # 1
    train_fnn_atan(file_name=path_prefix + "mnist_ffnn_3x700_with_positive_weights_atan", dataset='mnist',
                   layer_num=3, nodes_per_layer=700, num_epochs=300, activation=tf.atan)
    # 1
    train_fnn_atan(file_name=path_prefix + "fashion_mnist_ffnn_3x700_with_positive_weights_atan",
                   dataset='fashion_mnist',
                   layer_num=3, nodes_per_layer=700, num_epochs=300, activation=tf.atan)
    # 1
    train_fnn_atan(file_name=path_prefix + "cifar10_ffnn_3x700_with_positive_weights_atan", dataset='cifar10',
                   layer_num=3, nodes_per_layer=700, num_epochs=300, activation=tf.atan)


def cnn():
    path_prefix = "models/models_with_positive_weights/sigmod/"
    # 323
    # 0.9211
    train_cnn(file_name=path_prefix + "mnist_cnn_3layer_2_3_with_positive_weights", dataset='mnist',
              filters=[2, 2], kernels=[3, 3], num_epochs=50)
    # 0.8359
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_3layer_2_3_with_positive_weights", dataset='fashion_mnist',
              filters=[2, 2], kernels=[3, 3], num_epochs=50)
    # 0.3141
    train_cnn(file_name=path_prefix + "cifar10_cnn_3layer_2_3_with_positive_weights", dataset='cifar10',
              filters=[2, 2], kernels=[3, 3], num_epochs=50)

    # 343
    # 0.9381
    train_cnn(file_name=path_prefix + "mnist_cnn_3layer_4_3_with_positive_weights", dataset='mnist',
              filters=[4, 4], kernels=[3, 3], num_epochs=50)
    # 0.8568
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_3layer_4_3_with_positive_weights", dataset='fashion_mnist',
              filters=[4, 4], kernels=[3, 3], num_epochs=50)
    # 0.3908
    train_cnn(file_name=path_prefix + "cifar10_cnn_3layer_4_3_with_positive_weights", dataset='cifar10',
              filters=[4, 4], kernels=[3, 3], num_epochs=50)

    # 453
    # 0.9460
    train_cnn(file_name=path_prefix + "mnist_cnn_4layer_5_3_with_positive_weights", dataset='mnist',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50)
    # 0.8473
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_4layer_5_3_with_positive_weights",
              dataset='fashion_mnist',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50)
    # 0.3602
    train_cnn(file_name=path_prefix + "cifar10_cnn_4layer_5_3_with_positive_weights", dataset='cifar10',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50)
    # 553
    # 0.9424
    train_cnn(file_name=path_prefix + "mnist_cnn_5layer_5_3_with_positive_weights", dataset='mnist',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50)
    # 0.8332
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_5layer_5_3_with_positive_weights", dataset='fashion_mnist',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50)
    # 0.3340
    train_cnn(file_name=path_prefix + "cifar10_cnn_5layer_5_3_with_positive_weights", dataset='cifar10',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50)

    # 653
    # 0.9413
    train_cnn(file_name=path_prefix + "mnist_cnn_6layer_5_3_with_positive_weights", dataset='mnist',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50)
    # 0.8169
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_6layer_5_3_with_positive_weights", dataset='fashion_mnist',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50)
    # 0.3117
    train_cnn(file_name=path_prefix + "cifar10_cnn_6layer_5_3_with_positive_weights", dataset='cifar10',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50)

    ########################################################################################
    path_prefix = "models/models_with_positive_weights/tanh/"
    # 0.9474
    train_cnn(file_name=path_prefix + "mnist_cnn_3layer_2_3_with_positive_weights_tanh", dataset='mnist',
              filters=[2, 2], kernels=[3, 3], num_epochs=50, activation=nn.tanh)
    # 0.8730
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_3layer_2_3_with_positive_weights_tanh",
              dataset='fashion_mnist',
              filters=[2, 2], kernels=[3, 3], num_epochs=50, activation=nn.tanh)
    # 0.3952
    train_cnn(file_name=path_prefix + "cifar10_cnn_3layer_2_3_with_positive_weights_tanh", dataset='cifar10',
              filters=[2, 2], kernels=[3, 3], num_epochs=50, activation=nn.tanh)

    # 343
    # 0.9855
    train_cnn(file_name=path_prefix + "mnist_cnn_3layer_4_3_with_positive_weights_tanh", dataset='mnist',
              filters=[4, 4], kernels=[3, 3], num_epochs=50, activation=nn.tanh)
    # 0.8842
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_3layer_4_3_with_positive_weights_tanh",
              dataset='fashion_mnist',
              filters=[4, 4], kernels=[3, 3], num_epochs=50, activation=nn.tanh)
    # 0.4589
    train_cnn(file_name=path_prefix + "cifar10_cnn_3layer_4_3_with_positive_weights_tanh", dataset='cifar10',
              filters=[4, 4], kernels=[3, 3], num_epochs=50, activation=nn.tanh)

    # 453
    # 0.9893
    train_cnn(file_name=path_prefix + "mnist_cnn_4layer_5_3_with_positive_weights_tanh", dataset='mnist',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50, activation=nn.tanh)
    # 0.8999
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_4layer_5_3_with_positive_weights_tanh",
              dataset='fashion_mnist',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50, activation=nn.tanh)
    # 0.4889
    train_cnn(file_name=path_prefix + "cifar10_cnn_4layer_5_3_with_positive_weights_tanh", dataset='cifar10',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50, activation=nn.tanh)

    # 553
    # 0.9884
    train_cnn(file_name=path_prefix + "mnist_cnn_5layer_5_3_with_positive_weights_tanh", dataset='mnist',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50, activation=nn.tanh)
    # 0.8928
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_5layer_5_3_with_positive_weights_tanh",
              dataset='fashion_mnist',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50, activation=nn.tanh)
    # 0.4827
    train_cnn(file_name=path_prefix + "cifar10_cnn_5layer_5_3_with_positive_weights_tanh", dataset='cifar10',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50, activation=nn.tanh)

    # 653
    # 0.9890
    train_cnn(file_name=path_prefix + "mnist_cnn_6layer_5_3_with_positive_weights_tanh", dataset='mnist',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50, activation=nn.tanh)
    # 0.8909
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_6layer_5_3_with_positive_weights_tanh",
              dataset='fashion_mnist',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50, activation=nn.tanh)
    # 0.4600
    train_cnn(file_name=path_prefix + "cifar10_cnn_6layer_5_3_with_positive_weights_tanh", dataset='cifar10',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50, activation=nn.tanh)

    ############################################################################################
    path_prefix = "models/models_with_positive_weights/arctan/"
    # 0.9550
    train_cnn(file_name=path_prefix + "mnist_cnn_3layer_2_3_with_positive_weights_atan", dataset='mnist',
              filters=[2, 2], kernels=[3, 3], num_epochs=50, activation=tf.atan)
    # 0.8726
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_3layer_2_3_with_positive_weights_atan",
              dataset='fashion_mnist',
              filters=[2, 2], kernels=[3, 3], num_epochs=50, activation=tf.atan)
    # 0.4013
    train_cnn(file_name=path_prefix + "cifar10_cnn_3layer_2_3_with_positive_weights_atan", dataset='cifar10',
              filters=[2, 2], kernels=[3, 3], num_epochs=50, activation=tf.atan)

    # 343
    # 0.9828
    train_cnn(file_name=path_prefix + "mnist_cnn_3layer_4_3_with_positive_weights_atan", dataset='mnist',
              filters=[4, 4], kernels=[3, 3], num_epochs=50, activation=tf.atan)
    # 0.8816
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_3layer_4_3_with_positive_weights_atan",
              dataset='fashion_mnist',
              filters=[4, 4], kernels=[3, 3], num_epochs=50, activation=tf.atan)
    # 0.4516
    train_cnn(file_name=path_prefix + "cifar10_cnn_3layer_4_3_with_positive_weights_atan", dataset='cifar10',
              filters=[4, 4], kernels=[3, 3], num_epochs=50, activation=tf.atan)

    # 453
    # 0.9886
    train_cnn(file_name=path_prefix + "mnist_cnn_4layer_5_3_with_positive_weights_atan", dataset='mnist',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50, activation=tf.atan)
    # 0.8949
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_4layer_5_3_with_positive_weights_atan",
              dataset='fashion_mnist',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50, activation=tf.atan)
    # 0.4809
    train_cnn(file_name=path_prefix + "cifar10_cnn_4layer_5_3_with_positive_weights_atan", dataset='cifar10',
              filters=[5, 5, 5], kernels=[3, 3, 3], num_epochs=50, activation=tf.atan)

    # 553
    # 0.9872
    train_cnn(file_name=path_prefix + "mnist_cnn_5layer_5_3_with_positive_weights_atan", dataset='mnist',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50, activation=tf.atan)
    # 0.8907
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_5layer_5_3_with_positive_weights_atan",
              dataset='fashion_mnist',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50, activation=tf.atan)
    # 0.4827
    train_cnn(file_name=path_prefix + "cifar10_cnn_5layer_5_3_with_positive_weights_atan", dataset='cifar10',
              filters=[5, 5, 5, 5], kernels=[3, 3, 3, 3], num_epochs=50, activation=tf.atan)

    # 653
    # 0.9800
    train_cnn(file_name=path_prefix + "mnist_cnn_6layer_5_3_with_positive_weights_atan", dataset='mnist',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50, activation=tf.atan)
    # 0.8824
    train_cnn(file_name=path_prefix + "fashion_mnist_cnn_6layer_5_3_with_positive_weights_atan",
              dataset='fashion_mnist',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50, activation=tf.atan)
    # 0.4635
    train_cnn(file_name=path_prefix + "cifar10_cnn_6layer_5_3_with_positive_weights_atan", dataset='cifar10',
              filters=[5, 5, 5, 5, 5], kernels=[3, 3, 3, 3, 3], num_epochs=50, activation=tf.atan)


if __name__ == '__main__':
    fnn()
    cnn()
