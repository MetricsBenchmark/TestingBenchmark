# -----------------------------------------------------------------------------------------
#  Challenge #2 -  epoch_model.py - Model Structure
# -----------------------------------------------------------------------------------------

import sys
import tensorflow.keras as keras
from keras.models import Model, Sequential
#from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
#from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

def build_cnn(input_tensor=None,weights_path=None):
    img_input = input_tensor
    if input_tensor is None:
        img_input = Input(shape=(100,100,3))

    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(img_input, y)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss = 'mse')

    if weights_path:
        model.load_weights(weights_path)

    return model


def build_InceptionV3(image_size=None,weights_path=None):
    image_size = image_size or (299, 299)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )
    bottleneck_model = InceptionV3(weights='imagenet',include_top=False, 
                                   input_tensor=Input(input_shape))
    for layer in bottleneck_model.layers:
        layer.trainable = False

    x = bottleneck_model.input
    y = bottleneck_model.output
    # There are different ways to handle the bottleneck output
    y = GlobalAveragePooling2D()(x)
    #y = AveragePooling2D((8, 8), strides=(8, 8))(x)
    #y = Flatten()(y)
    #y = BatchNormalization()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(input=x, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
    return model

if __name__ == '__main__':
    from data_utils import *
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--mode", type=str, help="training or testing")
    console_flags, unparsed = parse.parse_known_args(sys.argv[1:])
    mode = console_flags.mode

    # train the model
    batch_size = 128
    nb_epoch = 50

    model = build_cnn()
    # the data, shuffled and split between train and test sets
    root = ''
    train_generator, train_samples = load_train_data(path=root+'/Ch2_002/', batch_size=batch_size, shape=(100, 100))
    test_generator, test_samples = load_test_data(path=root+'/testing/', batch_size=batch_size, shape=(100, 100))
    if mode == 'train':
        checkpoint = ModelCheckpoint(
            filepath='epoch.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,  
            mode='min',  
            verbose=1
        )
        model.fit_generator(train_generator,
                            validation_data=test_generator,
                            validation_steps=test_samples // batch_size,
                            steps_per_epoch= train_samples // batch_size,
                            epochs=nb_epoch,
                            workers=8,
                            callbacks=[checkpoint],
                            use_multiprocessing=True,
                            verbose=1)
        print(bcolors.OKGREEN + 'Model trained' + bcolors.ENDC)
    if mode == 'test':
        model.load_weights("epoch.h5")

        # evaluation
        K.set_learning_phase(0)

        results = model.evaluate_generator(test_generator,
                                 steps=math.ceil(test_samples * 1. / batch_size))
        print('mse loss for testing data', results)

    # save model
    #model.save_weights('./epoch.h5')