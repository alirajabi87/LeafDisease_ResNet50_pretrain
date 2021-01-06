from __future__ import print_function, division
from builtins import range, input
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, \
    Input, ZeroPadding2D, MaxPool2D, AvgPool2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ResNet:
    """
    This class is for creating ResNet architecture
    for convolution neural network.
    """

    def __init__(self, input_shape, classes):
        """
        Class for ResNet50
        Augments:
                input_shape = tuple, image size: (64, 64, 3)
                classes = integer, number of classes to identify
        """
        self.input_shape = input_shape
        self.classes = classes
        # define the name basis of this layer

    def identityBlock(self, X1, filters, kernel=(1, 1), stride=2, stage=1, block="a"):
        """
        Identity Block of ResNet50

        Augments:
            X_input = input tensor
            kernel = integer, specifying the shape of the middle Conv's window for the main path
            filter = python list of integers, defining the number of filters in the CONV layers of the main path
            stride = integer, specifying the strides of the convolution along the height and width.
            stage = integer, used to name the layers, depending on their position in the network
            block = string/character, used to name the layers, depending on their position in the network

        return: X
        """
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # First component of the main path
        x = Conv2D(filters=filters[0],
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '_I',
                   kernel_initializer=glorot_uniform(seed=0))(X1)
        x = BatchNormalization(axis=3, name=bn_name_base + '_I')(x)
        x = Activation('relu')(x)

        # Second component of the main path
        x = Conv2D(filters=filters[1],
                   kernel_size=kernel,
                   strides=(1, 1),
                   padding='same',
                   name=conv_name_base + '_II',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '_II')(x)
        x = Activation('relu')(x)

        # Third component of the main path
        x = Conv2D(filters=filters[2],
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '_III',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '_III')(x)

        # Final Step: Add original value to x and path it through activation layer
        x = Add()([x, X1])
        x = Activation('relu')(x)

        return x

    def convBlock(self, X1, filters, kernel=(1, 1), stride=2, stage=1, block="a"):
        """
                Conv Block

                Augments:
                    X_input = input tensor
                    kernel = integer, specifying the shape of the middle Conv's window for the main path
                    filter = python list of integers, defining the number of filters in the CONV layers of the main path
                    stride = integer, specifying the strides of the convolution along the height and width.
                    stage = integer, used to name the layers, depending on their position in the network
                    block = string/character, used to name the layers, depending on their position in the network

                return: X
                """
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # First path
        x = Conv2D(filters=filters[0],
                   kernel_size=(1, 1),
                   strides=(stride, stride),
                   padding='valid',
                   name=conv_name_base + '_I',
                   kernel_initializer=glorot_uniform(seed=0))(X1)
        x = BatchNormalization(axis=3, name=bn_name_base + '_I')(x)
        x = Activation('relu')(x)

        # second path
        x = Conv2D(filters=filters[1],
                   kernel_size=kernel,
                   strides=(1, 1),
                   padding='same',
                   name=conv_name_base + '_II',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '_II')(x)
        x = Activation('relu')(x)

        # Third path
        x = Conv2D(filters=filters[2],
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '_III',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '_III')(x)

        # Shortcut Path
        # Third component of the main path
        x_shortcut = Conv2D(filters=filters[2],
                            kernel_size=(1, 1),
                            strides=(stride, stride),
                            padding='valid',
                            name=conv_name_base + '_1',
                            kernel_initializer=glorot_uniform(seed=0))(X1)
        x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '_1')(x_shortcut)

        # Final Step: Add x and x_shortcut and pass it through activation layer
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)

        return x

    def ResNet50(self):
        # Define the input
        X_input = Input(self.input_shape)

        # Zero-padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1 Conv + BN + relu + Maxpool2D
        X = Conv2D(filters=64,
                   kernel_size=(7, 7),
                   strides=2,
                   padding='same',
                   kernel_initializer=glorot_uniform(seed=0),
                   name='Conv_Stage1')(X)
        X = BatchNormalization(axis=3, name='BN_stage1')(X)
        X = Activation('relu')(X)
        X = MaxPool2D((2, 2), strides=2)(X)

        # Stage 2 ConvBlock + 2 * IdBlock (X, filter_list, kernel_Size, stride)
        X = self.convBlock(X, filters=[64, 64, 256], kernel=(3, 3), stride=1, stage=2, block='a')
        X = self.identityBlock(X, filters=[64, 64, 256], kernel=(3, 3), stride=1, stage=2, block='b')
        X = self.identityBlock(X, filters=[64, 64, 256], kernel=(3, 3), stride=1, stage=2, block='c')

        # Stage 3 ConvBlock + 3 * IdBlock
        X = self.convBlock(X, filters=[128, 128, 512], kernel=(3, 3), stride=2, stage=3, block='a')
        X = self.identityBlock(X, filters=[128, 128, 512], kernel=(3, 3), stride=2, stage=3, block='b')
        X = self.identityBlock(X, filters=[128, 128, 512], kernel=(3, 3), stride=2, stage=3, block='c')
        X = self.identityBlock(X, filters=[128, 128, 512], kernel=(3, 3), stride=2, stage=3, block='d')

        # Stage 4 ConvBlock + 5 * IdBlock
        X = self.convBlock(X, filters=[256, 256, 1024], kernel=(3, 3), stride=2, stage=4, block='a')
        X = self.identityBlock(X, filters=[256, 256, 1024], kernel=(3, 3), stride=2, stage=4, block='b')
        X = self.identityBlock(X, filters=[256, 256, 1024], kernel=(3, 3), stride=2, stage=4, block='c')
        X = self.identityBlock(X, filters=[256, 256, 1024], kernel=(3, 3), stride=2, stage=4, block='d')
        X = self.identityBlock(X, filters=[256, 256, 1024], kernel=(3, 3), stride=2, stage=4, block='e')
        X = self.identityBlock(X, filters=[256, 256, 1024], kernel=(3, 3), stride=2, stage=4, block='f')

        # Stage 5 ConvBlock + 2 * IdBlock
        X = self.convBlock(X, filters=[512, 512, 2048], kernel=(3, 3), stride=2, stage=5, block='a')
        X = self.identityBlock(X, filters=[512, 512, 2048], kernel=(3, 3), stride=2, stage=5, block='b')
        X = self.identityBlock(X, filters=[512, 512, 2048], kernel=(3, 3), stride=2, stage=5, block='c')

        # Average Pooling
        X = AvgPool2D((2, 2), name='average_pooling')(X)

        # Flatten + Dense
        X = Flatten()(X)
        # X = Dropout(0.2)(X)
        X = Dense(1024, activation='relu')(X)
        X = Dropout(0.5)(X)
        X = Dense(1024, activation='relu')(X)
        X = Dropout(0.5)(X)
        X = Dense(self.classes, activation='softmax',
                  name='fc' + str(self.classes), kernel_initializer=glorot_uniform(seed=0))(X)

        # Create Model and return it
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model


if __name__ == '__main__':
    from glob import glob
    import pydot
    Train_path = '../DATA/LeafDisease/Datasets'
    trainImage = glob(Train_path+'/*/*.jp*g')
    folders = glob(Train_path+'/*')

    pathRandom = np.random.choice(trainImage)
    plt.imshow(load_img(pathRandom))
    plt.title(pathRandom.split('\\')[-2])
    plt.show()

    ImageSize = (100, 100, 3)

    gen = ImageDataGenerator(rotation_range=45,
                             zoom_range=0.1,
                             shear_range=0.1,
                             rescale=1. / 255,
                             height_shift_range=0.1,
                             width_shift_range=0.1,
                             horizontal_flip=True,
                             validation_split=0.25,
                             vertical_flip=True)

    batchSize = 2
    TestImageGen = gen.flow_from_directory(directory=Train_path,
                                           target_size=ImageSize[:2],
                                           batch_size=batchSize,
                                           subset='validation',
                                           shuffle=False)

    TrainImageGen = gen.flow_from_directory(directory=Train_path,
                                            target_size=ImageSize[:2],
                                            class_mode='categorical',
                                            shuffle=True,
                                            subset='training',
                                            batch_size=batchSize)

    ValidImageGen = gen.flow_from_directory(directory=Train_path,
                                            target_size=ImageSize[:2],
                                            class_mode='categorical',
                                            shuffle=True,
                                            subset='validation',
                                            batch_size=batchSize)



    # for x, y in ValidImageGen:
    #     plt.imshow(x[0])
    #     plt.show()
    #     break

    obj = ResNet(input_shape=ImageSize, classes=len(folders))
    model = obj.ResNet50()
    # print(model.summary())
    # plot_model(model,
    #            to_file="model.png",
    #            show_shapes=False,
    #            show_layer_names=True,
    #            rankdir="TB",
    #            expand_nested=False,
    #            dpi=96)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # path_checkpoint = './tmp'
    # if not os.path.exists(path_checkpoint):
    #     os.mkdir(path_checkpoint)

    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                                filepath='../MyFiles/LeafDisease.h5',
                                                # save_weights_only=True,
                                                mode='min',
                                                save_best_only='True',
                                                save_freq='epoch')
    epochs = 30
    Steps_train = TrainImageGen.n // batchSize
    Steps_valid = ValidImageGen.n // batchSize
    res = model.fit(TrainImageGen, epochs=epochs,
                    steps_per_epoch=Steps_train,
                    validation_data=ValidImageGen,
                    callbacks=[model_checkpoint_callback],
                    validation_steps=Steps_valid,
                    verbose=2)

    # model.save('LeafDisease.h5')
    plt.plot(res.history['accuracy'])
    plt.show()

    results = pd.DataFrame(model.history.history)
    results[['val_loss', 'loss']].plot()
    plt.title('loss')
    plt.show()

    from tensorflow.keras.models import load_model
    from sklearn.metrics import confusion_matrix, classification_report

    model = load_model('../MyFiles/LeafDisease.h5')

    label = []
    for k, v in TestImageGen.class_indices.items():
        label.append(k)

    print(label)

    print(model.evaluate(TestImageGen, verbose=2))
    pred = model.predict(TestImageGen,)
    pred = np.argmax(pred, axis=-1)

    print(classification_report(y_true=TestImageGen.classes, y_pred=pred, target_names=label))
"# LeafDisease_Inception" 
"# LeafDisease_ResNet50_pretrain" 
