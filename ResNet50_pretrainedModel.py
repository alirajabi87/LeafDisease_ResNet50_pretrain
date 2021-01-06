from __future__ import print_function, division
from builtins import range, input
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAvgPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop

ImageSize = (224, 224, 3)
batchSize = 2
path_Train = './DATA/LeafDisease/Datasets/'
folders = glob(path_Train + '/*')

gen = ImageDataGenerator(rotation_range=5,
                         shear_range=0.1,
                         zoom_range=0.1,
                         height_shift_range=0.1,
                         width_shift_range=0.1,
                         horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.2,
                         preprocessing_function=preprocess_input)

Test_gen = gen.flow_from_directory(path_Train,
                                   target_size=ImageSize[:2],
                                   interpolation='nearest',
                                   shuffle=False,
                                   batch_size=batchSize,
                                   subset='validation',
                                   class_mode='categorical')

Valid_gen = gen.flow_from_directory(path_Train,
                                    target_size=ImageSize[:2],
                                    interpolation='nearest',
                                    shuffle=True,
                                    batch_size=batchSize,
                                    subset='validation',
                                    class_mode='categorical')

Train_gen = gen.flow_from_directory(path_Train,
                                    target_size=ImageSize[:2],
                                    interpolation='nearest',
                                    shuffle=True,
                                    batch_size=batchSize,
                                    subset='training',
                                    class_mode='categorical')

labels = []

for k, v in Train_gen.class_indices.items():
    labels.append(k)

print(labels)
print(len(labels))

model_checkPoints = ModelCheckpoint(filepath='../MyFiles/LeafDisease.h5',
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=ImageSize)
# print(resnet.summary())

for layer in resnet.layers:
    layer.trainable = False

X = resnet.output
# X = GlobalAvgPool2D()(X)
X = Flatten()(X)
X = Dense(1024, activation='relu')(X)
X = Dropout(0.5)(X)
X = Dense(1024, activation='relu')(X)
X = Dropout(0.5)(X)
Predictions = Dense(len(folders), activation='softmax')(X)

model = Model(inputs=resnet.input, outputs=Predictions)

# print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

Steprs_epochs = Train_gen.n // batchSize
Steps_valid = Valid_gen.n // batchSize
epoch = 55
res = model.fit(Train_gen, epochs=epoch,
                steps_per_epoch=Steprs_epochs,
                validation_data=Valid_gen,
                validation_steps=Steps_valid,
                callbacks=[model_checkPoints],
                verbose=2)

df = pd.DataFrame(model.history.history)
df[['val_loss', 'loss']].plot()
plt.show()

df[['accuracy', 'val_accuracy']].plot()
plt.show()

model = load_model('../MyFiles/LeafDisease.h5')

pred = model.predict(Test_gen, verbose=2)
pred = np.argmax(pred, axis=1)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_true=Test_gen.classes, y_pred=pred, target_names=labels))

print(confusion_matrix(y_true=Test_gen.classes, y_pred=pred))
