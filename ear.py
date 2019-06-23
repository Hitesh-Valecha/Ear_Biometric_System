from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import numpy as np
import cv2
import random

def ear_final():

    # train with the train folder images
    DATADIR = "./train/"
    CATEGORIES = ["Hitesh", "Labhesh", "Tarun", "Varkha"] #subjects/classes to be recognized

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)   # convert in gray scale for faster computation
            break
        break

    IMG_WIDTH = 60      # small size images (not more than 200 * 200)
    IMG_HEIGHT = 100
    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))

    training_data = []

    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)  # assign distinct index to all classes

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # convert in gray scale for faster computation
                    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))  # make all images of standard/same size
                    training_data.append([new_array, class_num])    # assign distinct index to all classes
                except Exception as e:
                    pass

    print("Training Data")
    create_training_data()
    print(len(training_data))

    random.shuffle(training_data)   # shuffle for better training and learning of the machine

    a = []      #feature set
    b = []      #labels

    for features, labels in training_data:
        a.append(features)
        b.append(labels)

    a = np.array(a).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    print(a.shape)
    np.random.seed(1000)

    X_train = a
    Y_train = b

    X_train = X_train/255.0

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=4, strides=1,activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
    model.add(Conv2D(32, kernel_size=4, strides=2,activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=4, strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.summary()

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, to_categorical(Y_train), batch_size=32, epochs=5)
    # increase the epochs or decrease the batch size according to classes

    # test with the test folder images
    DATADIR = "./test/"
    CATEGORIES = ["Hitesh", "Labhesh", "Tarun", "Varkha"]

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            break
        break

    IMG_WIDTH = 60
    IMG_HEIGHT = 100
    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))

    testing_data = []

    def create_testing_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                    testing_data.append([new_array, class_num])
                except Exception as e:
                    pass

    print("Testing Data")
    create_testing_data()
    print(len(testing_data))

    random.shuffle(testing_data)

    p = []      #feature set
    q = []      #labels

    for features, labels in testing_data:
        p.append(features)
        q.append(labels)

    p = np.array(p).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    X_test = p
    Y_test = q

    X_test = X_test/255.0

    # Evaluate the model
    scores = model.evaluate(X_test, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    model.save('model_opt.h5')

ear_final()