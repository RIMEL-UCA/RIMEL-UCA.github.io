import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
import tensorflow as tf
import pickle, time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D

DATADIR = os.getcwd() + "/Accident Images Analysis Dataset/Mydata/"
#add path for dataset here.


CATEGORIES = ["No", "Yes"] # and changed from ["Yes", "No"]
# dataset has to be split into the respective classes and the folders should be named accordingly

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img),0)  # convert to array
        plt.imshow(img_array)  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
ss = img_array
ss.shape

IMG_SIZE = 200

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

ss = new_array
new_array.shape
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
import random
random.shuffle(training_data)
# 0 indicates no accident and 1 indicates an accident has occured
for i in range (5):
    new_array = cv2.resize(training_data[i][0], (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    print(training_data[i][1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X.shape[0]
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
X = np.array(X).reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
from sklearn.model_selection import train_test_split
X = X/255.0
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

# model = Sequential()

# model.add(ZeroPadding2D((1, 1), input_shape=X.shape[1:]))
# model.add(Conv2D(16, (3, 3)))
# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(32,(3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, (3,3)))
# model.add(Activation("relu"))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation("relu"))
# model.add(Dense(1))
# model.add(Activation("sigmoid"))
# from tensorflow.keras.optimizers import SGD
# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)


# model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])


# model.fit(X_train,y_train,batch_size=32, epochs = 10, validation_data=(X_test, y_test),verbose=1)

# first model 
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=X.shape[1:]))
model.add(Conv2D(16, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train,y_train,batch_size=32, epochs = 10, validation_data=(X_test, y_test),verbose=1)

model.save('74%acc')
model = tf.keras.models.load_model('74%acc')

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

CATEGORIES = ["No", "Yes"]

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



prediction = model.predict([prepare('1789.jpg')])


a = numpy.array(prediction)
print(a)
prediction
prediction
xx = model.layers.pop()
xx
model.summary()

model2 = model.layers[:-1]
model2
xmodel = Sequential()
for layer in model.layers[:-1]: # just exclude last layer from copying
    xmodel.add(layer)
import tensorflow as tf


