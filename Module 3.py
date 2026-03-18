from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

Classifier=Sequential()

Classifier.add(Convolution2D(32,3,3,input_shape=(225,225,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Convolution2D(128,3,3,activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Flatten())
Classifier.add(Dense(256, activation='relu'))
Classifier.add(Dense(4, activation='softmax'))

Classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
Classifier.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('dataset/Train',target_size=(225,225),batch_size=32,class_mode='categorical')

test_set=test_datagen.flow_from_directory('dataset/Test',target_size=(225,225),batch_size=32,class_mode='categorical')

from IPython.display import display

img_dims = 150
epochs = 60
batch_size = 32

Classifier.fit_generator( training_set, steps_per_epoch=training_set.samples // batch_size, 
           epochs=epochs, 
           validation_data=test_set,validation_steps=test_set.samples // batch_size)


import h5py 

Classifier.save('e.h5')

from keras.models import load_model

model=load_model('e.h5')

import numpy as np

from tensorflow.keras.preprocessing import image
test_image=image.load_img('c5.jpg',target_size=(225,225))

import matplotlib.pyplot as plt
img = plt.imshow(test_image)

test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=model.predict(test_image)

result

prediction = result[0]

classes=training_set.class_indices

classes

prediction=list(prediction)

prediction

classes=['plain', 'pothole']
    output = zip(classes, prediction)
    output = dict(output)
if output['plain']==1.0:
    print("plain")
elif output['pothole']==1.0:
    print("pothole")
