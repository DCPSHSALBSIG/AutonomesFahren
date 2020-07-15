import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from skimage.draw import polygon
import PIL
import json
import base64
import os.path
from os.path import splitext
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
from keras_segmentation.models.segnet import segnet
from keras_segmentation.models.unet import unet
from keras.models import model_from_json
from keras.models import load_model, save_model

colors = {0 : (0,0,0), 
          1 : (0,0,255), 
          2 : (0,255,0),
          3 : (255,0,0), 
          4 : (0,255,255),         
         }

dim = (256, 256) 

path =  r'D:\dev\dcps\AutonomesFahren\data'
dirimages = "images_augmented"
dirmasks = "masks_augmented"
dirmodels = "models"

dirimagesvalid = "images_valid"
dirmasksvalid = "masks_valid"

dirimagestest = "images_test"
dirmaskstest = "masks_test"

dirpredictions = "predictions"

fullpathimages = os.path.join(path, dirimages)
fullpathmasks = os.path.join(path, dirmasks)
fullpathpredictions = os.path.join(path, dirpredictions)

fullpathimagesvalid = os.path.join(path, dirimagesvalid)
fullpathmasksvalid = os.path.join(path, dirmasksvalid)

fullpathimagestest = os.path.join(path, dirimagestest)
fullpathmaskstest = os.path.join(path, dirmaskstest)

print(os.path.exists(fullpathimagesvalid))
print(os.path.exists(fullpathmasks))
print(os.path.exists(fullpathpredictions))
print(os.path.exists(os.path.join(path, dirmodels)))

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, ZeroPadding2D, UpSampling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def my_unet(classes):
    
    dropout = 0.4
    input_img = Input(shape=(dim[0], dim[1], 3))
    
    #contracting
    x = (ZeroPadding2D((1, 1)))(input_img)
    x = (Conv2D(64, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    c0 = Dropout(dropout)(x)
    
    x = (ZeroPadding2D((1, 1)))(c0)
    x = (Conv2D(128, (3, 3),padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    c1 = Dropout(dropout)(x)

    x = (ZeroPadding2D((1, 1)))(c1)
    x = (Conv2D(256, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    c2 = Dropout(dropout)(x)
    
    x = (ZeroPadding2D((1, 1)))(c2)
    x = (Conv2D(256, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    c3 = Dropout(dropout)(x)
    
    x = (ZeroPadding2D((1, 1)))(c3)
    x = (Conv2D(512, (3, 3), padding='valid'))(x)
    c4 = (BatchNormalization())(x)

    x = (UpSampling2D((2, 2)))(c4)
    x = (concatenate([x, c2], axis=-1))
    x = Dropout(dropout)(x)
    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(x)
    e4 = (BatchNormalization())(x)
    
    x = (UpSampling2D((2, 2)))(e4)
    x = (concatenate([x, c1], axis=-1))
    x = Dropout(dropout)(x)
    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(x)
    e3 = (BatchNormalization())(x)
    
    x = (UpSampling2D((2, 2)))(e3)
    x = (concatenate([x, c0], axis=-1))
    x = Dropout(dropout)(x)
    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(x)
    x = (BatchNormalization())(x)

    x = (UpSampling2D((2, 2)))(x)
    x = Conv2D(classes, (3, 3), padding='same')(x)
    
    x = (Activation('softmax'))(x)
    
    model = Model(input_img, x)
        
    return model

model = my_unet(len(colors))
model.compile(loss="categorical_crossentropy",optimizer="adadelta", metrics=['accuracy'])

def makecolormask(mask):
    ret_mask = np.zeros((mask.shape[0], mask.shape[1], len(colors)), 'uint8')
    
    for col in range(len(colors)):
        ret_mask[:, :, col] = (mask == col).astype(int)
                       
    return ret_mask

modeljsonname="model-chkpt.json"
modelweightname="model-chkpt.h5"

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(os.path.join(path, dirmodels,modelweightname), verbose=1, save_best_only=True, save_weights_only=True)
]

def generatebatchdata(batchsize, fullpathimages, fullpathmasks):
  
    imagenames = os.listdir(fullpathimages)
    imagenames.sort()

    masknames = os.listdir(fullpathmasks)
    masknames.sort()

    assert(len(imagenames) == len(masknames))
    
    for i in range(len(imagenames)):
        assert(imagenames[i] == masknames[i])

    while True:
        batchstart = 0
        batchend = batchsize    
        
        while batchstart < len(imagenames):
            
            imagelist = []
            masklist = []
            
            limit = min(batchend, len(imagenames))

            for i in range(batchstart, limit):
                if imagenames[i].endswith(".png"):
                    imagelist.append(cv2.imread(os.path.join(fullpathimages,imagenames[i]),cv2.IMREAD_COLOR ))
                if masknames[i].endswith(".png"):
                    masklist.append(makecolormask(cv2.imread(os.path.join(fullpathmasks,masknames[i]),cv2.IMREAD_UNCHANGED )))


            train_data = np.array(imagelist, dtype=np.float32)
            train_mask= np.array(masklist, dtype=np.float32)

            train_data /= 255.0
    
            yield (train_data,train_mask)    

            batchstart += batchsize   
            batchend += batchsize

generator_train = generatebatchdata(2, fullpathimages, fullpathmasks)
generator_valid = generatebatchdata(2, fullpathimagesvalid, fullpathmasksvalid)

model.load_weights(os.path.join(path, dirmodels,"model-chkpt.h5"))

#model.fit_generator(generator_train,steps_per_epoch=2500, epochs=10, callbacks=callbacks, validation_data=generator_valid, validation_steps=250)

modeljsonname="roadunet256-my-dropout-huge.json"
modelweightname="roadunet256-my-dropout-huge.h5"

model_json = model.to_json()
with open(os.path.join(path, dirmodels,modeljsonname), "w") as json_file:
    json_file.write(model_json)

model.save_weights(os.path.join(path, dirmodels,modelweightname))
"""
#model.load_weights(os.path.join(path, dirmodels,modelweightname))

imagetestlist = []

imagetestnames = os.listdir(fullpathimagestest)
imagetestnames.sort()

for imagename in imagetestnames:
    if imagename.endswith(".png"):
        imagetestlist.append(cv2.imread(os.path.join(fullpathimagestest,imagename),cv2.IMREAD_COLOR ))
        
test_data = np.array(imagetestlist, dtype=np.float32)
test_data /= 255.0

predictions_test = model.predict(test_data, batch_size=1, verbose=1)

def predictedmask(masklist):
    y_list = []
    for mask in masklist:
        assert mask.shape == (dim[0], dim[1], len(colors))
        imgret = np.zeros((dim[0], dim[1]), np.uint8)

        ##imgret = np.where(mask[:,:,:] == np.amax(mask[:,:,:]))
        #assert result < len(colors)
        #imgret = result[0][0]
        
        imgret = mask.argmax(axis=2)
        
        #print(imgret)
                
        y_list.append(imgret)
                    
    return y_list

def makemask(mask):
    ret_mask = np.zeros((mask.shape[0], mask.shape[1], 3), 'uint8')

    for col in range(len(colors)):
        layer = mask[:, :] == col
        ret_mask[:, :, 0] += ((layer)*(colors[col][0])).astype('uint8')
        ret_mask[:, :, 1] += ((layer)*(colors[col][1])).astype('uint8')
        ret_mask[:, :, 2] += ((layer)*(colors[col][2])).astype('uint8')
    
    return ret_mask

mymasks = predictedmask(predictions_test)
"""