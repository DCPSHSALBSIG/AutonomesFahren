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
#model.load_weights(os.path.join(path, dirmodels,modelweightname))

imagetestlist = []

imagetestnames = os.listdir(fullpathimagestest)
imagetestnames.sort()

for imagename in imagetestnames:
    if imagename.endswith(".png"):
        imagetestlist.append(cv2.imread(os.path.join(fullpathimagestest,imagename),cv2.IMREAD_COLOR ))
        
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

modelweightname = "roadunet256-my-dropout-huge.h5"
model.load_weights(os.path.join(path, dirmodels,modelweightname))

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
        for i in range(dim[0]):
            for j in range(dim[1]):
                result = np.where(mask[i,j,:] == np.amax(mask[i,j,:]))
                assert result[0][0] < len(colors)
                imgret[i,j] = result[0][0]
        y_list.append(imgret)
        #for k in range(len(colors)):
        #    sum1 = 0;
        #    for i in range(dim[0]):
        #        for j in range(dim[1]):
        #                sum1  += mask[i,j,k]
        #    print("{}: {}".format(k, sum1))
                    
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
ind = 1

img = np.zeros((dim[0], dim[1], 3), 'uint8')

img = makemask(mymasks[ind])
#img = predictedmask(masktestlist)

cv2.imshow("maske",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

weighted = np.zeros((dim[0], dim[1], 3), 'uint8')
cv2.addWeighted(imagetestlist[ind], 0.5, img, 0.5, 0, weighted)

cv2.imshow("weighted",weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
for ind in range(5):
    img = np.zeros((dim[0], dim[1], 3), 'uint8')
    img = makemask(mymasks[ind])
    weighted = np.zeros((dim[0], dim[1], 3), 'uint8')
    cv2.addWeighted(imagetestlist[ind], 0.5, img, 0.5, 0, weighted)
    filename = "prediction" + str(ind) + ".png"
    cv2.imwrite(os.path.join(path, dirpredictions,filename), weighted)