from skimage.draw import polygon
import PIL
import json
import base64
import os.path
from os.path import splitext
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import re
from random import *

colors = {0 : (0,0,0), 
          1 : (0,0,255), 
          2 : (0,255,0),
          3 : (255,0,0), 
          4 : (0,255,255),         
         }


dim = (256, 256) 

path =  r'D:\dev\dcps\AutonomesFahren\data'
dirjson = "jsons"
dirimages = "images"
dirmasks = "masks"
dirimagesaugmented = "images_augmented"
dirmasksaugmented = "masks_augmented"

dirjsonsvalid = "jsons_valid"
dirimagesvalid = "images_valid"
dirmasksvalid = "masks_valid"

dirjsonstest = "jsons_test"
dirimagestest = "images_test"
dirmaskstest = "masks_test"

fullpathjson = os.path.join(path, dirjson)
fullpathimages = os.path.join(path, dirimages)
fullpathmasks = os.path.join(path, dirmasks)

fullpathimagesaugmented = os.path.join(path, dirimagesaugmented)
fullpathmasksaugmented = os.path.join(path, dirmasksaugmented)

fullpathjsonsvalid = os.path.join(path, dirjsonsvalid)
fullpathimagesvalid = os.path.join(path, dirimagesvalid)
fullpathmasksvalid = os.path.join(path, dirmasksvalid)

fullpathjsonstest = os.path.join(path, dirjsonstest)
fullpathimagestest = os.path.join(path, dirimagestest)
fullpathmaskstest = os.path.join(path, dirmaskstest)

def makemask(mask):
    ret_mask = np.zeros((mask.shape[0], mask.shape[1], 3), 'uint8')

    assert (mask.shape[0] == dim[0] and mask.shape[1] == dim[1])
    assert (len(mask.shape)<3)
    
    for col in range(len(colors)):
        layer = mask[:, :] == col
        ret_mask[:, :, 0] += ((layer)*(colors[col][0])).astype('uint8')
        ret_mask[:, :, 1] += ((layer)*(colors[col][1])).astype('uint8')
        ret_mask[:, :, 2] += ((layer)*(colors[col][2])).astype('uint8')
    
    return ret_mask


def checkimages(fullpathimages, fullpathmasks):
    imagenames = []
    masknames = []
    imagenames = os.listdir(fullpathimages)
    imagenames.sort()

    masknames = os.listdir(fullpathmasks)
    masknames.sort()

    assert (len(imagenames) == len(masknames))

    for j in range(len(imagenames)):
        assert(imagenames[j] == masknames[j])

    for j in range(len(imagenames)):
        print(imagenames[j])
        
        origname =os.path.join(fullpathimages,imagenames[j])
        maskname = os.path.join(fullpathmasks, masknames[j])
        weighted = np.zeros((dim[0], dim[1], 3), 'uint8')
        origimg = cv2.imread(origname,cv2.IMREAD_COLOR )
        maskimg =cv2.imread(maskname,cv2.IMREAD_UNCHANGED )
    
        

        if origimg.shape[0] != dim[0] or origimg.shape[1] != dim[1]:
            print("fail opening orig with {}".format(origname))
            continue
        if maskimg.shape[0] != dim[0] or maskimg.shape[1] != dim[1]:
            print("fail opening mask with {}".format(maskname))
            continue           

        cv2.addWeighted(origimg, 0.5, makemask(maskimg), 0.5, 0, weighted)
        cv2.imwrite(os.path.join(path,"comparison", imagenames[j]),weighted)
    

#checkimages(fullpathimagesaugmented, fullpathmasksaugmented)
checkimages(fullpathimagesvalid, fullpathmasksvalid)
#checkimages(fullpathimagestest, fullpathmaskstest)