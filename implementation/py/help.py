
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
from shutil import copyfile
from shutil import move

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
"""
for i in os.listdir(fullpathimagesvalid):
    imgpath = os.path.join(fullpathimagesvalid, i)
    bild = cv2.imread(imgpath,cv2.IMREAD_COLOR)
    bild = cv2.resize(bild, (256,256))
    cv2.imwrite(imgpath,bild)

for i in os.listdir(fullpathmasksvalid):
    maskpath = os.path.join(fullpathmasksvalid, i)
    bild = cv2.imread(maskpath,cv2.IMREAD_UNCHANGED)
    bild = cv2.resize(bild, (256,256))
    cv2.imwrite(maskpath,bild)
"""

imglist = os.listdir(fullpathimagesaugmented)
np.random.shuffle(imglist)

for i in range(500):
    imgpath_src = os.path.join(fullpathimagesaugmented, imglist[i])
    imgpath_dst = os.path.join(fullpathimagesvalid, imglist[i])
    #copyfile(imgpath_src,imgpath_dst)
    move(imgpath_src,imgpath_dst)
    filenamesplit = imgpath_src.split(".")
    filenamesplit = filenamesplit[0].split("\\")
    truefilename = filenamesplit[-1]
    #print(truefilename)
    maskfilename = truefilename + ".png"
    maskpath_src = os.path.join(fullpathmasksaugmented, maskfilename)
    maskpath_dst = os.path.join(fullpathmasksvalid, maskfilename)
    #copyfile(maskpath_src,maskpath_dst)
    move(maskpath_src,maskpath_dst)
    #print(jsonpath)
    