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


classes = {"Strasse" : 1, 
           "strasse" : 1,
          "Streifen" : 2,
           "streifen" : 2,
          "Auto" : 3, 
          "auto" : 3,
          "Schild" : 4,
           "schild" : 4,
         }

colors = {0 : (0,0,0), 
          1 : (0,0,255), 
          2 : (0,255,0),
          3 : (255,0,0), 
          4 : (0,255,255),         
         }

dim = (512, 512) 

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

print(os.path.exists(fullpathjson))
print(os.path.exists(fullpathimages))
print(os.path.exists(fullpathmasks))
print(os.path.exists(fullpathmasksvalid))


def makesquare2(img):
    
    assert(img.ndim == 2) 
    
    edge = min(img.shape[0],img.shape[1])
        
    img_sq = np.zeros((edge, edge), 'uint8')
    
    if(edge == img.shape[0]):
        img_sq[:,:] = img[:,int((img.shape[1] - edge)/2):int((img.shape[1] - edge)/2)+edge]
    else:
        img_sq[:,:,:] = img[int((img.shape[0] - edge)/2):int((img.shape[0] - edge)/2)+edge,:]

    assert(img_sq.shape[0] == edge and img_sq.shape[1] == edge)
    
    return img_sq

def makesquare3(img):
    
    assert(img.ndim == 3)
    
    edge = min(img.shape[0],img.shape[1])
        
    img_sq = np.zeros((edge, edge, 3), 'uint8')
    
    if(edge == img.shape[0]):
        img_sq[:,:,:] = img[:,int((img.shape[1] - edge)/2):int((img.shape[1] - edge)/2)+edge,:]
    else:
        img_sq[:,:,:] = img[int((img.shape[0] - edge)/2):int((img.shape[0] - edge)/2)+edge,:,:]

    assert(img_sq.shape[0] == edge and img_sq.shape[1] == edge)
    
    return img_sq

def testMasks(sourcejsonsdir, destimagesdir, destmasksdir):
    count = 0
    directory = sourcejsonsdir
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            print("{}:{}".format(count,os.path.join(directory, filename)))
            f = open(os.path.join(directory, filename))
            data = json.load(f)
            #print(data['imagePath'])
            img_arr = data['imageData']  
            imgdata = base64.b64decode(img_arr)

            img = cv2.imdecode(np.frombuffer(imgdata, dtype=np.uint8), flags=cv2.IMREAD_COLOR)

            #print(img.shape[0])
            #print(img.shape[1])
            
            assert (img.shape[0] > dim[0])
            assert (img.shape[1] > dim[1])
           
            img = makesquare3(img)

            #assert(img.shape[0] == img.shape[1])
          
            for shape in data['shapes']:
                if not (shape['label'] in classes):
                    f.close()
                    assert(0 > 1)
                
            count += 1
            
            f.close()


testMasks(fullpathjson, fullpathimages, fullpathmasks)
testMasks(fullpathjsonsvalid, fullpathimagesvalid, fullpathmasksvalid)
testMasks(fullpathjsonstest, fullpathimagestest, fullpathmaskstest)


def createMasks(sourcejsonsdir, destimagesdir, destmasksdir):

    assocf = open(os.path.join(path,"assoc_orig.txt"), "w")
    #assocfile = open("assoc_orig.txt", "w")
    #print(os.path.join(path,"assoc_orig.txt"))
    #print(os.path.exists(os.path.join(path,"assoc_orig.txt")))
    #print(f.writable())
    #f.write("")
    count = 0
    directory = sourcejsonsdir
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            print("{}:{}".format(count,os.path.join(directory, filename)))
            
            f = open(os.path.join(directory, filename))
            data = json.load(f)
            #print(data['imagePath'])
            img_arr = data['imageData']  
            imgdata = base64.b64decode(img_arr)

            img = cv2.imdecode(np.frombuffer(imgdata, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
            
            assert (img.shape[0] > dim[0])
            assert (img.shape[1] > dim[1])
           
            finalmask = np.zeros((img.shape[0], img.shape[1]), 'uint8')
            #finalmask_strasse = np.zeros((img.shape[0], img.shape[1]), 'uint8')
        
            masks=[]
            masks_strassen=[]
            mask_strasse = np.zeros((img.shape[0], img.shape[1]), 'uint8')
            
            
            for shape in data['shapes']:
                assert(shape['label'] in classes)

                vertices = np.array([[point[1],point[0]] for point in shape['points']])
                vertices = vertices.astype(int)

                rr, cc = polygon(vertices[:,0], vertices[:,1], img.shape)
                mask_orig = np.zeros((img.shape[0], img.shape[1]), 'uint8')

                mask_orig[rr,cc] = classes[shape['label']]
                if classes[shape['label']] != classes['Strasse']:
                    masks.append(mask_orig)
                else:
                    masks_strassen.append(mask_orig)
                    
            for m in masks_strassen:
                mask_strasse += m
                    
            for m in masks:
                _,mthresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY_INV)
                finalmask = cv2.bitwise_and(finalmask,finalmask,mask = mthresh)
                finalmask += m

            _,mthresh = cv2.threshold(finalmask,0,255,cv2.THRESH_BINARY_INV)
            mask_strasse = cv2.bitwise_and(mask_strasse,mask_strasse, mask = mthresh)
            finalmask += mask_strasse  
                
            img = makesquare3(img)
            finalmask = makesquare2(finalmask)

            img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
            finalmask_resized = cv2.resize(finalmask, dim, interpolation = cv2.INTER_NEAREST)
            
            filepure,extension = splitext(filename)
            
            cv2.imwrite(os.path.join(destimagesdir, "{}o.png".format(filepure)), img_resized)
            cv2.imwrite(os.path.join(destmasksdir, "{}o.png".format(filepure)), finalmask_resized)

            assocf.write("{:05d}o:{}\n".format(count,filename))
            assocf.flush()
            count += 1

        else:
            continue
    f.close()


def createMasksAugmented(ident, sourcejsonsdir, destimagesdir, destmasksdir):

    assocf = open(os.path.join(path,"assoc_{}_augmented.txt".format(ident)), "w")

    count = 0
    directory = sourcejsonsdir
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            print("{}:{}".format(count,os.path.join(directory, filename)))
            #print(os.path.join(directory, filename))

            f = open(os.path.join(directory, filename))
            data = json.load(f)
            #print(data['imagePath'])
            img_arr = data['imageData']  
            imgdata = base64.b64decode(img_arr)

            img = cv2.imdecode(np.frombuffer(imgdata, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
            
            assert (img.shape[0] > dim[0])
            assert (img.shape[1] > dim[1])
            
            zoom = randint(75,90)/100.0
            angle = (2*random()-1)*3.0
            img_rotated = imutils.rotate_bound(img, angle)
            
            xf = int(img_rotated.shape[0]*zoom)
            yf = int(img_rotated.shape[1]*zoom)            
            
            img_zoomed = np.zeros((xf, yf, img_rotated.shape[2]), 'uint8')
            img_zoomed[:,:,:] = img_rotated[int((img_rotated.shape[0]-xf)/2):int((img_rotated.shape[0]-xf)/2)+xf,int((img_rotated.shape[1]-yf)/2):int((img_rotated.shape[1]-yf)/2)+yf,:] 
        

            finalmask = np.zeros((img_zoomed.shape[0], img_zoomed.shape[1]), 'uint8')
            mthresh = np.zeros((img_zoomed.shape[0], img_zoomed.shape[1]), 'uint8')
            masks=[]
            masks_strassen=[]
            mask_strasse = np.zeros((img_zoomed.shape[0], img_zoomed.shape[1]), 'uint8')

            for shape in data['shapes']:
                assert(shape['label'] in classes)

                vertices = np.array([[point[1],point[0]] for point in shape['points']])
                vertices = vertices.astype(int)

                rr, cc = polygon(vertices[:,0], vertices[:,1], img.shape)
                mask_orig = np.zeros((img.shape[0], img.shape[1]), 'uint8')
                mask_orig[rr,cc] = classes[shape['label']]
                
                mask_rotated = imutils.rotate_bound(mask_orig, angle)
                
                mask_zoomed = np.zeros((xf, yf), 'uint8')
                mask_zoomed[:,:] = mask_rotated[int((img_rotated.shape[0]-xf)/2):int((img_rotated.shape[0]-xf)/2)+xf,int((img_rotated.shape[1]-yf)/2):int((img_rotated.shape[1]-yf)/2)+yf] 
                
                if classes[shape['label']] != classes['Strasse']:
                    masks.append(mask_zoomed)
                else:
                    masks_strassen.append(mask_zoomed)

            for m in masks_strassen:
                mask_strasse += m
                    
            for m in masks:
                _,mthresh = cv2.threshold(m,1,255,cv2.THRESH_BINARY_INV)
                finalmask = cv2.bitwise_and(finalmask,finalmask,mask = mthresh)
                finalmask += m

            _,mthresh = cv2.threshold(finalmask,1,255,cv2.THRESH_BINARY_INV)
            mask_strasse = cv2.bitwise_and(mask_strasse,mask_strasse, mask = mthresh)
            finalmask += mask_strasse    
    
            # contrast-> alpha: 1.0 - 3.0; brightness -> beta: 0 - 100
            alpha = 0.8 + 0.4*random();
            beta = int(random()*15)
    
            img_adjusted = cv2.convertScaleAbs(img_zoomed, alpha=alpha, beta=beta)
        
        
            img_adjusted = makesquare3(img_adjusted)
            finalmask = makesquare2(finalmask)

            img_resized = cv2.resize(img_adjusted, dim, interpolation = cv2.INTER_NEAREST)
            finalmask_resized = cv2.resize(finalmask, dim, interpolation = cv2.INTER_NEAREST)
            
            filepure,extension = splitext(filename)
            
            cv2.imwrite(os.path.join(destimagesdir, "{}{}.png".format(filepure, ident)), img_resized)
            cv2.imwrite(os.path.join(destmasksdir, "{}{}.png".format(filepure, ident)), finalmask_resized) 

            assocf.write("{:05d}:{}\n".format(count, filename))
            assocf.flush()
            count += 1

        else:
            continue
    f.close()

createMasks(fullpathjson, fullpathimages, fullpathmasks)
#createMasksAugmented(fullpathjson, fullpathimages, fullpathmasks)
#createMasksAugmented("a2",fullpathjson, fullpathimages, fullpathmasks)
#createMasksAugmented("a3",fullpathjson, fullpathimages, fullpathmasks)

createMasks(fullpathjsonsvalid, fullpathimagesvalid, fullpathmasksvalid)
#createMasksAugmented("a4", fullpathjsonsvalid, fullpathimagesvalid, fullpathmasksvalid)

def makemask(mask):
    assert(dim[0] == mask.shape[0])
    assert(dim[1] == mask.shape[1])
    ret_mask = np.zeros((dim[0], dim[1], 3), 'uint8')
    for i in range(dim[0]):
        for j in range(dim[1]):
                assert(mask[i,j] < len(colors))
                ret_mask[i,j,0] = colors[mask[i,j]][0]
                ret_mask[i,j,1] = colors[mask[i,j]][1]
                ret_mask[i,j,2] = colors[mask[i,j]][2]
    return ret_mask

createMasks(fullpathjsonstest, fullpathimagestest, fullpathmaskstest)