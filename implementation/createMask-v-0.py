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


#name = r"Testbild5 gelabelt_NeleSchäfer.json"
#path = r"C:\Users\User\Documents\DeepLearning\AutonomerTransport\Testbilder gelabelt_NeleSchaefer"

name = r"kang10.json"
path = r"./jsons"

fullpath = os.path.join(path,name)
f = open(fullpath)
data = json.load(f)
img_arr = data['imageData']


imgdata = base64.b64decode(img_arr)

colors = {"schild"   :    (0, 0, 255),
          "auto"     :    (127, 127, 127),   
          "streifen" :    (255, 0, 0),
          "strasse"  :    (0, 255, 0),          
         }

masks={}

im_arr = np.frombuffer(imgdata, dtype=np.uint8)
img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
mask = np.zeros((img.shape[0], img.shape[1]), 'uint8')
#print(item)



for shape in data['shapes']:
    mask = np.zeros((img.shape[0], img.shape[1]), 'uint8')
    print(shape['label'])
    masks[shape['label']] = mask





i = 0
for item in colors:
    print(item)
    for shape in data['shapes']:
        #print(shape['label'])
        if shape['label'] == item:
            print(shape['label'])
            
            vertices = np.array([[point[1],point[0]] for point in shape['points']])
            vertices = vertices.astype(int)
            
            rr, cc = polygon(vertices[:,0], vertices[:,1], masks[shape['label']].shape)
            masks[shape['label']][rr,cc] = 1
            i += 1
    




for l, m in masks.items():
    print("{} and {}".format(l, m.shape))



plt.imshow(img)
print(img.shape)



dim = (128, 128) 

def makecolor(mask, color):
    ret_mask = np.zeros((mask.shape[0], mask.shape[1], 3), 'uint8')
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] >= 1: 
                ret_mask[i,j,0] = color[0]
                ret_mask[i,j,1] = color[1]
                ret_mask[i,j,2] = color[2]
    return ret_mask


ret_mask = np.zeros((dim[0], dim[1], 3), 'uint8')

for l, m in masks.items():
    print(l)
    #ret_mask = makecolor(masks['1-Nadel2'], colors['1-Nadel2']) + makecolor(masks['Tape1'], colors['Tape1'])
    #ret_mask += makecolor(masks[l], colors[l])
    ret_mask += makecolor(cv2.resize(masks[l], dim, interpolation = cv2.INTER_AREA), colors[l])


plt.imshow(ret_mask)
#ret_mask += ret_mask


weighted = np.zeros((dim[0], dim[1], 3), 'uint8')
img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.addWeighted(img_resized, 0.5, ret_mask, 0.5, 0, weighted)
plt.imshow(weighted)


imgname,extension = splitext(fullpath)

cleanimgname = re.sub(r"[üäöß]","", imgname)


cv2.imwrite(cleanimgname+"_mask.png",ret_mask)
cv2.imwrite(cleanimgname+"_weighted.png",weighted)