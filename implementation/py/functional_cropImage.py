
#======================================================
#......Data Augmentation (functional version)..........
# instructions : ......................................
# a : move rect left...................................
# d : move rect right..................................
# w : move rect up.....................................
# x : move rect down...................................
# s : crop image and save..............................
# b : random brightness................................
# f : flip image.......................................
# r : rotate image.....................................
# o : automate the previous operations.................
# q : quite............................................ 
#======================================================

# TODO : function docs 


#=====================================================
# import packages
import cv2
import imutils
import numpy as np
import os 
import random 


#=====================================================
# translation  
def up(p1, p2,  param=10) : 
    p1[1] -= param ; p2[1] -= param
    return p1, p2 

def down(p1, p2,  param=10): 
    p1[1] += param ; p2[1] += param
    return p1, p2 

def left(p1, p2,  param=10): 
    p1[0] -= param ; p2[0] -= param
    return p1, p2

def right(p1, p2,  param=10): 
    p1[0] += param ; p2[0] += param 
    return p1, p2 

#=====================================================
# rotation 
def rotate_l(index): 
    index += 1
    return index 

def rotate_r(index ): 
    index -= 1 
    return index 
    
#=====================================================
# change brightness randomly
# todo : random value : 0.8, 1.5
#        random value : 5 , 105 
def brightness() : 
    alpha = round(random.uniform(0.8, 1.5), 1)
    beta = random.randint(5, 105)
    return alpha, beta


#=====================================================
# flip horizontally 
def flipImage(flipedIndex): 
    flipedIndex +=1
    return flipedIndex


#=====================================================
# crop and save
def crop(imagecopy, maskcopy, p1, p2, alpha, beta, setbrightness, 
    imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0):
    image = imutils.rotate(imagecopy, index)
    mask = imutils.rotate(maskcopy, index)
    print(image)
    if setbrightness:
        image =  cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    x1 = p1[0]; y1 = p1[1]
    x2 = p2[0]  ; y2 = p2[1]
    cropedImage = image[x1:x2 , y1:y2]
    cropedMask = mask[x1:x2 , y1:y2]
    imageFullName = os.path.join(imagesAugmentedPath, imageAugmentedName)
    maskFullName = os.path.join(masksAugmentedPath, maskAugmentedName)
    cv2.imwrite(imageFullName ,cropedImage)
    print(imageFullName + " was saved successfully")
    cv2.imwrite(maskFullName, cropedMask)
    print(maskFullName + " was saved successfully")
    return 


#=====================================================
# do all previous operations (augmentation)
def augment(difference , image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0): 
    #  center : 
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    # up 
    p1, p2 = up(p1, p2, difference)

    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)

    p1, p2 = down(p1, p2, difference)
    #down
    p1, p2 = down(p1, p2, difference)
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = up(p1, p2, difference)
    # left 
    p1, p2 = left(p1, p2, difference)
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = right(p1, p2, difference)
    # right 
    p1, p2 = right(p1, p2, difference)
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = left(p1, p2, difference)

    # repeat with brithniss
    #  center : 
    alpha, beta = brightness()
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    # up 
    p1, p2 = up(p1, p2, difference)
    alpha, beta = brightness()
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = down(p1, p2, difference)
    #down
    p1, p2 = down(p1, p2, difference)
    alpha, beta = brightness()
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = up(p1, p2, difference)
    # left 
    p1, p2 = left(p1, p2, difference)
    alpha, beta = brightness()
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = right(p1, p2, difference)
    # right 
    p1, p2 = right(p1, p2, difference)
    alpha, beta = brightness()
    crop(image, mask, p1, p2, alpha, beta, setbrightness, 
            imagesAugmentedPath, masksAugmentedPath, imageAugmentedName, maskAugmentedName, index=0)
    p1, p2 = left(p1, p2, difference)



#=====================================================
# main Programm
def main():
    print("program has started ...")
    imagesDir = "/home/moamen/myGitRepos/AutonomesFahren/data/images"
    masksDir = "/home/moamen/myGitRepos/AutonomesFahren/data/masks"
    images_augemented = "/home/moamen/myGitRepos/AutonomesFahren/data/images_augmented"
    masks_augemented  = "/home/moamen/myGitRepos/AutonomesFahren/data/masks_augmented"
    images = os.listdir(imagesDir)
    masks = os.listdir(masksDir)
    
    assert len(images) == len(masks)

    images_fullName = [os.path.join(imagesDir, i) for i in images ]
    masks_fullName = [os.path.join(masksDir, i) for i in masks ]

    #for image, mask in zip(images_fullName, masks_fullName): 
    image = images_fullName[0]
    mask = masks_fullName[0]
    print(image)
    print(mask)
    # variablen : 
    # feste groesse : (256, 256)
    
    p1 = np.array([10, 10 ]) 
    p2 = np.array([410, 410 ])
    #index= 0 
    setbrightness=False
    alpha = 1.0
    beta = 0
    difference = 50 
    #fliped = 0
    #flipedIndex = 0

    imageObj = cv2.imread(image)
    maskObj = cv2.imread(mask)
    
    augment(difference, imageObj, maskObj, p1, p2, alpha, beta, setbrightness, 
        images_augemented, masks_augemented, "test1.png", "test1.png", index=0)


    




if __name__ == "__main__": 
    main()
