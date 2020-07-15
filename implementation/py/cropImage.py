
import cv2
import imutils
import numpy as np
import os 
import random 


#======================================================
#.............Data Augmentation........................
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


# todo : 

class myclass: 
    def __init__(self, imageName, maskName, imagesDir, masksDir, images_augemented, masks_augemented) : 
        #bild lesen 
        self.imageName = imageName 
        self.maskName = maskName 
        self.imagesDir = imagesDir 
        self.masksDir = masksDir 
        self.images_augemented = images_augemented
        self.masks_augemented = masks_augemented
        self.image = cv2.imread(os.path.join(imagesDir, imageName))
        self.mask = cv2.imread(os.path.join(masksDir, maskName), cv2.IMREAD_GRAYSCALE )

        # bild spiegelung 
        
        # original bild 
        #image_v = cv2.flip(self.image , 0)
        #self.image = cv2.vconcat([image_v, self.image, image_v])
        #image_h = cv2.flip(self.image, 1)
        #self.image = cv2.hconcat([image_h, self.image, image_h])
        
        # variablen : 
        # feste groesse : (256, 256)
        c_h = self.image.shape[0]//2
        c_v = self.image.shape[1]//2
        self.p1 = np.array([c_h - 230, c_v - 230 ]) ; self.p2 = np.array([c_h + 230, c_v + 230 ])
        self.index= 0 
        self.dim = (256, 256)
        self.setbrightness=False
        self.alpha = 1.0
        self.beta = 0
        self.fliped = 0
        self.flipedIndex = 0
        self.imageIndex = 0
        self.imagecopy = self.image.copy()
        self.maskcopy = self.mask.copy()
    
    def save(self):
        output = imutils.rotate(self.image, self.index)
        output2 = imutils.rotate(self.mask, self.index)
        if self.setbrightness:
            output =  cv2.convertScaleAbs(output, alpha=self.alpha, beta=self.beta)
        if self.fliped != self.flipedIndex : 
            self.fliped = self.flipedIndex
            print(self.flipedIndex, self.fliped)
            output = cv2.flip(output, 1)
            output2 = cv2.flip(output,1)

        center = (self.p1 + self.p2) // 2 
        a = center + [5, 0] ; b = center - [5, 0] 
        c = center + [0, 5] ; d = center - [0, 5]
        output = cv2.rectangle(output,tuple(self.p1) , tuple(self.p2)  , (0,0,255), 2)
        output2 = cv2.rectangle(output2, tuple(self.p1) , tuple(self.p2)  , (0,0,255), 2)
        output = cv2.line(output,tuple(a) , tuple(b) ,(0,0,255),2)
        output = cv2.line(output, tuple(c) , tuple(d) , (0,0,255),2 )
        output2 = cv2.line(output2,tuple(a) , tuple(b) ,(0,0,255),2)
        output2 = cv2.line(output2, tuple(c) , tuple(d) , (0,0,255),2 )
        output_output2 = np.hstack((output, output))
        cv2.namedWindow('myW',cv2.WINDOW_NORMAL)
        cv2.imshow("myW" , output_output2)
        


    # def increase(self): 
    #     self.p1 -= 10 ;  self.p2 += 10 
    #     self.save() 

    # def decrease(self): 
    #     self.p1 += 10 ; self.p2 -= 10 
    #     self.save() 

    def up(self, param=10) : 
        self.p1[1] -= param ; self.p2[1] -= param
        #self.save() 

    def down(self, param=10): 
        self.p1[1] += param ; self.p2[1] += param
        #self.save() 

    def left(self, param=10): 
        self.p1[0] -= param ; self.p2[0] -= param
        #self.save()

    def right(self, param=10): 
        self.p1[0] += param ; self.p2[0] += param
        #self.save()

    def rotate_l(self, param=1):
        self.index += param
        #self.save()

    def rotate_r(self, param=1):
        self.index -= param
        #self.save()

    # todo : random value : 0.8, 1.5
    #        random value : 5 , 105 
    def brightness(self):
        self.setbrightness = True
        self.alpha = round(random.uniform(0.8, 1.5), 1)
        self.beta = random.randint(5, 105)
        #self.save()


    def flipImage(self):
        self.flipedIndex +=1
        #self.save()
        

    def crop(self):
        image = imutils.rotate_bound(self.imagecopy, self.index)
        assert (len(self.maskcopy.shape) < 3 )
        mask = imutils.rotate_bound(self.maskcopy, self.index)
        assert (len(self.mask.shape) < 3 )
        
        if self.setbrightness:
            image =  cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        x1 = self.p1[0]; y1 = self.p1[1]
        x2 = self.p2[0]  ; y2 = self.p2[1]
        cropedimage = image[x1:x2 , y1:y2]
        cropedmask = mask[x1:x2 , y1:y2]
        cropedImageFullName = os.path.join(self.images_augemented, "croped_" + str(self.imageIndex) + self.imageName)
        cropedMaskFullName = os.path.join(self.masks_augemented, "croped_" + str(self.imageIndex) + self.maskName)
        self.imageIndex += 1 
        cropedimage = cv2.resize(cropedimage, self.dim, cv2.INTER_NEAREST)
        cv2.imwrite(cropedImageFullName,cropedimage)
        print(cropedImageFullName + " was saved")
        cropedmask = cv2.resize(cropedmask, self.dim, cv2.INTER_NEAREST) 
        cv2.imwrite(cropedMaskFullName , cropedmask)
        print(cropedMaskFullName + " was saved")
        return cropedimage, cropedmask
    
  

    def automate(self):
        #  center : 
        self.crop()
        # up 
        #self.up(10)
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.crop()
        self.left()


        # flip 
        self.crop()
        # up 
        #self.up(10)
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.crop()
        self.left()

         # rotatation left  : 10 
        self.rotate_l(3)
        self.crop()
        # up 
        #self.up(10)
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.crop()
        self.left()
        self.rotate_r(3)

        # rotatation right  : 10 
        self.rotate_r(3)
        self.crop()
        # up 
        #self.up(10)
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.crop()
        self.left()
        self.rotate_l(3)




        # repeat with brithniss
        #  center : 
        self.brightness()
        self.crop()
        # up 
        #self.up(10)
        #self.brightness()
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.brightness()
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.brightness()
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.brightness()
        self.crop()
        self.left()

        # rotatation left  : 10 
        self.rotate_l(10)
        self.crop()
        # up 
        #self.up(10)
        #self.brightness()
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.brightness()
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.brightness()
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.brightness()
        self.crop()
        self.left()
        self.rotate_r(10)

        # rotatation right  : 10 
        self.rotate_r(10)
        self.crop()
        # up 
        #self.up(10)
        #self.brightness()
        #self.crop()
        #self.down(10)
        #down
        self.down(10)
        self.brightness()
        self.crop()
        self.up(10)
        # left 
        self.left(10)
        self.brightness()
        self.crop()
        self.right(10)
        # right 
        self.right(10)
        self.brightness()
        self.crop()
        self.left()
        self.rotate_l(10)




    def ausgabe(self) : 
        self.save()
        while True : 
            k = cv2.waitKey(0)
            if k == ord('l'):
                self.rotate_l()
            elif k== ord('r'):
                self.rotate_r()
            elif k == ord('d'):
                self.right()
            elif k== ord('a') : 
                self.left()
            elif k== ord('w') : 
                self.up()
            elif k== ord('f'):
                self.fliped = not self.fliped
            elif k== ord('x') : 
                self.down()
            elif k==ord('b') : 
                self.brightness()
            elif k==ord("s"):
                self.crop()
            elif k== ord('o'):
                self.automate()
            elif k== ord("q"):
                cv2.destroyAllWindows()

                break
            else : 
                pass

#=====================================================
# main Programm
def main():
    print("program has started ...")
    imagesDir = "D:\dev\dcps\AutonomesFahren\data\images"
    masksDir = "D:\dev\dcps\AutonomesFahren\data\masks"
    images_augemented = "D:\dev\dcps\AutonomesFahren\data\images_augmented"
    masks_augemented  = "D:\dev\dcps\AutonomesFahren\data\masks_augmented"
    images = os.listdir(imagesDir)
    masks = os.listdir(masksDir)

    assert len(images) == len(masks)

    for image, mask in zip(images, masks): 
        print((image, mask))
        p = myclass(image, mask, imagesDir, masksDir, images_augemented, masks_augemented)
        p.automate()




if __name__ == "__main__": 
    main()


