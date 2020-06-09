
import cv2
import imutils
import numpy as np
import os 
class myclass: 
    def __init__(self, path, imagename, maskename) : 
        #bild lesen 
        self.path = path
        self.imagename =  imagename
        self.maskename = maskename  
        self.image = cv2.imread(self.path + self.imagename)
        self.maske = cv2.imread(self.path + self.maskename)
        # bild spiegelung 
        
        # original bild 
        #image_v = cv2.flip(self.image , 0)
        #self.image = cv2.vconcat([image_v, self.image, image_v])
        #image_h = cv2.flip(self.image, 1)
        #self.image = cv2.hconcat([image_h, self.image, image_h])
        
        # variablen : 
        # feste groesse : (256, 256)
        self.p1 = np.array([10, 10 ]) ; self.p2 = np.array([410, 410 ])
        self.index= 0 
        self.setbrightness=False
        self.alpha = 1.0 
        self.beta = 0
        self.fliped = False 
        self.copy = self.image.copy()
        self.maskecopy = self.maske.copy()
    
    def save(self):
        output = imutils.rotate(self.image, self.index)
        output2 = imutils.rotate(self.maske, self.index)
        if self.setbrightness:
            output =  cv2.convertScaleAbs(output, alpha=self.alpha, beta=self.beta)
        if self.fliped : 
            output = cv2.flip(output, 2)
            output2 = cv2.flip(output, 2)
        elif not self.fliped : 
            output = cv2.flip(output, 2)
            output2 = cv2.flip(output, 2)
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

    def up(self) : 
        self.p1[1] -= 10 ; self.p2[1] -= 10
        self.save() 

    def down(self): 
        self.p1[1] += 10 ; self.p2[1] += 10
        self.save() 

    def left(self): 
        self.p1[0] -= 10 ; self.p2[0] -= 10
        self.save()

    def right(self): 
        self.p1[0] += 10 ; self.p2[0] += 10
        self.save()

    def rotate_l(self):
        self.index += 1
        self.save()

    def rotate_r(self):
        self.index -= 1 
        self.save()
        
    def schneiden(self):
        image = imutils.rotate(self.copy, self.index)
        maske = imutils.rotate(self.maskecopy, self.index)
        if self.setbrightness:
            image =  cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        x1 = self.p1[0]; y1 = self.p1[1]
        x2 = self.p2[0]  ; y2 = self.p2[1]
        geschnitten = image[x1:x2 , y1:y2]
        maske_geschnitten = maske[x1:x2 , y1:y2]
        name =self.path +  "croped_" + self.imagename 
        maske_name =self.path +  "croped_" + self.maskename 
        cv2.imwrite(name ,geschnitten)
        cv2.imwrite(maske_name, maske_geschnitten)
        currentdir = os.getcwd()
        print("saved", self.path, currentdir)
        return geschnitten, maske_geschnitten
    
    # todo : random value : 0.8, 1.5
    #        random value : 5 , 105 
    def morebrightness(self):
        self.setbrightness = True
        if self.alpha <= 2.8:
            self.alpha += 0.2
        if self.beta <= 95 : 
            self.beta += 5
        self.save()

    def lessbrightness(self):
        self.setbrightness = True
        if self.alpha >0.2:
            self.alpha -= 0.2
        if self.beta >= 5 : 
            self.beta -= 5
        self.save()

    def flipImage(self):
        self.fliped = not self.fliped 

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
            elif k==ord('+') : 
                self.morebrightness()
            elif k==ord('-') : 
                self.lessbrightness()
            elif k==ord("s"):
                self.schneiden()
            elif k== ord("q"):
                cv2.destroyAllWindows()

                break
            else : 
                pass

image = "image"
maske = "maske"
path = "../data/images/"
for i in range(10):
    image_name = path + image + str(i) + ".png"
    maske_name = path + maske + str(i) + ".png"
# p = myclass(image_name, maske_name)
# p.ausgabe
p = myclass("../data/images/" , "scene00001.png", "scene00011.png")
p.ausgabe()


# problem bilder automatisch lesen 
# loesung : (einheitliche namen)
# scene + i + png 
# scene + i + masek + png

