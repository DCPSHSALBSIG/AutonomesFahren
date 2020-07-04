import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as c 


image = c.imread("../data/images/scene00011.png")

shape = image.shape
testImage = np.zeros(shape)
output = np.hstack((image, testImage))

img = c.cvtColor(image, c.COLOR_BGR2HSV)
img[:2] -= 128
img =  c.cvtColor(img, c.COLOR_HSV2BGR)
print(img)

l = np.array([[1,2],[1,2],[1,2]])
l[:2] = l[:2] + 10
print(l)

while  True : 
    k = c.waitKey(0)
    c.namedWindow("my", c.WINDOW_NORMAL)
    c.imshow("my", img)
    if k== ord("q"):
        c.destroyAllWindows()
        break