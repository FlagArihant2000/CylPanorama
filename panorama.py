import cv2
import numpy as np
#import os
import math
def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img
#---------------------------------------------------------------------------------------------------------------------------------------------
img1 = cv2.imread('1.jpeg')
img3 = img1.copy()
canvas = np.zeros((img1.shape[0]+200,img1.shape[1]*5,img1.shape[2]),np.uint8)

canvas[0:img1.shape[0],0:img1.shape[1]] = img1
h = np.eye(3)
for i in range(2,5):
        img1 = img3
        img2 = cv2.imread(str(i)+'.jpeg')
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret,thresh= cv2.threshold(gray1,0,255,cv2.THRESH_BINARY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1,desc1 = sift.detectAndCompute(gray1,mask = thresh)
        kp2,desc2 = sift.detectAndCompute(gray2,None)
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good = []
        pt1 = []
        pt2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        good = sorted(good,key = lambda x:x.distance)
        good = good[:15]
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        H =  np.linalg.inv(H)
        h= H
        img3 = cv2.warpPerspective(img2,H,(canvas.shape[1],canvas.shape[0]))
        print(canvas.shape,img3.shape)
        canvas = image_stiching(canvas,img3)
       
        if cv2.waitKey(1) == 2:
            break
cv2.imshow('panorama',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
