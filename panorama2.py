import cv2
import numpy as np
import os
import math

def required_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    image,contour,heic = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contour[0]
    cnt = cnt.reshape(-1,2)
    x = np.max(cnt[:,0],axis = 0)
    y = np.max(cnt[:,1],axis = 0)
    img2 = np.zeros((y,x,3),np.uint8)
    img2 = img[0:y,0:x]
    return img2

def black_remove(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
	image,contour,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt =contour[0]
	epsilon = 0.1*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)
	approx = arrange_points(approx)
	x, y, w, h = cv2.boundingRect(cnt)
        #  img = img[y:y+h,x:x+p]
	pts1 = np.float32(approx)
	pts2 = np.float32([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	img =cv2.warpPerspective(img,M,(gray.shape[1],gray.shape[0]))
        #a = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
	image,contour,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt =contour[0]
	x,y,w,h = cv2.boundingRect(cnt)
	img = img[y:y+h,x:x+w]
	return  img



	
def image_stitching(img1,img2,M1):
	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	(h,w,c) = img1.shape
	K = np.array([[807.3179989, 0, 307.83299049], [0, 804.22240703, 229.03518511], [0, 0, 1]])
	img1 = cv2.copyMakeBorder(img1,50,50,300,300, cv2.BORDER_CONSTANT)	

	
	(h,w,c) = img2.shape
	K = np.array([[807.3179989, 0, 307.83299049], [0, 804.22240703, 229.03518511], [0, 0, 1]])
	img2 = cv2.copyMakeBorder(img2,50,50,300,300, cv2.BORDER_CONSTANT)
	img3 = cv2.warpAffine(img2,M1,(gray2.shape[1]+gray1.shape[1],gray1.shape[0]))

	temp = np.zeros_like(img3)
	col = img3.shape[1]-img1.shape[1]
	temp = cv2.resize(temp,(col,img3.shape[0]),interpolation = cv2.INTER_CUBIC)
	print(np.shape(temp),np.shape(img1))
	img1 = np.hstack((img1,temp))
	gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
	print('inside')

	flag = np.array((gray1,gray3))
	indx = np.argmax(flag,axis=0)
	ind1 = np.where(indx ==0)
	ind2 = np.where(indx==1)
	img = np.zeros_like(img1)
	img[ind1] = img1[ind1]
	img[ind2] = img3[ind2]
	
	print('middle')
	return img


def cylindrical_warp(img,K):
    foc_len = (K[0][0] +K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
    # print('img p=color',img[0,0])
    theta= (x- K[0][2])/foc_len # angle theta
    h = (y-K[1][2])/foc_len # height
    p = np.array([np.sin(theta),h,np.cos(theta)])
    p = p.T
    p = p.reshape(-1,3)
    image_points = K.dot(p.T).T
    points = image_points[:,:-1]/image_points[:,[-1]]
    points = points.reshape(img.shape[0],img.shape[1],-1)
    cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
    return cylinder


cap = cv2.VideoCapture('output.mp4')
img = []
K = np.array([[807.3179989, 0, 307.83299049], [0, 804.22240703, 229.03518511], [0, 0, 1]])
while(cap.isOpened()):
    ret,frame = cap.read()
    img = img + [frame]
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('Total Frames :',len(img))
orb = cv2.ORB_create(nfeatures=75)
i = 0
img1 = img[0]
img1 = cylindrical_warp(img1,K)
img1 = cv2.copyMakeBorder(img1,50,50,300,300, cv2.BORDER_CONSTANT)
canvas = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]),np.uint8)
canvas[0:img1.shape[0],0:img1.shape[1]] = img1
while i < len(img)-1:
    img2 = img[i+1]
    img2 = cylindrical_warp(img2,K)
    img2 = cv2.copyMakeBorder(img2,50,50,300,300, cv2.BORDER_CONSTANT)
    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # BRUTE FORCE parameters
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2)
    pt1 = []
    pt2 = []
    good = []
    for m,n in matches:
         if m.distance < 0.7*n.distance:
             good.append(m)
    #good = sorted(good,key = lambda x:x.distance)
    #good = good[:15]
    pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    pt = (pt1 - pt2)
    std = np.std(pt)
    print(i,std)
    if std > 50:
        i = i + 1
        continue
    else:
        
        #output = np.ones(img1.shape, dtype = 'uint8')
        M, mask = cv2.estimateAffine2D(pt1, pt2, cv2.RANSAC, ransacReprojThreshold=0.4)
        M1 = np.array([[1,0,M[0,2]],[0,1,M[1,2]]])
            
       
            #print(M1)
        stitch = image_stitching(canvas,img2,M1)
        canvas = required_img(canvas)
            #H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
            #out1 = cv2.warpAffine(img2, M1, (img1.shape[1]+img2.shape[1],img2.shape[0]))
            #out1[0:img1.shape[0],0:img2.shape[1]] = img1
            #out1 = cv2.warpPerspective(img2,H,(img1.shape[1]+ img2.shape[1],img1.shape[0]))
            #case1 = np.logical_and(img1 == [0,0,0],out1 == [0,0,0])
            #case2 = np.logical_and(img1 == [0,0,0],out1 != [0,0,0])
            #case3 = np.logical_and(img1 != [0,0,0],out1 == [0,0,0])
            #case4 = np.logical_and(img1 != [0,0,0],out1 != [0,0,0])
            #case1index = np.where(case1)
            #case2index = np.where(case2)
            #case3index = np.where(case3)
            #case4index = np.where(case4)
            #output[case1index[0],case1index[1]] = [0,0,0]
            #output[case2index] = out1[case2index]
            #output[case3index] = img1[case3index]
            #output[case4index] = img1[case4index]/2 + out1[case4index]/2
            #cv2.imshow('image',img1)
            #cv2.waitKey(0)
        #cv2.imshow('image',img1)
        #cv2.waitKey(0)
        i = i + 1
        continue
        #except:
        #    i = i + 1
        #    continue
    if 0xff == ord('q'):
        break
cv2.imshow('output',stitch)  
    
cv2.waitKey(0)
cv2.destroyAllWindows()
