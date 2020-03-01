import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import time as t
"""Use of SIFT for feature detection, FLANN for feature matching and using translational model on cylindrical images, following by inverse cylindrical operations."""
#the values of K have been taken for the camera in the lab.
def feature_matching(img1, img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create(nfeatures=75)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    print('1.1')
    # FLANN parameters
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    pt1 = []
    pt2 = []
    print('1.2')
    for m,n in matches:
         if m.distance < 0.7*n.distance:
             good.append(m)
    #good = sorted(good,key = lambda x:x.distance)
    #good = good[:15]
    pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2) 
    return pt1, pt2


def cylinderconversion(img,K):
    foc_len = (K[0][0] +K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
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


def affine(img1, img2):
    img1_pts,img2_pts = feature_matching(img1,img2)
    pts = img1_pts - img2_pts
    std = np.std(pts)
    #img1_pts = np.float32(pts1).reshape(-1,1,2)
    #img2_pts = np.float32(pts2).reshape(-1,1,2)
    M, mask = cv2.estimateAffine2D(img1_pts, img2_pts, cv2.RANSAC, ransacReprojThreshold=0.4)
    M = np.array([[1,0,M[0,2]],[0,1,M[1,2]]])
    return M


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

def arrange_points(points):
	number = np.array([a[0] for a in points])
	x_cen = 0
	y_cen = 0
	for [x, y] in number:
		x_cen = x_cen + x
		y_cen = y_cen + y
	x_cen = x_cen // 4
	y_cen = y_cen // 4
    # img = cv2.circle(img,(x_cen,y_cen),2 , (0,0,255), -1)
	sorted_x = np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])
	for [x, y] in number:
		if x <= x_cen and y <= y_cen:
			sorted_x[0][0] = np.array([x, y])
		if x <= x_cen and y >= y_cen:
			sorted_x[1][0] = np.array([x, y])
		if x >= x_cen and y >= y_cen:
			sorted_x[2][0] = np.array([x, y])
		if x >= x_cen and y <= y_cen:
			sorted_x[3][0] = np.array([x, y])
	points = np.array(sorted_x)
	return  points

def cylinderstitch(img1, img2):
	start = t.time()
	(h,w,c) = img1.shape
	#f = 582.5
	K = np.array([[807.3179989, 0, 307.83299049], [0, 804.22240703, 229.03518511], [0, 0, 1]]) # mock calibration matrix
	img1 = cylinderconversion(img1, K)
	img1 = cv2.copyMakeBorder(img1,50,50,300,300, cv2.BORDER_CONSTANT)
	
	
	(h,w,c) = img2.shape
	#f = 582.5
	K = np.array([[807.3179989, 0, 307.83299049], [0, 804.22240703, 229.03518511], [0, 0, 1]]) # mock calibration matrix
	img2 = cylinderconversion(img2, K)
	img2 = cv2.copyMakeBorder(img2,50,50,300,300, cv2.BORDER_CONSTANT)
   
	
	#(h,w,c) = img3.shape
	#f = 583
	#K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	#img3 = cylinderconversion(img3, K)
	#img3 = cv2.copyMakeBorder(img3,50,50,300,300, cv2.BORDER_CONSTANT)
	print('1')
	#(M, pts1, pts2, mask5) = affine(img3, img1)
	M1 = affine(img2, img1)
		
	print('2')
	#out1 = cv2.warpAffine(img3, M, (img1.shape[1],img1.shape[0]))
	out1 = cv2.warpAffine(img2, M1, (img1.shape[1],img1.shape[0]))
	output = np.ones(img1.shape, dtype = 'uint8')
	#output1 = np.ones(output.shape, dtype = 'uint8')
	#gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
	#grayout1 = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)

	case1 = np.logical_and(img1 == [0,0,0],out1 == [0,0,0])
	case2 = np.logical_and(img1 == [0,0,0],out1 != [0,0,0])
	case3 = np.logical_and(img1 != [0,0,0],out1 == [0,0,0])
	case4 = np.logical_and(img1 != [0,0,0],out1 != [0,0,0])
	case1index = np.where(case1)
	case2index = np.where(case2)
	case3index = np.where(case3)
	case4index = np.where(case4)
	output[case1index[0],case1index[1]] = [0,0,0]
	output[case2index] = out1[case2index]
	output[case3index] = img1[case3index]
	output[case4index] = img1[case4index]/2 + out1[case4index]/2 # getting ghosting because of this command

	print('3')
		
	#			
	print('_________________')

	end = t.time()
	print("TIME in seconds: ",(end-start))
	return output,out1

def planing(stitch):
	output = np.zeros(stitch.shape,dtype = 'uint8')
	xc = 0
	yc = 0
	f = 650
	theta = stitch/650


cap = cv2.VideoCapture(0)
current = t.time()
img = []
counter = 0
while(1):
	decision = int(input('Enter the direction of rotation of camera: (clockwise: 0, anticlockwise: 1)'))
	if decision == 1 or decision == 0:
		break
	else:
		print('Invalid Input')
while(1):
	ret,frame = cap.read()
	print('Press "q" to exit, press "c" to take a snapshot')
	cv2.imshow('frame',frame)
	key = cv2.waitKey(1)
	if key == ord('c'):
		print("CAPTURED!!")
		h, w = frame.shape[:2]
		#K = np.array([[600,0,w/2],[0,600,h/2],[0,0,1]])
		#cyl,cyl_mask = cylindricalWarpImage(frame, K)
		img = img + [frame]
		
	if key == ord('q'):
		print("QUIT")
		break
"""while(1):
	try:
		print(counter)
		counter = counter + 1
		ret,frame = cap.read()
		#h,w = frame.shape[:2]
		#K = np.array([[600,0,w/2],[0,600,h/2],[0,0,1]])
		#cyl,cyl_mask = cylindricalWarpImage(frame,K)
		if counter%50 == 0:
			img = img + [frame]
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
	except:
		break
cap.release()
cv2.destroyAllWindows()"""
print("Total Images: "+str(len(img)))
#for i in range(len(img)):	
#	cv2.imshow('image '+str(i+1),img[i])
starttime = t.time()
stitch,out1 = cylinderstitch(img[0],img[1]) if decision == 1 else cylinderstitch(img[1],img[0])
j = 2
while j < len(img):
	stitch,out2 = cylinderstitch(stitch,img[j]) if decision == 1 else cylinderstitch(img[j],stitch)
	j = j+1
#stitch = black_remove(stitch)
#image1 = cv2.imread('1.jpeg')
#image2 = cv2.imread('2.jpeg')
#image3 = cv2.imread('3.jpeg')
#image4 = cv2.imread('4.jpeg')
#stitch,out = cylinderstitch(image1,image2)
#stitch = black_remove(stitch)
#stitch = cylinderstitch(stitch,image3)
#stitch = black_remove(stitch)
#stitch = cylinderstitch(stitch,image4)
#stitch = black_remove(stitch)
#stitch = planing(stitch)
cv2.imshow('Panaroma',stitch)
#cv2.imshow('out',out)
#cv2.imshow('1',image1)
#cv2.imshow('2',image2)
#cv2.imshow('3',image3)
#cv2.imshow('4',image4)

cv2.waitKey(0)
cv2.destroyAllWindows()
