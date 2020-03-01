import cv2
import numpy as np
import os
import time as t
import math


def cylindricalWarpImage(img1, K):
	f = K[0,0]
	(im_h,im_w,channel) = img1.shape

    # go inverse from cylindrical coord to the image
	cyl = np.zeros_like(img1)
	cyl_mask = np.zeros_like(img1)
	(cyl_h,cyl_w,channel) = cyl.shape
	x_c = float(cyl_w) / 2.0
	y_c = float(cyl_h) / 2.0

	for x_cyl in np.arange(0,cyl_w):
		for y_cyl in np.arange(0,cyl_h):
			theta = (x_cyl - x_c) / f
			h     = (y_cyl - y_c) / f
			X = np.array([math.sin(theta), h, math.cos(theta)])
			X = np.dot(K,X)
			x_im = X[0] / X[2]
			if x_im < 0 or x_im >= im_w:
				continue
			y_im = X[1] / X[2]
			if y_im < 0 or y_im >= im_h:
				continue

			cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
			cyl_mask[int(y_cyl),int(x_cyl)] = 255
	return (cyl,cyl_mask)

def inverseCylinder(img1, K):
	f = K[0,0]
	(im_h,im_w,channel) = img1.shape

    # go inverse from cylindrical coord to the image
	cyl = np.zeros_like(img1)
	cyl_mask = np.zeros_like(img1)
	(cyl_h,cyl_w,channel) = cyl.shape
	x_c = float(cyl_w) / 2.0
	y_c = float(cyl_h) / 2.0

	for x_cyl in np.arange(0,cyl_w):
		for y_cyl in np.arange(0,cyl_h):
			theta = (x_cyl - x_c) / f
			h     = (y_cyl - y_c) / f
			X = np.array([math.sin(theta), h, math.cos(theta)])
			X = np.dot(K,X)
			x_im = f*(X[0]/X[2]) + x_c
			#x_im = X[0] / X[2]
			if x_im < 0 or x_im >= im_w:
				continue
			#y_im = X[1] / X[2]
			y_im = f*(X[1] / X[2]) + y_c
			if y_im < 0 or y_im >= im_h:
				continue

			cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
			cyl_mask[int(y_cyl),int(x_cyl)] = 255
	return (cyl,cyl_mask)

def stitching(img1, img2, gray1, gray2):

	#print(img1.shape)
	sift = cv2.xfeatures2d.SIFT_create()
	#sift = cv2.ORB_create()
	#kp1 = sift.detect(gray1, None)
	#kp1, desc1 = sift.compute(gray1, kp1)
	kp1,desc1 = sift.detectAndCompute(gray1,None)
	#sift1 = cv2.drawKeypoints(img1,kp1,img1)
	kp2,desc2 = sift.detectAndCompute(gray2,None)
	#sift2 = cv2.drawKeypoints(img2,kp2,img2)
	#kp2 = sift.detect(gray2, None)
	#kp2, desc2 = sift.compute(gray2, kp2)
	bf = cv2.BFMatcher()
	try:
		matches = bf.knnMatch(desc1,desc2,k=2)
	except: 
		pass
	#print(matches)
	#matches = sorted(matches, key = lambda x:x.distance)
	good = []
	pt1 = []
	pt2 = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
	pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
	#mean = np.mean(pt1-pt2)
	#std = np.std(pt1-pt2)
	#print(mean)
	#print(std)
	#print(pt2)
	H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
	#gray1.shape = [gray1.shape[0]+50,gray1.shape[1]]
	#gray2.shape = [gray2.shape[0]+50,gray2.shape[1]]
	#  inverse of  the homographhy we got from pt1 to pt2
	try: 
		H = np.linalg.inv(H)
	except:
		H = H
	try:
		gray33 = cv2.warpPerspective(img2,H,(gray1.shape[1]+ gray2.shape[1],gray2.shape[0]))
		gray33[0:gray1.shape[0],0:gray1.shape[1]] = img1
		return gray33
	except:
		pass
	



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


cap = cv2.VideoCapture('reverse.mp4')
img = []
imggray = []
while(1):
	decision = int(input('Enter the direction of rotation of camera: (clockwise: 0, anticlockwise: 1)'))
	if decision == 1 or decision == 0:
		break
	else:
		print('Invalid Input')
canvas = np.zeros((1000,1000,3),dtype = 'uint8')
counter = 0
totalimages = 0
while(1):
	#try:	
		ret,frame = cap.read()
		#print(frame.shape)
		frame = cv2.resize(frame,(709,400))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		totalimages = totalimages + 1
		if counter == 0 or counter == 1:
			img = img + [frame]
			counter = counter + 1
			imggray = imggray + [gray]
		elif counter == 2:
			stitch = stitching(img[1],img[0],imggray[1],imggray[0]) if decision == 1 else stitching(img[0],img[1],imggray[0],imggray[1])
			#stitch = black_remove(stitch)
			stitchgray = cv2.cvtColor(stitch,cv2.COLOR_BGR2GRAY)
			counter = counter + 1
		else:
			if counter%200 == 0:
				stitch = stitching(frame, stitch, gray, stitchgray) if decision == 1 else stitching(stitch, frame, stitchgray, gray)
					#stitch = black_remove(stitch)
				stitchgray = cv2.cvtColor(stitch, cv2.COLOR_BGR2GRAY)
			counter = counter + 1
		if counter <= 2:
			pass
		else:
			cv2.imshow('panaroma',stitch)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
	#except:
	#	break

cap.release()
print("Total Images: ",totalimages)
stitch = black_remove(stitch)
cv2.imshow('final panaroma',stitch)
cv2.imwrite('panaroma.png',stitch)
cv2.waitKey(0)
cv2.destroyAllWindows()

