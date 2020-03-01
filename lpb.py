import cv2
import numpy as np


def cylindrical_warp(img,K):
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
    
img = cv2.imread('4.jpeg')
k = np.array([[807.3179989, 0, 307.83299049], [0, 804.22240703, 229.03518511], [0, 0, 1]])

cv2.imshow('cylinder',cylindrical_warp(img,k))
cv2.waitKey(0)
cv2.destroyAllWindows()
