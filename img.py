import cv2
import numpy as np


cap = cv2.VideoCapture('output.mp4')
i = 1
while(1):
	ret,frame = cap.read()
	cv2.imwrite(str(i)+'.png',frame)
	i = i + 1
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
