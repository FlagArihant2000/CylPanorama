import pyrealsense2 as rs
import cv2
import numpy as np
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)

pipeline.start(config)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('lab25.mp4',fourcc,30.0,(640,480))
i = 1
try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        color_image_1 = np.asanyarray(color.get_data())
        
        #out.write(color_image_1)
        cv2.imshow('frame',color_image_1)
        if cv2.waitKey(1) & 0xff == ord('c'):
        	cv2.imwrite(str(i)+'.png',color_image_1)
        	i = i + 1
finally:
    pipeline.stop()

#out.release()
cv2.destroyAllWindows()
