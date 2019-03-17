from copy import deepcopy as dc
from imutils.video import FPS
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import imutils
import os
import sys
import time

# Select Directories For Darknet Files
scriptDir = sys.path[0]
os.chdir(scriptDir)
sys.path.append(os.path.join(scriptDir, 'dnFiles'))

import darknet as dn
        
# Start The Webcam Stream
print("Starting Webcam...")
vs = cv2.VideoCapture(0)
#vs = VideoStream(src=0).start()
time.sleep(1.0)

# Configure YOLO
net = dn.load_net((os.path.join(scriptDir, "dnFiles/yolov3.cfg")).encode(), (os.path.join(scriptDir, "dnFiles/yolov3.weights")).encode(), 0)
meta = dn.load_meta((os.path.join(scriptDir, "dnFiles/coco.data")).encode())
 
# Initialize FPS Estimator
fps = FPS().start()

# Create A Window And Link The Mouse Event To Its Function
cv2.namedWindow('Frame')

while True :
        # Capture Frame From Stream
        frame = vs.read()[1]

        # Check If End Of Stream Has Been Reached
        if frame is None :
                print("Stream Ended")
                break

        # Store Dimensions Of Frame
        H, W = frame.shape[:2]

        # Save The Frame To Memory
        cv2.imwrite("f.png",frame)

        # Run YOLO On The Stored File
        r = dn.detect(net, meta, b"f.png")

        
        # Loop Through The Detections Given By YOLO
        for i in r :
                # Focus Only On The Detections Of Specified Type
                if i[0] == b"person" :
                        # Convert YOLO Coordinates To OpenCV Coordinates
                        [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]

                        # Limit The Coordinates To Image Boundaries
                        [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W-1), min(d_yb, H-1)]

                        # Draw A Rectangle To Show The Detection
                        cv2.rectangle(frame, (d_xa, d_ya), (d_xb, d_yb), (0, 255, 0), 2)
                        
                                                
        # Update The FPS Counter
        fps.update()
        fps.stop()

        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        
        # Display FPS On The Frame
        cv2.putText(frame, "FPS : " + "{:.2f}".format(fps.fps()), (10, 2*H - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)

        # Display The Frame
        cv2.imshow("Frame", frame)

        # Detect Keypress
        key = cv2.waitKey(1) & 0xFF

        # On Pressing 'q', Break From Infinite Loop
        if key == ord("q"):
                break
        # On Pressing 'c', Cancel The Current Tracking
        elif key == ord("c"):
                init = False


# Release The WebCam Pointer
vs.release()
 
# Close All Windows
cv2.destroyAllWindows()
