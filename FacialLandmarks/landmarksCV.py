import cv2
import urllib.request as urlreq
import os
import numpy as np
import time
# save picture's url in pics_url variable

# save picture's name as pic

# download picture from url and save locally as image.jpg

# read image with openCV
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("Haar model exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("Haar model downloaded")

# create an instance of the Face Detection Cascade Classifier
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("LBF model exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("LBF model downloaded")
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)
    
cap = cv2.VideoCapture(0)
fps = np.array([time.time()])
n=1
while(True):
    # Capture frame-by-frame
    _, frame = cap.read()

    # Our operations on the frame come here
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image_gray)
    
    # Print coordinates of detected faces
    
    
    # save facial landmark detection model's url in LBFmodel_url variable
    
    
#     create an instance of the Facial landmark Detector with the model
    if len(faces) > 0:
    # Detect landmarks on "image_gray"
        _, landmarks = landmark_detector.fit(image_gray, faces)
        fps = np.append(fps, [time.time()])
        if (n == 5):
            fps = 1/np.mean(np.diff(fps))
            print(fps)
            fps = np.array([time.time()])
            n = 0
        n = n+1  
        for landmark in landmarks:
            for x,y in landmark[0]:
        		# display landmarks on "image_cropped"
        		# with white colour in BGR and thickness 1
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow('my webcam', frame)
        # Display the resulting frame
#
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()