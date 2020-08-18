from head_detection import headDetection
import cv2 as cv
import numpy as np
import imutils
from imutils import face_utils

cap = cv.VideoCapture(0)

dist_coeffs = np.zeros((4,1))
ret , frame = cap.read()
frame = imutils.resize(frame,width=500)
size = frame.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)

head_analyzer = headDetection()
head_analyzer.camera_calibration(dist_coeffs,focal_length,center)


while True:
    ret,frame = cap.read()
    frame = imutils.resize(frame,width=500)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = head_analyzer.detect_face(gray)
    for face in faces:
        [leftEye,rightEye] = head_analyzer.eye_detector(gray,face)

    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
