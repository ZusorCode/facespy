import numpy as np
import cv2 as cv
import time
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
cap = cv.VideoCapture(0)

cv.namedWindow("Live Facial rec", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("Live Facial rec",cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4)
##    if len(faces) != 0:
##        cv.imwrite(f"{time.time()}.png", img)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=3)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv.imshow("Live Facial rec",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
                exit()
