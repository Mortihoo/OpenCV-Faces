import numpy as np
import cv2

from config import CascadeClassifierDefault
from toolset import rotate


# faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_alt.xml')
faceCascade = cv2.CascadeClassifier(CascadeClassifierDefault)
video = "http://admin:admin@192.168.8.106:8081/"


cap = cv2.VideoCapture(video)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height
while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    img = rotate(img, 90)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(64, 48)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
