# -*- coding: utf-8 -*-
# coding:utf-8

import cv2
import numpy as np
import os
import argparse
from config import names, CascadeClassifierDefault
from toolset import rotate, getSingleImagesAndLabels

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(CascadeClassifierDefault)

font = cv2.FONT_HERSHEY_SIMPLEX

# 构造参数解析器
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# Initialize and start realtime video capture
img = cv2.imread('testdata/wj2.jpg')
# cv2.imshow("Original", img)

# Define min window size to be recognized as a face
minW = 0.1 * img.shape[:2][1]
minH = 0.1 * img.shape[:2][0]


# img = rotate(img, -20)
# img = cv2.flip(img, -1)  # Flip vertically
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(int(minW), int(minH)),
)

for (x, y, w, h) in faces:

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

    # Check if confidence is less them 100 ==> "0" is perfect match
    if (confidence <= 20):
        id = names[id]
        confidence = "  {0}%".format(round(100 - confidence))
    elif confidence <= 90:
        id = names[id] + '(unsure)'
        confidence = "  {0}%".format(round(100 - confidence))
    else:
        id = "unknown"
        confidence = "  {0}%".format(round(100 - confidence))

    cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(1000) & 0xff  # Press 'ESC' for exiting video
    if k == 13:
        face_id = input('\n enter user id end press <return> ==>  ')
        maxCount = 0
        for fileName in os.listdir('dataset'):
            nowFile = os.path.split(fileName)[-1].split(".")
            if int(nowFile[1]) == int(face_id):
                maxCount = max(maxCount, int(nowFile[2]))
        maxCount += 1
        imgFileName = "dataset/User." + str(face_id) + '.' + str(maxCount) + ".jpg"
        cv2.imwrite(imgFileName, gray[y:y + h, x:x + w])
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces2, ids2 = getSingleImagesAndLabels(imgFileName, face_id, faceCascade)
        recognizer.update(faces2, np.array(ids2))
        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi
        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained.".format(len(np.unique(ids2))))
        break

if img.shape[:2][1] / 1920.0 > img.shape[:2][0] / 1080.0 and img.shape[:2][1] > 1920:
    r = img.shape[:2][1] / 1920.0
    newW = 1920
    newH = img.shape[:2][0] / r
    img = cv2.resize(img, (int(newW), int(newH)), interpolation=cv2.INTER_AREA)
elif img.shape[:2][1] / 1920.0 < img.shape[:2][0] / 1080.0 and img.shape[:2][0] > 1080:
    r = img.shape[:2][0] / 1080.0
    newW = img.shape[:2][1] / r
    newH = 1080
    img = cv2.resize(img, (int(newW), int(newH)), interpolation=cv2.INTER_AREA)

cv2.imshow('camera', img)

k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video

while True:
    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()
