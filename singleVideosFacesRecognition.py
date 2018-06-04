# -*- coding: utf-8 -*-
# coding:utf-8

import cv2
import numpy as np
import os
from config import names, CascadeClassifierDefault
from toolset import rotate, getSingleImagesAndLabels

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(CascadeClassifierDefault)

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# Initialize and start realtime video capture
video = "testdata/FeiFace1.mp4"
cam = cv2.VideoCapture(video)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    if not ret:
        break
    # img = rotate(img, -90)
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
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
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

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
