''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc


'''
from time import sleep

import cv2
import os

from config import CascadeClassifierDefault
from toolset import rotate

video = "http://admin:admin@192.168.8.106:8081/"
# video = "testdata/FeiFace1.mp4"
cam = cv2.VideoCapture(video)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier(CascadeClassifierDefault)

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count

maxCount = 0
for fileName in os.listdir('dataset'):
    nowFile = os.path.split(fileName)[-1].split(".")
    if int(nowFile[1]) == int(face_id):
        maxCount = max(maxCount, int(nowFile[2]))

print('maxCount:%d' % maxCount)

count = maxCount

while (True):

    ret, img = cam.read()
    img = rotate(img, 90)
    # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    nowCount = count - maxCount
    maxPhotos = 30
    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif nowCount >= maxPhotos:  # Take 30 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
