# -*- coding: utf-8 -*-
# coding:utf-8
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
from config import CascadeClassifierDefault


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


def getSingleImagesAndLabels(path, faceid, detector):
    faceSamples = []
    ids = []

    PIL_img = Image.open(path).convert('L')  # convert it to grayscale
    img_numpy = np.array(PIL_img, 'uint8')

    id = int(faceid)
    print('id:%d' % id)
    faces = detector.detectMultiScale(img_numpy)

    for (x, y, w, h) in faces:
        faceSamples.append(img_numpy[y:y + h, x:x + w])
        ids.append(id)

    return faceSamples, ids


def trainSingleImage(gray, x, y, w, h, faceCascade, recognizer):
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

