from os.path import exists
import cv2
import numpy as np
import math
from scipy import ndimage
import os
import time


# reads and crops images base upon the mask
def cropped_Images(file):
    img = cv2.imread(file)
    lower = np.array([0, 100, 20])
    upper = np.array([105, 250, 255])
    mask = cv2.inRange(img, lower, upper)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_coordinates = []
    if len(contours) != 0:
        count = 0
        for contour in contours:
            if count == 4:
                break
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                print(x, y, w, h)
                coordinates = [x, y, w, h]
                all_coordinates.append(coordinates)
                '''cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.imshow("image", img)
                cv2.waitKey(0)'''
                count += 1
                return_pics = []
    name = 0
    for coordinate in all_coordinates:
        if not os.path.exists("CroppedImage1.jpeg"):
            cv2.imwrite("CroppedImage" + str(name) + ".jpeg", img[coordinate[1]:coordinate[1] + coordinate[3],
                                                              coordinate[0]:coordinate[0] + coordinate[2]])
        return_pics.append(img[coordinate[1]:coordinate[1] + coordinate[3],
                          coordinate[0]:coordinate[0] + coordinate[2]])
        name += 1
    return return_pics


# firstCropped = img[x:x+w, y:y+h]
# cv2.imshow('image-1', img)
# cv2.waitKey(0)

cropped_images = cropped_Images("1631657219229-d3132052-4701-4fe3-ae90-5406cfe17c82.jpg")
for item in cropped_images:
    cv2.imshow('image', item)
    cv2.waitKey(0)

# converts image to proper formats
# images = np.zeros((1, 784))
img = cv2.imread("1631657219229-d3132052-4701-4fe3-ae90-5406cfe17c82.jpg")
croppedImage = img[756 - 10:756 + 65 + 10, 244 - 10:244 + 33 + 10]
resizedImage = cv2.resize(croppedImage, (28, 28))
greyScaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
(thresh, whiteAndBlackImage) = cv2.threshold(greyScaleImage, 180, 255, cv2.THRESH_BINARY)
blackAndWhiteImage = cv2.bitwise_not(whiteAndBlackImage)
data = blackAndWhiteImage
data = data / 255.0
# print(data)
# cv2.imshow("image", data)
cv2.waitKey(0)

# remove black pixels at edge of image
while np.sum(data[0]) == 0:
    data = data[1:]

while np.sum(data[:, 0]) == 0:
    data = np.delete(data, 0, 1)

while np.sum(data[-1]) == 0:
    data = data[:-1]

while np.sum(data[:, -1]) == 0:
    data = np.delete(data, -1, 1)

rows, cols = data.shape

data = cv2.resize(data, (20, 20))
print(data)

colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
data = np.lib.pad(data, (rowsPadding, colsPadding), 'constant')

# print(data)
