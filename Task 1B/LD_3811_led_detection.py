# Team ID:     3811
# Theme:       Luminosity Drone
# Filename:    LD_3811_led_detection.py

# Finds the bright LEDs in an image and writes their centroid and area.
# usage: python3 LD_3811_led_detection.py [image]   (default led.jpg)

from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'led.jpg'
image = cv2.imread(path, 1)
if image is None:
    sys.exit("could not read " + path)

# grayscale and blur so each LED becomes one smooth blob
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11, 11), 0)

# keep only the bright regions
thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]

# erode then dilate to remove small specks of noise
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# label connected regions and keep the big ones in a mask
labels = measure.label(thresh, connectivity=2, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

for label in np.unique(labels):
    # label 0 is the background
    if label == 0:
        continue

    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    if numPixels > 300:
        mask = cv2.add(mask, labelMask)

# contours of the LEDs, sorted left to right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
if len(cnts) > 0:
    cnts = contours.sort_contours(cnts)[0]

centroids = []
areas = []

for i, c in enumerate(cnts):
    area = cv2.contourArea(c)

    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = (M["m10"] / M["m00"])
        cY = (M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # outline, centre dot and number for each LED
    cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
    cv2.circle(image, (int(cX), int(cY)), 7, (255, 255, 255), -1)
    cv2.putText(image, "#{}".format(i + 1), (int(cX) - 20, int(cY) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    centroids.append((cX, cY))
    areas.append(area)

cv2.imwrite("LD_3811_led_detection_results.png", image)

with open("LD_3811_led_detection_results.txt", "w") as file:
    file.write(f"No. of LEDs detected: {len(cnts)}\n")
    for i, (centroid, area) in enumerate(zip(centroids, areas)):
        file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")
