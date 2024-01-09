# Team ID:     3811
# Theme:       Luminosity Drone
# Filename:    LD_3811_led_detection.py

# Finds LED clusters (organisms) in an image from the real drone camera.
# Each LED is found on its own, then LEDs closer than --dist pixels are put
# in the same cluster and the cluster type comes from its LED count.
#
# usage: python3 LD_3811_led_detection.py --image photo.jpg [--dist 100]
# writes photo_results.png and photo.txt next to the image

import argparse
import os
import sys
from skimage import measure
import numpy as np
import imutils
import cv2


def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def determine_organism_type(num_leds):
    if num_leds == 2:
        return "alien_a"
    elif num_leds == 3:
        return "alien_b"
    elif num_leds == 4:
        return "alien_c"
    elif num_leds == 5:
        return "alien_d"
    return "unknown"


def find_leds(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # only the white centre of each LED is above 225, the pink glow around it is not.
    # this keeps two LEDs that touch in the image as two separate blobs.
    thresh = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)

    # drop small specks
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        if cv2.countNonZero(labelMask) > 30:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    leds = []
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        # LEDs are round, bright strips like reflections or a window bar are not
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.6:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        leds.append((c, (M["m10"] / M["m00"], M["m01"] / M["m00"])))

    return leds


# LEDs that are within max_dist of any LED already in a cluster join that cluster
def group_leds(centroids, max_dist):
    clusters = []
    unused = list(range(len(centroids)))

    while unused:
        cluster = [unused.pop(0)]
        k = 0
        while k < len(cluster):
            near = [j for j in unused if euclidean_distance(centroids[cluster[k]], centroids[j]) < max_dist]
            for j in near:
                unused.remove(j)
            cluster += near
            k += 1
        clusters.append([centroids[i] for i in cluster])

    return clusters


def process_led_image(image_path, max_dist):
    image = cv2.imread(image_path)
    if image is None:
        sys.exit("could not read " + image_path)

    leds = find_leds(image)
    centroids = [centroid for c, centroid in leds]

    for c, (cX, cY) in leds:
        cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
        cv2.circle(image, (int(cX), int(cY)), 3, (255, 255, 255), -1)

    # left to right so the output order does not change between runs
    clusters = group_leds(centroids, max_dist)
    clusters.sort(key=lambda cluster: min(pt[0] for pt in cluster))

    organism_info = []
    for cluster in clusters:
        num_leds = len(cluster)
        organism_type = determine_organism_type(num_leds)
        centroid_x = sum(pt[0] for pt in cluster) / num_leds
        centroid_y = sum(pt[1] for pt in cluster) / num_leds
        cv2.putText(image, "{}".format(num_leds), (int(centroid_x), int(centroid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        organism_info.append(((centroid_x, centroid_y), organism_type))

    base = os.path.splitext(image_path)[0]
    cv2.imwrite(base + "_results.png", image)

    with open(base + ".txt", "w") as file:
        for centroid, organism_type in organism_info:
            file.write(f"Organism Type: {organism_type}\n")
            file.write(f"Centroid: {centroid}\n\n")

    print(f"{len(centroids)} LEDs in {len(clusters)} clusters, saved {base}.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to the input image")
    ap.add_argument("--dist", type=float, default=100, help="max pixel gap between LEDs of one cluster")
    args = ap.parse_args()

    process_led_image(args.image, args.dist)
