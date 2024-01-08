'''
# Team ID:          < 3811 >
# Theme:            < Luminosity Drone >
# Filename:         < LD_3811_led_detection.py >
# Functions:        < euclidean_distance,  determine_organism_type, process_led_image>
# Global variables: < None >
'''

import argparse
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

def euclidean_distance(pt1, pt2):

    '''
    Calculate the Euclidean distance between two points.

    Arguments:
    ---
    pt1 : tuple
        Coordinates of the first point (x1, y1).
    pt2 : tuple
        Coordinates of the second point (x2, y2).

    Returns:
    ---
    distance : float
        Euclidean distance between the two points.
    '''

    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def determine_organism_type(num_leds):

    '''
    Determine the type of organism based on the number of LEDs.

    Arguments:
    ---
    num_leds : int
        Number of LEDs detected.

    Returns:
    ---
    organism_type : str
        Type of the organism based on LED count.
    '''

    if num_leds == 2:
        return "alien_a"
    elif num_leds == 3:
        return "alien_b"
    elif num_leds == 4:
        return "alien_c"
    elif num_leds == 5:
        return "alien_d"
    else:
        return "unknown"

def process_led_image(image_path):

    '''
    Process an image to detect and analyze LED clusters representing organisms.

    Arguments:
    ---
    image_path : str
        Path to the input image file.

    Returns:
    ---
    None
    '''

    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image based on specific color ranges for white and pink
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask_white = cv2.inRange(image, lower_white, upper_white)

    lower_pink = np.array([150, 100, 150])
    upper_pink = np.array([200, 150, 200])
    mask_pink = cv2.inRange(image, lower_pink, upper_pink)

    # Combine masks for white and pink
    mask = cv2.bitwise_or(mask_white, mask_pink)

    # Erosion and dilation to enhance the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Connected component analysis
    labels = measure.label(mask, connectivity=2, background=0)
    mask = np.zeros(mask.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(mask.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 100:  # Adjust this threshold based on your specific case
            mask = cv2.add(mask, labelMask)

    # Find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    # Initialize lists to store centroid coordinates
    centroids = []

    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
        cv2.circle(image, (int(cX), int(cY)), 7, (255, 255, 255), -1)
        

        centroids.append((cX, cY))

    # Group centroids that are close to each other within a threshold
    clustered_centroids = []
    clustered = set()

    for i, centroid1 in enumerate(centroids):
        if i in clustered:
            continue

        cluster = [centroid1]
        for j, centroid2 in enumerate(centroids[i + 1:], start=i + 1):
            if euclidean_distance(centroid1, centroid2) < 200 and j not in clustered:
                cluster.append(centroid2)
                clustered.add(j)

        if cluster:
            clustered_centroids.append(cluster)

    # Calculate number of LEDs in each organism cluster and its centroid
    organism_info = []
    for idx, cluster in enumerate(clustered_centroids, start=1):
        num_leds = len(cluster)
        organism_type = determine_organism_type(num_leds)
        centroid_x = sum(pt[0] for pt in cluster) / num_leds
        centroid_y = sum(pt[1] for pt in cluster) / num_leds
        cv2.putText(image, "{}".format(num_leds), (int(centroid_x), int(centroid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        organism_info.append((f"Organism_{idx}", (centroid_x, centroid_y), num_leds, organism_type))

    # Save the output image as a PNG file
    output_image_path = image_path.split('.')[0] + "_results.png"
    cv2.imwrite(output_image_path, image)

    # Save organism information to a text file
    output_txt_file = image_path.split('.')[0] + ".txt"
    with open(output_txt_file, "w") as file:
        for organism, centroid, num_leds, organism_type in organism_info:
            file.write(f"Organism Type: {organism_type}\n")
            file.write(f"Centroid: {centroid}\n\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to the input image")
    args = vars(ap.parse_args())
    
    image_path = args["image"]
    process_led_image(image_path)
