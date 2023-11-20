#!/usr/bin/env python3

# Team ID:     3811
# Theme:       Luminosity Drone
# Filename:    LD_3811_biota_detector.py

# Task 2C: same idea as 2B but there are several LED clusters in the arena.
# The drone flies a search route, centres over every cluster it sees,
# publishes it on /astrobiolocation and keeps going. At the end it flies
# to the research station and lands.

import rospy
from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
from luminosity_drone.msg import Biolocation
from sensor_msgs.msg import Image
import cv2
import numpy as np
from skimage import measure
import imutils
from cv_bridge import CvBridge, CvBridgeError


class swift:
    def __init__(self):
        rospy.init_node('drone_control')
        self.drone_position = [0.0,0.0,0.0]
        self.led_detected = False
        self.final_x = False
        self.final_y = False
        self.landing = False
        self.moveon = True
        self.reachcenter = False

        # whycon x, y of the clusters already published
        self.found = []

        # average centroid of the LEDs in the camera image (500x500)
        self.center_x = 0
        self.center_y = 0
        self.hover_x = 0
        self.hover_y = 0

        # tolerance for reaching a waypoint, loose while searching
        self.abs_err = 1.5

        # search route
        self.setpoints = [
            [-9, -9, 28],
            [-9, -4, 28],
            [-9, 0, 28],
            [-9, 4, 28],
            [-9.5, 9.5, 28],
            [0, 9, 28],
            [0, 4, 28],
            [0, 0, 28],
            [0, -4, 28],
            [0, -9, 28],
            [7, -9, 28],
            [7, -4, 28],
            [7, 0, 28],
            [6, 4, 28],
            [6, 6, 28],
            [7, 9, 28]
        ]
        self.current_setpoint = 0

        # (number of leds, centroids, areas) from the latest camera frame
        self.led_detection_result = None
        self.bridge = CvBridge()

        self.cmd = swift_msgs()
        self.cmd.rcRoll = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcThrottle = 1500
        self.cmd.rcAUX1 = 1500
        self.cmd.rcAUX2 = 1500
        self.cmd.rcAUX3 = 1500
        self.cmd.rcAUX4 = 1500

        # gains for [roll, pitch, throttle]
        self.Kp = [29,19.5,11.5]
        self.Ki = [0.06,0.08,0.07]
        self.Kd = [281,240,269]

        self.error = [0, 0, 0]
        self.prev_error = [0, 0, 0]
        self.differential_error = [0, 0, 0]
        self.sum_error = [0, 0, 0]
        self.min_values = [1000, 1000, 1000]
        self.max_values = [2000, 2000, 2000]

        self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
        self.throttleError = rospy.Publisher('/alt_error', Float64, queue_size=1)
        self.pitchError = rospy.Publisher('/pitch_error', Float64, queue_size=1)
        self.rollError = rospy.Publisher('/roll_error', Float64, queue_size=1)
        self.biolocation_pub = rospy.Publisher('/astrobiolocation', Biolocation, queue_size=1)

        rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
        rospy.Subscriber('/pid_tuning_altitude', PidTune, self.altitude_set_pid)
        rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)
        rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)
        rospy.Subscriber('/swift/camera_rgb/image_raw', Image, self.image_callback)

        # one rate shared by all the loops, 30 Hz
        self.rate = rospy.Rate(30)

        self.arm()

    def disarm(self):
        self.cmd.rcAUX4 = 1100
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)

    def arm(self):
        self.disarm()
        self.cmd.rcRoll = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcThrottle = 1000
        self.cmd.rcAUX4 = 1500
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)

    def whycon_callback(self, msg):
        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z

    # reset integral and derivative terms when the target changes
    def set_setpoint(self, setpoint):
        self.setpoints[self.current_setpoint] = setpoint
        self.sum_error = [0, 0, 0]
        self.prev_error = [0, 0, 0]

    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.Kp * 0.06
        self.Ki[2] = alt.Ki * 0.00008
        self.Kd[2] = alt.Kd * 0.3

    def pitch_set_pid(self, pitch):
        self.Kp[1] = pitch.Kp * 0.06
        self.Ki[1] = pitch.Ki * 0.00008
        self.Kd[1] = pitch.Kd * 0.3

    def roll_set_pid(self, roll):
        self.Kp[0] = roll.Kp * 0.06
        self.Ki[0] = roll.Ki * 0.00008
        self.Kd[0] = roll.Kd * 0.3

    def finished(self):
        return self.current_setpoint >= len(self.setpoints)

    def pid(self):
        # nothing left to fly to after landing
        if self.finished():
            return

        self.error[0] = -(self.drone_position[0] - self.setpoints[self.current_setpoint][0])
        self.error[1] = self.drone_position[1] - self.setpoints[self.current_setpoint][1]
        self.error[2] = self.drone_position[2] - self.setpoints[self.current_setpoint][2]

        self.differential_error[0] = self.error[0] - self.prev_error[0]
        self.differential_error[1] = self.error[1] - self.prev_error[1]
        self.differential_error[2] = self.error[2] - self.prev_error[2]

        self.sum_error[0] = self.sum_error[0] + self.error[0]
        self.sum_error[1] = self.sum_error[1] + self.error[1]
        self.sum_error[2] = self.sum_error[2] + self.error[2]

        self.cmd.rcRoll = int(1500 + self.error[0] * self.Kp[0] + self.differential_error[0] * self.Kd[0] + self.sum_error[0] * self.Ki[0])
        self.cmd.rcPitch = int(1500 + self.error[1] * self.Kp[1] + self.differential_error[1] * self.Kd[1] + self.sum_error[1] * self.Ki[1])
        self.cmd.rcThrottle = int(1560 + self.error[2] * self.Kp[2] + self.differential_error[2] * self.Kd[2] + self.sum_error[2] * self.Ki[2])

        if self.cmd.rcThrottle > self.max_values[2]:
            self.cmd.rcThrottle = self.max_values[2]
        if self.cmd.rcThrottle < self.min_values[2]:
            self.cmd.rcThrottle = self.min_values[2]

        if self.cmd.rcPitch > self.max_values[1]:
            self.cmd.rcPitch = self.max_values[1]
        if self.cmd.rcPitch < self.min_values[1]:
            self.cmd.rcPitch = self.min_values[1]

        if self.cmd.rcRoll > self.max_values[0]:
            self.cmd.rcRoll = self.max_values[0]
        if self.cmd.rcRoll < self.min_values[0]:
            self.cmd.rcRoll = self.min_values[0]

        self.command_pub.publish(self.cmd)
        self.prev_error[0] = self.error[0]
        self.prev_error[1] = self.error[1]
        self.prev_error[2] = self.error[2]

        self.rollError.publish(self.error[0])
        self.pitchError.publish(self.error[1])
        self.throttleError.publish(self.error[2])

    # same steps as Task 1B, returns (number of leds, centroids, areas)
    def detect_leds(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        labels = measure.label(thresh, connectivity=2, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > 300:
                mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        centroid = []
        areas = []

        for c in cnts:
            area = cv2.contourArea(c)

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = (M["m10"] / M["m00"])
                cY = (M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            centroid.append([cX, cY])
            areas.append(area)

        # middle of the whole cluster, used for centring the drone
        if centroid:
            self.center_x = sum(p[0] for p in centroid) / len(centroid)
            self.center_y = sum(p[1] for p in centroid) / len(centroid)

        return len(cnts), centroid, areas

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)
            return

        self.led_detection_result = self.detect_leds(cv_image)

    # organism type from the number of LEDs in the cluster
    def determine_organism_type(self, num_leds):
        if num_leds == 2:
            return "alien_a"
        elif num_leds == 3:
            return "alien_b"
        elif num_leds == 4:
            return "alien_c"
        return "unknown"

    def publish_led_detection_results(self):
        num_leds, centroid_coordinates, areas = self.led_detection_result

        # the route can pass close to a cluster twice, publish it only once
        for x, y in self.found:
            if abs(x - self.drone_position[0]) < 1.5 and abs(y - self.drone_position[1]) < 1.5:
                print("Already published this one, skipping")
                return

        if num_leds > 0 and centroid_coordinates:
            self.found.append((self.drone_position[0], self.drone_position[1]))
            biolocation_msg = Biolocation()
            biolocation_msg.organism_type = self.determine_organism_type(num_leds)
            biolocation_msg.whycon_x = self.drone_position[0]
            biolocation_msg.whycon_y = self.drone_position[1]
            biolocation_msg.whycon_z = self.drone_position[2]

            self.biolocation_pub.publish(biolocation_msg)
            print(f"Published: Type={biolocation_msg.organism_type}, X={biolocation_msg.whycon_x}, Y={biolocation_msg.whycon_y}")

    # move the setpoint a little towards the side of the image the cluster is on.
    # x and y are each locked once the cluster centre is within 5 px of the image middle.
    def step_towards_led(self, z):
        dx = 0.5 if self.center_x >= 250 else -0.5
        dy = 0.5 if self.center_y > 250 else -0.5

        self.hover_x = self.drone_position[0] + self.center_x * 0.001
        self.hover_y = self.drone_position[1] + self.center_y * 0.00079
        self.set_setpoint([self.hover_x + dx, self.hover_y + dy, z])
        self.pid()

        if 245 <= self.center_x <= 255:
            self.final_hover_x = self.drone_position[0]
            self.final_x = True
        if 245 <= self.center_y <= 255:
            self.final_hover_y = self.drone_position[1]
            self.final_y = True

    def center_align(self):
        while self.reachcenter and not rospy.is_shutdown():
            self.pid()
            self.rate.sleep()
            if self.final_x and self.final_y:
                self.reachcenter = False
                break
            self.step_towards_led(28)

    def run(self):
        # no camera frame yet
        if self.led_detection_result is None:
            return

        if not self.led_detected and not self.finished():
            num_leds, centroid_coordinates, _ = self.led_detection_result
            if num_leds > 0 and centroid_coordinates:
                self.led_detected = True
                self.reachcenter = True
                self.moveon = False
                self.center_align()
                if self.final_x and self.final_y:
                    self.set_setpoint([self.final_hover_x, self.final_hover_y, 28])
                    self.pid()
                    self.final_x = False
                    self.final_y = False
                    self.publish_led_detection_results()
                    self.led_detected = False
                    self.moveon = True

            # carry on with the route while the last cluster is still in view
            # so the same one is not picked up twice
            while self.moveon and not rospy.is_shutdown():
                num_leds, centroid_coordinates, _ = self.led_detection_result
                self.pid()
                self.rate.sleep()
                self.run_next_setpoint()
                if 2 <= num_leds <= 4:
                    self.run_next_setpoint()

                if num_leds == 0 or self.finished():
                    break

    def run_next_setpoint(self):
        if self.finished():
            return

        setpoint = self.setpoints[self.current_setpoint]
        error = [setpoint[0] - self.drone_position[0], setpoint[1] - self.drone_position[1], setpoint[2] - self.drone_position[2]]

        if not self.landing:
            if all(abs(e) <= self.abs_err for e in error):
                self.current_setpoint += 1
                if self.current_setpoint < len(self.setpoints):
                    self.set_setpoint(self.setpoints[self.current_setpoint])
                    self.pid()
                else:
                    # search done, go to the research station
                    self.landing = True
                    self.setpoints.append([11, 11, 37])
        else:
            self.abs_err = 0.1
            if all(abs(e) <= self.abs_err for e in error):
                self.current_setpoint += 1
                if self.current_setpoint < len(self.setpoints):
                    self.set_setpoint(self.setpoints[self.current_setpoint])
                else:
                    self.disarm()

if __name__ == '__main__':
    swift_drone = swift()
    while not rospy.is_shutdown():
        swift_drone.run()
        swift_drone.rate.sleep()
