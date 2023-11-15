#!/usr/bin/env python3

"""
# Team ID:          < 3811 >
# Theme:            < Luminosity Drone >
# Filename:         < LD_3811_life_form_detector.py >
# Functions:        < __init__, disarm, arm, whycon_callback, set_setpoint, altitude_set_pid, pitch_set_pid, roll_set_pid, pid, detect_leds, image_callback, convert_image_msg_to_cv, publish_led_detection_results, determine_organism_type, reached, drone_hover, run, run_next_setpoint >
# Global variables: < drone_position, setpoints, current_setpoint, cmd, Kp, Ki, Kd, error, prev_error, differential_error, sum_error, min_values, max_values, command_pub, throttleError, pitchError, rollError, biolocation_pub, arm, disarm, whycon_callback, set_setpoint, altitude_set_pid, pitch_set_pid, roll_set_pid, pid, detect_leds, image_callback, convert_image_msg_to_cv, publish_led_detection_results, determine_organism_type, reached, drone_hover, run, run_next_setpoint >


Purpose:
---

This python file runs a ROS-node of name drone_control which holds the position of Swift-Drone on the given dummy.
This node publishes and subscribes the following topics:

    PUBLICATIONS                SUBSCRIPTIONS
    /drone_command              /whycon/poses
    /alt_error                  /pid_tuning_altitude
    /pitch_error                /pid_tuning_pitch
    /roll_error                 /pid_tuning_roll
    /astrobiolocation           /swift/camera_rgb/image_raw

This node uses the following algorithms:

    1. PID algorithm for controlling the drone.
    2. Image processing algorithm for detecting the LEDs.
    3. Image processing algorithm for detecting the centroid of the LEDs.
    4. Image processing algorithm for detecting the area of the LEDs.
    5. Image processing algorithm for detecting the type of organism based on the number of LEDs detected.
    6. Image processing algorithm for detecting the final position of the drone over the detected LED.

This node uses the following packages:

    1. cv_bridge
    2. imutils
    3. numpy
    4. rospy
    5. sensor_msgs
    6. swift_msgs
    7. geometry_msgs
    8. std_msgs
    9. pid_tune
    10. luminosity_drone

This node uses the following messages:

    1. swift_msgs.msg
    2. geometry_msgs.msg
    3. std_msgs.msg
    4. pid_tune.msg
    5. luminosity_drone.msg

This node performs the following tasks:

    1. Initialize the class variables and set the PID constants.
    2. Arm the drone.
    3. Disarm the drone.
    4. Callback function for /whycon/poses topic. Update the drone position.
    5. Set the setpoint for the drone.
    6. Set the PID constants for altitude control.
    7. Set the PID constants for pitch control.
    8. Set the PID constants for roll control.
    9. Implement the PID algorithm for the drone.
    10. Detect the LEDs in the image and return the number of LEDs detected, centroid coordinates and areas of the LEDs.
    11. Callback function for /swift/camera_rgb/image_raw topic. Detect the LEDs in the image and publish the results.
    12. Convert the image message to a numpy array.
    13. Publish the results of LED detection.
    14. Determine the type of organism based on the number of LEDs detected.
    15. Check if the drone has reached the setpoint and update the setpoints if it has.
    16. Hover the drone over the detected LED.
    17. Run the drone control node.
    18. Move to the next setpoint if the current setpoint has been reached.

"""

import rospy
from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
from luminosity_drone.msg import Biolocation
from sensor_msgs.msg import Image
import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils
from cv_bridge import CvBridge, CvBridgeError


class swift:
    """docstring for swift"""
    def __init__(self):
        """
        Purpose:
        ---
        Initialize the class variables and set the PID constants.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        swift_drone = swift()
        """
        rospy.init_node('drone_control')
        self.drone_position = [0.0,0.0,0.0]
        self.led_detected = False
        self.reached_setpoint = False
        self.led_published = False
        self.publish = False
        self.final_x = False
        self.final_y = False
        self.dronehover_detected = False
        self.executable = False

        self.center_x = 0
        self.center_y = 0
        self.center = [0.0,0.0]
        self.hover = False
        self.hover_x = 0
        self.hover_y = 0
        self.abs_err = 1.5

        self.setpoints = [
            [-8, -8, 25],
            [-8, -4, 25],
            [-8, 0, 25],
            [-8, 4, 25],
            [-8, 8, 25],

            [-4, 8, 25],
            [-4, 4, 25],
            [-4, 0, 25],
            [-4, -4, 25],
            [-4, -8, 25],

            [0, -8, 25],
            [0, -4, 25],
            [0, 0, 25],
            [0, 4, 25],
            [0, 8, 25],

            [4, 8, 25],
            [4, 4, 25],
            [4, 0, 25],
            [4, -4, 25],
            [4, -8, 25],

            [8, -8, 25],
            [8, -4, 25],
            [8, 0, 25],
            [8, 4, 25],
            [8, 8, 25],
        ]

        self.current_setpoint = 0

        self.led_detection_result = None

        self.cmd = swift_msgs()
        self.cmd.rcRoll = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcThrottle = 1500
        self.cmd.rcAUX1 = 1500
        self.cmd.rcAUX2 = 1500
        self.cmd.rcAUX3 = 1500
        self.cmd.rcAUX4 = 1500

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

        self.arm()

    def disarm(self):
        """
        Purpose:
        ---
        Disarm the drone.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.disarm()
        """
        self.cmd.rcAUX4 = 1100
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)

    def arm(self):
        """
        Purpose:
        ---
        Arm the drone.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.arm()
        """
        self.disarm()
        self.cmd.rcRoll = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcThrottle = 1000
        self.cmd.rcAUX4 = 1500
        self.command_pub.publish(self.cmd) # Publishing /drone_command
        rospy.sleep(1)

    def whycon_callback(self, msg):
        """
        Purpose:
        ---
        Callback function for /whycon/poses topic. Update the drone position.

        Input Arguments:
        ---
        `msg` :  [ PoseArray ]

        Returns:
        ---
        None

        Example Call:
        ---
        self.whycon_callback(msg)
        """
        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z

                    
    def set_setpoint(self, setpoint):
        """
        Purpose:
        ---
        Set the setpoint for the drone.

        Input Arguments:
        ---
        setpoint :  [ list ]

        Returns:
        ---
        None

        Example Call:
        ---
        self.set_setpoint(setpoint)
        """
        self.setpoints[self.current_setpoint] = setpoint
        self.sum_error = [0, 0, 0]
        self.prev_error = [0, 0, 0]

    def altitude_set_pid(self, alt):
        """
        Purpose:
        ---
        Set the PID constants for altitude control.

        Input Arguments:
        ---
        alt :  [ Float64 ]

        Returns:
        ---
        None

        Example Call:
        ---
        self.altitude_set_pid(alt)
        """
        self.Kp[2] = alt.Kp * 0.06
        self.Ki[2] = alt.Ki * 0.00008
        self.Kd[2] = alt.Kd * 0.3

    def pitch_set_pid(self, pitch):
        """
        Purpose:
        ---
        Set the PID constants for pitch control.

        Input Arguments:
        ---
        pitch :  [ Float64 ]

        Returns:
        ---
        None

        Example Call:
        ---
        self.pitch_set_pid(pitch)
        """
        self.Kp[0] = pitch.Kp * 0.06
        self.Ki[0] = pitch.Ki * 0.00008
        self.Kd[0] = pitch.Kd * 0.3

    def roll_set_pid(self, roll):
        """
        Purpose:
        ---
        Set the PID constants for roll control.

        Input Arguments:
        ---
        `roll` :  [ Float64 ]

        Returns:
        ---
        None

        Example Call:
        ---
        self.roll_set_pid(roll)
        """
        self.Kp[1] = roll.Kp * 0.06
        self.Ki[1] = roll.Ki * 0.00008
        self.Kd[1] = roll.Kd * 0.3

    def pid(self):
        """
        Purpose:
        ---
        Implement the PID algorithm for the drone.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.pid()
        """
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

    def detect_leds(self, image):
        """
        Purpose:
        ---
        Detect the LEDs in the image and return the number of LEDs detected, centroid coordinates and areas of the LEDs.

        Input Arguments:
        ---
        `image` :  [ numpy array ]
            The image in which the LEDs are to be detected.

        Returns:
        ---
        `num_leds` :  [ int ]
            The number of LEDs detected in the image.
        `centroid` :  [ list ]
            The centroid coordinates of the LEDs detected in the image.
        `areas` :  [ list ]
            The areas of the LEDs detected in the image.

        Example Call:
        ---
        self.detect_leds(image)
        """
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

        for i, c in enumerate(cnts):

                area = cv2.contourArea(c)

                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = (M["m10"] / M["m00"])
                    cY = (M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                centroid.append([cX, cY])
                areas.append(area)
                center = [0.0, 0.0]
                for sublist in centroid:
                    for i, value in enumerate(sublist):
                        center[i] += value

                # Calculate the average by dividing the sums by the number of sublists
                average_values = [total / len(centroid) for total in center]
                self.center_x = average_values[0]
                self.center_y = average_values[1]
                # print("Average Values:", average_values)
                
        return len(cnts), centroid, areas

    def image_callback(self, msg):
        """
        Purpose:
        ---
        Callback function for /swift/camera_rgb/image_raw topic. Detect the LEDs in the image and publish the results.

        Input Arguments:
        ---
        `msg` :  [ Image ]
            The image message from the topic.

        Returns:
        ---
        None

        Example Call:
        ---
        self.image_callback(msg)
        """
        bridge = CvBridge()

        try:
            cv_image = self.convert_image_msg_to_cv(msg)
        except CvBridgeError as e:
            print(e)

        num_leds, centroid_coordinates, areas = self.detect_leds(cv_image)
        self.led_detection_result = (num_leds, centroid_coordinates, areas)

    def convert_image_msg_to_cv(self, image_msg):
        """
        Purpose:
        ---
        Convert the image message to a numpy array.

        Input Arguments:
        ---
        image_msg :  [ Image ]

        Returns:
        ---
        cv_image :  [ numpy array ]

        Example Call:
        ---
        self.convert_image_msg_to_cv(image_msg)
        """
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

    def publish_led_detection_results(self):
        """
        Purpose:
        ---
        Publish the results of LED detection.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.publish_led_detection_results()
        """
        if not self.led_published and (self.led_detected or self.current_setpoint == len(self.setpoints)):
            num_leds, centroid_coordinates, areas = self.led_detection_result

            if num_leds > 0 and centroid_coordinates:
                biolocation_msg = Biolocation()
                biolocation_msg.organism_type = self.determine_organism_type(num_leds)
                biolocation_msg.whycon_x = self.drone_position[0]
                biolocation_msg.whycon_y = self.drone_position[1]
                biolocation_msg.whycon_z = self.drone_position[2]

                self.biolocation_pub.publish(biolocation_msg)
                print(f"Published LED detection results: Type={biolocation_msg.organism_type}, X={biolocation_msg.whycon_x}, Y={biolocation_msg.whycon_y}")

                self.led_detected = False
                self.led_published = True  
                if self.current_setpoint == len(self.setpoints):
                    self.abs_err = 0.1
                    self.setpoints = [
                        [6.5, 6.5, 25],
                        [10, 10, 30],
                        [11, 11, 37.5]
                    ]

    def determine_organism_type(self, num_leds):
        """
        Purpose:
        ---
        Determine the type of organism based on the number of LEDs detected.

        Input Arguments:
        ---
        `num_leds` :  [ int ]

        Returns:
        ---
        `organism_type` :  [ string ]

        Example Call:
        ---
        self.determine_organism_type(num_leds)
        """
        organism_type=""
        if num_leds == 2:
            organism_type = "alien_a"
            return organism_type
        elif num_leds == 3:
            organism_type="alien_b"
            return organism_type
        elif num_leds == 4:
            organism_type="alien_c"
            return organism_type
        else:
            organism_type="unknown"
            return organism_type

    def reached(self):
        """
        Purpose:
        ---
        Check if the drone has reached the setpoint and update the setpoints if it has.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.reached()
        """
        if self.reached_setpoint:
            self.abs_err = 0.1
            new_setpoints = [
                [6.5, 6.5, 25],
                [10, 10, 30],
                [11, 11, 37.5]
            ]

            # Update setpoints
            self.setpoints = new_setpoints

            # Reset flags or perform any other necessary actions
            self.reached_setpoint = False

    def drone_hover(self): 
        """
        Purpose:
        ---
        Hover the drone over the detected LED.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.drone_hover()
        """
        while self.dronehover_detected:
            self.pid()
            r.sleep()
            if self.publish:
                if (self.final_x and self.final_y):
                    self.set_setpoint([self.final_x, self.final_y, 25])
                    print("Reached Centroid")
                    self.pid()
                    self.publish_led_detection_results()
                    self.publish = False 
                    self.dronehover_detected = False
                    self.executable = True
                    break
                else:
                    if(250<=self.center_x<=500 and 0<=self.center_y<=250): #1st quadrant
                        scale_factor_x = 0.001
                        scale_factor_y = 0.00079
                        adjusted_x = self.center_x * scale_factor_x
                        adjusted_y = self.center_y * scale_factor_y

                        self.hover_x = self.drone_position[0] + adjusted_x
                        self.hover_y = self.drone_position[1] + adjusted_y
                        self.set_setpoint([self.hover_x+0.5, self.hover_y-0.5, 25])
                        self.pid()
                        print(f"Drone Position: X={self.drone_position[0]}, Y={self.drone_position[1]}")
                        print(f"Adjusted Drone Position: X={self.hover_x}, Y={self.hover_y}")
                        print("Center X:",self.center_x,"Center Y",self.center_y)
                        if(245<=self.center_x<=255):
                            self.final_hover_x = self.drone_position[0]
                            print("final_hover_x:",self.final_hover_x)
                            self.final_x = True
                        elif(245<=self.center_y<=255):
                            self.final_hover_y = self.drone_position[1]
                            print("final_hover_y:",self.final_hover_y)
                            self.final_y = True
                    elif(0<=self.center_x<=250 and 0<=self.center_y<=250): #2nd quadrant
                        scale_factor_x = 0.001
                        scale_factor_y = 0.00079
                        adjusted_x = self.center_x * scale_factor_x
                        adjusted_y = self.center_y * scale_factor_y

                        self.hover_x = self.drone_position[0] + adjusted_x
                        self.hover_y = self.drone_position[1] + adjusted_y
                        self.set_setpoint([self.hover_x-0.5, self.hover_y-0.5, 25])
                        self.pid()
                        print(f"Drone Position: X={self.drone_position[0]}, Y={self.drone_position[1]}")
                        print(f"Adjusted Drone Position: X={self.hover_x}, Y={self.hover_y}")
                        print("Center X:",self.center_x,"Center Y",self.center_y)
                        if(245<=self.center_x<=255):
                            self.final_hover_x = self.drone_position[0]
                            print("final_hover_x:",self.final_hover_x)
                            self.final_x = True
                        elif(245<=self.center_y<=255):
                            self.final_hover_y = self.drone_position[1]
                            print("final_hover_y:",self.final_hover_y)
                            self.final_y = True
                    elif(0<=self.center_x<=250 and 250<=self.center_y<=500): #3rd quadrant
                        scale_factor_x = 0.001
                        scale_factor_y = 0.00079
                        adjusted_x = self.center_x * scale_factor_x
                        adjusted_y = self.center_y * scale_factor_y

                        self.hover_x = self.drone_position[0] + adjusted_x
                        self.hover_y = self.drone_position[1] + adjusted_y
                        self.set_setpoint([self.hover_x-0.5, self.hover_y+0.5, 25])
                        self.pid()
                        print(f"Drone Position: X={self.drone_position[0]}, Y={self.drone_position[1]}")
                        print(f"Adjusted Drone Position: X={self.hover_x}, Y={self.hover_y}")
                        print("Center X:",self.center_x,"Center Y",self.center_y)
                        if(245<=self.center_x<=255):
                            self.final_hover_x = self.drone_position[0]
                            print("final_hover_x:",self.final_hover_x)
                            self.final_x = True
                        elif(245<=self.center_y<=255):
                            self.final_hover_y = self.drone_position[1]
                            print("final_hover_y:",self.final_hover_y)
                            self.final_y = True
                    elif(250<=self.center_x<=500 and 250<=self.center_y<=500): #4th quadrant
                        scale_factor_x = 0.001
                        scale_factor_y = 0.00079
                        adjusted_x = self.center_x * scale_factor_x
                        adjusted_y = self.center_y * scale_factor_y

                        self.hover_x = self.drone_position[0] + adjusted_x
                        self.hover_y = self.drone_position[1] + adjusted_y
                        self.set_setpoint([self.hover_x+0.5, self.hover_y+0.5, 25])
                        self.pid()
                        print(f"Drone Position: X={self.drone_position[0]}, Y={self.drone_position[1]}")
                        print(f"Adjusted Drone Position: X={self.hover_x}, Y={self.hover_y}")
                        print("Center X:",self.center_x,"Center Y",self.center_y)
                        if(245<=self.center_x<=255):
                            self.final_hover_x = self.drone_position[0]
                            print("final_hover_x:",self.final_hover_x)
                            self.final_x = True
                        elif(245<=self.center_y<=255):
                            self.final_hover_y = self.drone_position[1]
                            print("final_hover_y:",self.final_hover_y)
                            self.final_y = True
            else:
                break
        
    def run(self):
        """
        Purpose:
        ---
        Run the drone control node.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.run()
        """
        self.led_detected = False
        if not self.led_detected and (self.current_setpoint < len(self.setpoints)) and not self.executable :
            num_leds, centroid_coordinates, _ = self.led_detection_result
            if num_leds > 0 and centroid_coordinates:
                self.abs_err = 0.15
                self.led_detected = True
                self.dronehover_detected = True
                self.publish = True
                self.drone_hover()
                self.reached_setpoint = True
                self.dronehover_detected = False
                print("Reached")
                self.current_setpoint = 0 

        self.reached()

        if not self.reached_setpoint and self.current_setpoint < len(self.setpoints):
            self.pid()
            self.run_next_setpoint()

    def run_next_setpoint(self):
        """
        Purpose:
        ---
        Move to the next setpoint if the current setpoint has been reached.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example Call:
        ---
        self.run_next_setpoint()
        """
        if self.current_setpoint < len(self.setpoints):
            setpoint = self.setpoints[self.current_setpoint]
            error = [setpoint[0] - self.drone_position[0], setpoint[1] - self.drone_position[1], setpoint[2] - self.drone_position[2]]

            if all(abs(e) <= self.abs_err for e in error):
                # Waypoint is considered successful
                self.current_setpoint += 1
                if self.current_setpoint < len(self.setpoints):
                    # Move to the next setpoint
                    print(f"Reached setpoint nxt {self.current_setpoint}: {setpoint}")
                    self.set_setpoint(self.setpoints[self.current_setpoint])

                else:
                    print(f"Reached setpoint nxt {self.current_setpoint}: {setpoint}")
                    self.cmd.rcAUX4 = 1100
                    self.command_pub.publish(self.cmd)
                    self.disarm()
                    print("All waypoints reached. nxt")

if __name__ == '__main__':
    """
    Purpose:
    ---
    Initialize the swift class and run the node.
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    
    Example Call:
    ---
    swift_drone = swift()
    """
    swift_drone = swift()
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        swift_drone.run()
        r.sleep()

