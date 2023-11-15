#!/usr/bin/env python3

import rospy
from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
from pid_tune.msg import PidTune

class SwiftDrone:
    def __init__(self):
        rospy.init_node('drone_control')
        self.drone_position = [0.0, 0.0, 0.0]
        self.setpoints = [
            [0, 0, 23],
            [2, 0, 23],
            [2, 2, 23],
            [2, 2, 25],
            [-5, 2, 25],
            [-5, -3, 25],
            [-5, -3, 21],
            [7, -3, 21],
            [7, 0, 21],
            [0, 0, 19]
        ]
        self.current_setpoint = 0

        self.cmd = swift_msgs()
        self.cmd.rcRoll = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcThrottle = 1500
        self.cmd.rcAUX1 = 1500
        self.cmd.rcAUX2 = 1500
        self.cmd.rcAUX3 = 1500
        self.cmd.rcAUX4 = 1500

        self.Kp = [93,87.78,90.18]
        self.Ki = [0.00176,0.00088,0.008]	
        self.Kd = [753.3,818,780.3]

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

        rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
        rospy.Subscriber('/pid_tuning_altitude', PidTune, self.altitude_set_pid)
        rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)
        rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)

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

        if self.current_setpoint < len(self.setpoints):
            setpoint = self.setpoints[self.current_setpoint]
            error = [setpoint[0] - self.drone_position[0], setpoint[1] - self.drone_position[1], setpoint[2] - self.drone_position[2]]

            if all(abs(e) <= 0.2 for e in error):
                # Waypoint is considered successful
                self.current_setpoint += 1
                if self.current_setpoint < len(self.setpoints):
                    # Move to the next setpoint
                    print(f"Reached setpoint {self.current_setpoint}: {setpoint}")
                    self.set_setpoint(self.setpoints[self.current_setpoint])
                   
                else:
                    print(f"Reached setpoint {self.current_setpoint}: {setpoint}")
                    print("All waypoints reached.")
                    self.current_setpoint = len(self.setpoints)-1
                    self.set_setpoint(self.setpoints[-1])  # Hover at the last setpoint or perform other actions

    def set_setpoint(self, setpoint):
        self.setpoints[self.current_setpoint] = setpoint
        self.sum_error = [0, 0, 0]
        self.prev_error = [0, 0, 0]

    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.Kp * 0.06
        self.Ki[2] = alt.Ki * 0.00008
        self.Kd[2] = alt.Kd * 0.3

    def pitch_set_pid(self, pitch):
        self.Kp[0] = pitch.Kp * 0.06
        self.Ki[0] = pitch.Ki * 0.00008
        self.Kd[0] = pitch.Kd * 0.3

    def roll_set_pid(self, roll):
        self.Kp[1] = roll.Kp * 0.06
        self.Ki[1] = roll.Ki * 0.00008
        self.Kd[1] = roll.Kd * 0.3

    def pid(self):
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

if __name__ == '__main__':
    swift_drone = SwiftDrone()
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        swift_drone.pid()
        r.sleep()
