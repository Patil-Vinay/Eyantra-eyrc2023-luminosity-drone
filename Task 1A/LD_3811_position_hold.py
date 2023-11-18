#!/usr/bin/env python3

# Team ID:     3811
# Theme:       Luminosity Drone
# Filename:    LD_3811_position_hold.py

# ROS node drone_control: holds the Swift drone at a fixed whycon position.
# Publishes /drone_command, /alt_error, /pitch_error, /roll_error
# Subscribes /whycon/poses, /pid_tuning_altitude, /pid_tuning_pitch, /pid_tuning_roll

from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy


class swift():
	def __init__(self):
		rospy.init_node('drone_control')

		# current position [x, y, z], updated in whycon_callback
		self.drone_position = [0.0,0.0,0.0]

		# position to hold [x, y, z]
		self.setpoint = [2,2,20]

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

		self.error = [0,0,0]
		self.prev_error = [0,0,0]
		self.differential_error = [0,0,0]
		self.sum_error = [0,0,0]

		self.min_values = [1000,1000,1000]
		self.max_values = [2000,2000,2000]

		self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
		self.throttleError = rospy.Publisher('/alt_error',Float64, queue_size=1)
		self.pitchError = rospy.Publisher('/pitch_error', Float64, queue_size=1)
		self.rollError = rospy.Publisher('/roll_error', Float64, queue_size=1)

		rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
		rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
		rospy.Subscriber('/pid_tuning_roll',PidTune,self.roll_set_pid)

		self.arm()

	def disarm(self):
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)

	# disarm first and then arm
	def arm(self):
		self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1500
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)

	def whycon_callback(self,msg):
		self.drone_position[0] = msg.poses[0].position.x
		self.drone_position[1] = msg.poses[0].position.y
		self.drone_position[2] = msg.poses[0].position.z

	# callbacks for the pid tuner sliders
	def altitude_set_pid(self,alt):
		self.Kp[2] = alt.Kp * 0.06
		self.Ki[2] = alt.Ki * 0.0008
		self.Kd[2] = alt.Kd * 0.3

	def pitch_set_pid(self,pitch):
		self.Kp[1] = pitch.Kp * 0.06
		self.Ki[1] = pitch.Ki * 0.0008
		self.Kd[1] = pitch.Kd * 0.3

	def roll_set_pid(self,roll):
		self.Kp[0] = roll.Kp * 0.06
		self.Ki[0] = roll.Ki * 0.0008
		self.Kd[0] = roll.Kd * 0.3

	def pid(self):
		# x error drives roll, y drives pitch, z drives throttle
		self.error[0]=-(self.drone_position[0] - self.setpoint[0])
		self.error[1]= (self.drone_position[1] - self.setpoint[1])
		self.error[2]= (self.drone_position[2] - self.setpoint[2])

		self.differential_error[0] = self.error[0] - self.prev_error[0]
		self.differential_error[1] = self.error[1] - self.prev_error[1]
		self.differential_error[2] = self.error[2] - self.prev_error[2]

		self.sum_error[0] = self.sum_error[0] + self.error[0]
		self.sum_error[1] = self.sum_error[1] + self.error[1]
		self.sum_error[2] = self.sum_error[2] + self.error[2]

		# 1560 is roughly the hover throttle
		self.cmd.rcRoll = int(1500 + self.error[0]*self.Kp[0] + self.differential_error[0]*self.Kd[0]+ self.sum_error[0]*self.Ki[0])
		self.cmd.rcPitch = int(1500 + self.error[1]*self.Kp[1] + self.differential_error[1]*self.Kd[1] + self.sum_error[1]*self.Ki[1])
		self.cmd.rcThrottle = int(1560 + self.error[2]*self.Kp[2] + self.differential_error[2]*self.Kd[2] + self.sum_error[2]*self.Ki[2])

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

		self.prev_error[0]=self.error[0]
		self.prev_error[1]=self.error[1]
		self.prev_error[2]=self.error[2]

		self.rollError.publish(self.error[0])
		self.pitchError.publish(self.error[1])
		self.throttleError.publish(self.error[2])

if __name__ == '__main__':

	swift_drone = swift()
	r = rospy.Rate(30) # pid runs every 33 ms
	while not rospy.is_shutdown():
		swift_drone.pid()
		r.sleep()
