#! /usr/bin/env python

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float64
from cv_bridge import CvBridge

class RobotController:
    
  def __init__(self):
    self.move = Twist()
    self.move.linear.x = 3.0
    self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    self.img_msg = rospy.Subscriber("/robot/camera/image_raw", Image, self.img_processor)
    self.bridge = CvBridge()
    self.kP = 0.02
    self.kI = 0.0
    self.kD = 0.04
    self.prev_error = 1
    self.prev_centroid = np.ones(2)
    
  def get_centroid(self, data):
    gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    crop_color = color[450:800,:]

    threshold = 80
    _, crop_threshold = cv2.threshold(crop_color, threshold, 255, cv2.THRESH_BINARY_INV)

    M = cv2.moments(crop_threshold[:,:,0])

    if (M["m00"] < 0.000000000000000000000000001):
      cX = int(self.prev_centroid[0])
      cY = int(self.prev_centroid[1])
    else: 
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"]) + 450

    # Storing the previous centroid so that if we lose the road the we can just use the previous one
    self.prev_centroid = [cX,cY]
    return [cX, cY]

  def img_processor(self, data):
    img = self.bridge.imgmsg_to_cv2(data, "passthrough")
    #find the centroid of img
    self.centroid = self.get_centroid(img)

    #find the center of the image:
    self.center_x = 400

    # PID to control YAW
    self.move.angular.z = self.get_pid(self.centroid[0], self.center_x)
    print(self.move.angular.z)
    self.pub.publish(self.move)

  def get_pid(self, target, actual):
    # used the Wikipedia PID page to help write this function
    error = target - actual
    p = self.kP * error
    i = self.kI * (error + self.prev_error)
    d = self.kD * (error - self.prev_error)
    self.prev_error = error
    return -(p + i + d)
  #other functions...

if __name__ == '__main__':
  rospy.init_node('robot_controller', anonymous=True)
  rc = RobotController()
  rospy.spin()