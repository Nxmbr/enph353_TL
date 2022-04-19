#! /usr/bin/env python

import rospy
import cv2
import numpy as np
import time
import imagehash
from PIL import Image as Image2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float64
from cv_bridge import CvBridge


class RobotController:
  #PID Control Constants
  kP = 0.0075
  kI = 0.0
  kD = 0.015

  # Possible States
  DRIVE = 0
  WAIT_FOR_PED = 1

  #time constants
  SLEEP_TIMER = 1.5

  # Computer Vision Constants
  lower_hsv_road = np.array([0,0,80])
  upper_hsv_road = np.array([0,0,87])
  lower_hsv_ped = np.array([0,46,230])
  upper_hsv_ped = np.array([255,255,255])

  center_x = 640
  
  crosswalk_threshold = 670
  pedestrian_timer = 2

  # Image Delta 
  prev_img = None
  prev_delta = -1


  def __init__(self):
    # Robot State
    self.state = self.DRIVE
    self.current_time = 0
    self.pedestrian_time = 0

    #Publishers and Subscribers
    self.move = Twist()
    self.move.linear.x = 0.35
    self.vel_publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.img_msg = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.img_processor)

    #Bridge for changning ros image messages into openCV images
    self.bridge = CvBridge()

    #PID Constants
    self.prev_error = 1
    self.PID_correction_factor = 7

    #Containers for the Previous centroid so there aren't any artifacts for not seeing the road
    self.prev_centroids = np.ones((2,2))
    self.road_key = 0
    self.ped_key = 1

    time.sleep(2)

  def img_processor(self, data):
    img = self.bridge.imgmsg_to_cv2(data, "passthrough")

    #find the centroid of the road and also check whether there is a crosswalk indicator for us to stop before.
    self.centroid_road = self.get_centroid(img, self.lower_hsv_road, self.upper_hsv_road, self.road_key)
    if(time.time() > self.current_time + self.SLEEP_TIMER + 2):
      self.centroid_ped = self.get_centroid(img, self.lower_hsv_ped, self.upper_hsv_ped, self.ped_key)

    # Check the state of the robot. If it is safe to proceed, do so and check if we are approaching a crosswalk. 
    # If not safe to proceed, wait until the human comes closer and then 
    if(self.state is self.DRIVE):

      if (self.centroid_ped[1] > self.crosswalk_threshold): # If the crosswalk indicator is sufficiently low on the screen, stop and wait until its safe to proceed
        print('We see a crosswalk!')
        self.move.angular.z = 0
        self.move.linear.x = 0
        self.vel_publisher.publish(self.move)
        self.state = self.WAIT_FOR_PED
        self.pedestrian_time = time.time()
      else: # Else proceed to continue driving
        #print('We dont see a crosswalk')
        # PID to control YAW
        self.move.angular.z = self.get_pid(self.centroid_road[0], self.center_x)
        self.move.linear.x = 0.25
        self.vel_publisher.publish(self.move)

        #print('Z Angular Velocity: ', self.move.angular.z)

    elif (self.state is self.WAIT_FOR_PED):
      delta = self.get_delta(img)
      print('delta is ' + str(delta))

      if (delta > 40 and time.time() > self.pedestrian_time + self.SLEEP_TIMER + 1):
        self.current_time = time.time()
        self.state = self.DRIVE
        self.move.linear.x = 0.15
        self.vel_publisher.publish(self.move)
        self.centroid_ped = [1,1]

      self.prev_delta = delta
    self.prev_img = img

  def get_pid(self, target, actual):
    # used the Wikipedia PID page to help write this function
    error = target - actual
    p = self.kP * error
    i = self.kI * (error + self.prev_error)
    d = self.kD * (error - self.prev_error)
    self.prev_error = error
    return -(p + i + d)

  def get_centroid(self, data, lower_hsv, upper_hsv, centroid_key):
    mask, img_masked = self.threshold_hsv(data, lower_hsv, upper_hsv)

    if centroid_key is 1:
      density = cv2.countNonZero(mask)
      #print('the densit is ' + str(density))
      if density < 5000 :
        return [1,1]

    # Crop the image
    crop_mask = mask[500:720,:]
            
    # Calculate moments of binary image
    M = cv2.moments(crop_mask[:,:])
    #print(M)

    if (M["m00"] < 0.000000000000000000000000001):
      cX = int(self.prev_centroids[centroid_key][0])
      cY = int(self.prev_centroids[centroid_key][1])
    else: 
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"]) + 500
      
    # Storing the previous centroid so that if we lose the road the we can just use the previous one
    self.prev_centroids[centroid_key] = [cX,cY]
    return [cX, cY]

  def threshold_hsv(self, img, lower_hsv, upper_hsv):
      img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
      mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
      img_masked = cv2.bitwise_and(img, img, mask=mask)
      return mask, img_masked

  def get_delta(self, curr_img):
    delta_curr = 0

    if self.prev_img is None:
      self.prev_img = curr_img
    
    delta_curr = self.get_imagehash(curr_img)

    return delta_curr

  def get_imagehash(self, current_image):
    hash1 = imagehash.average_hash(self.convert_cv2pil(current_image[200:700,300:1000]), hash_size = 64)
    hash2 = imagehash.average_hash(self.convert_cv2pil(self.prev_img[200:700,300:1000]), hash_size = 64)
    delta = hash1 - hash2
    return delta

  def convert_cv2pil(self, img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image2.fromarray(img2)
    return img_pil

  #other functions...

if __name__ == '__main__':
  rospy.init_node('robot_controller', anonymous=True)
  rc = RobotController()
  rospy.spin()