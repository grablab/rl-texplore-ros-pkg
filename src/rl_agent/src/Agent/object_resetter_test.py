#! /usr/bin/python

import rospy
import time
from common_msgs_gl.srv import SendInt
from std_srvs.srv import SetBool
from std_msgs.msg import UInt32
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
import auto_data_collector.keyboard_encodings as ENC

from random import *
import numpy as np

initialize_hand_srv_ = rospy.ServiceProxy('/manipulation_manager/set_mode', SendInt)
reset_obj_srv_ = rospy.ServiceProxy('/system_reset/reset_object', SendInt)

fake_keyboard_pub_ = rospy.Publisher('/keyboard_input', UInt32, queue_size=1)

RAISE_CRANE_RESET =0
LOWER_CRANE_OBJECT_READY =1

GET_GRIPPER_READY = 0
ALLOW_GRIPPER_TO_MOVE =1
OPEN_GRIPPER =2

seconds_until_next_reset = 8

class AutoDataCollectorNode():
    def __init__(self):

        # Determine if any modes are occuring
        self.is_dropped_ = False
        self.is_stuck_ = False

        # Count how many of each occured
        self.num_normal_ = 0
        self.num_dropped_ = 0
        self.num_stuck_ = 0
        self.num_sliding = 0

        # This is used to reset the motors if something bad occurs
        self.load_history_ = []
        self.load_history_length_ = 5
        self.reset_to_save_motors = False
        self.single_motor_save_ = []

        self.recent_object_move_dist_ = 50  # Arbritrary value to start



        # These will be enables later
        self.enable = False
        self.enable_fake_keyboard_ = False
        self.master_tic = time.time()
        self.initialized = False

        # Load in random keyboard enocdings
        self.keyboardDict = ENC.Encodings
        self.selection_choices = self.keyboardDict.keys()  # Note the stop key is index 0
        self.current_selection = 1  # arb assignment
        self.time_taken = 45  # This will give us a fresh start
        self.time_to_live = 45  # Determines how many cycles the movement will occur

        rospy.Subscriber("/stuck_drop_detector/object_dropped_msg", Bool, self.itemDroppedCallback)
        rospy.Subscriber("/stuck_drop_detector/object_stuck_msg", Bool, self.itemStuckCallback)
        rospy.Subscriber("/stuck_drop_detector/object_move_dist", Int32, self.objectMoveDist)
        rospy.Subscriber("/gripper/load", Float32MultiArray, self.gripperLoadCallback)

        enable_srv_ = rospy.Service("/auto_data_collector/enable_collection", SetBool, self.enable_data_save)
        print("aaaaaaaaaaaaaaaaaaa")
        time.sleep(2)
        reset_obj_srv_(LOWER_CRANE_OBJECT_READY)

        #fake_keyboard_pub_.publish(self.keyboardDict["KEY_S_"])
        '''
        print("ccccccccccccccccccc")
        time.sleep(5)
        reset_obj_srv_(RAISE_CRANE_RESET)
        print("ddddddddddddddddddd")
        time.sleep(5)
        reset_obj_srv_(LOWER_CRANE_OBJECT_READY)
        print("eeeeeeeeeeeeeeeeee")
        time.sleep(5)
        reset_obj_srv_(RAISE_CRANE_RESET)
        '''

        r = rospy.Rate(30)  # 30hz
        while not rospy.is_shutdown():
            if self.enable == True and self.initialized == True:
                '''
                print("rospy running: self.is_dropped_: {}".format(self.is_dropped_))
                '''
                if self.is_dropped_ == True:
                    #resp = record_hand_srv_(DROP_CASE)
                    print("Dropped Object")
                    rospy.loginfo("Dropped Object")
                    self.resetObject()
                    self.num_dropped_ = self.num_dropped_ + 1

                if self.is_stuck_ == True or self.reset_to_save_motors == True:
                    #resp = record_hand_srv_(STUCK_CASE)
                    print("Got stuck!!!!")
                    rospy.logwarn("Stuck Object")
                    self.resetObject()
                    self.num_stuck_ = self.num_stuck_ + 1
                    self.reset_to_save_motors = False
                '''
                if time.time() - self.master_tic > 10:
                    #resp = record_hand_srv_(NORMAL_CASE)
                    self.master_tic = time.time()
                    rospy.loginfo("Normal Object")
                    self.num_normal_ = self.num_normal_ + 1
                    self.reset_to_save_motors = False
                '''
                if self.enable_fake_keyboard_:
                    self.fake_keyboard_call_()
            r.sleep()


    def fake_keyboard_call_(self):

        if self.time_taken >= self.time_to_live:
            self.time_to_live = randint(5, 25)  # between 1.5-2.5 seconds
            self.current_selection = randint(1, len(self.selection_choices) - 1)
            self.time_taken = 0
            new_value = True
        # elif self.time_taken >= self.time_to_live-2:
        #    self.current_selection = 0 #this will stop for 2 cycles

        fake_keyboard_pub_.publish(self.keyboardDict[self.selection_choices[self.current_selection]])
        print("Keyboard input: {}, is_dropped_: {}".format(self.selection_choices[self.current_selection], self.is_dropped_))
        #fake_keyboard_pub_.publish(self.keyboardDict["KEYX"])
        self.time_taken = self.time_taken + 1

    def enable_data_save(self, req):
        self.enable = req.data
        print("ddddddddddddddddd")
        print("eeeeeee: {}".format(self.enable))
        if self.enable == True:
            print("eeeeeeeeeeeeeee")
            initialize_hand_srv_(GET_GRIPPER_READY)
            print("ffffffffffffff")
            time.sleep(12)
            # time.sleep(12)
            print("eeeeeeeeeeeeeeeffffffffff")
            self.resetObject()
            self.initialized = True

        return [self.enable, "Successfully changed enable bool"]

    def gripperLoadCallback(self, msg):  # We will only keep track of when load history is exceeded
        #print("gripperLoad: {}")
        history = msg.data
        if len(history) > 0:
            # Do the two motor case first
            if len(self.load_history_) >= self.load_history_length_:
                self.load_history_.pop(0)  # Remove the first one

            if history[0] < (-800) and history[1] < (-800):
                self.load_history_.append(1)
            else:
                self.load_history_.append(0)

            # Now do the two motor case
            if len(self.single_motor_save_) >= self.load_history_length_ + 20:
                self.single_motor_save_.pop(0)  # Remove the first

            temp0 = history[0] < (-800)
            temp1 = history[1] < (-800)
            # print("gripperLoad: {}, {}".format(history[0], history[1]))
            if abs(history[0]) > 750 or abs(history[1]) > 750:
                print("ggg load is too much: {}, {}".format(history[0], history[1]))
                self.reset_to_save_motors = True
            else:
                self.reset_to_save_motors = False
            '''
            self.single_motor_save_.append([temp0, temp1])
            arrayed = np.asarray(self.single_motor_save_)

            # First check: Are they both being pulled really hard
            # Second check: Has either been pulled hard for the past while
            if np.mean(self.load_history_) >= 0.9 or np.mean(arrayed[:, 0]) >= 0.9 or np.mean(arrayed[:, 1]) >= 0.9:
                # print 'Load 1: ', np.mean(arrayed[:,0])>=0.9
                # print 'Load2: ', np.mean(arrayed[:,1])>=0.9
                # print 'Both: ', np.mean(self.load_history_) >=0.9
                self.reset_to_save_motors = True
            else:
                self.reset_to_save_motors = False
            '''

    def itemDroppedCallback(self, msg):
        self.is_dropped_ = msg.data
        #print('is_dropped: {}'.format(self.is_dropped))

    def itemStuckCallback(self, msg):
        self.is_stuck_ = msg.data

    def itemSlidingCallback(self, msg):
        self.is_sliding_ = msg.data

    def objectMoveDist(self, msg):
        self.object_move_dist_ = msg.data

    def resetObject(self):
        print("resetting object!!!!!!!!!!!!!!!!!!!!!1")
        time_to_break = False
        self.enable_fake_keyboard_= False
        self.enable = False
        fake_keyboard_pub_.publish(self.keyboardDict["KEY_S_"])
        initialize_hand_srv_(OPEN_GRIPPER) # let the system stop
        time.sleep(8)

        while time_to_break == False:
            resp =  reset_obj_srv_(RAISE_CRANE_RESET)
            time.sleep(5)
            tic = time.time()
            print("self.object_move_dist: {}".format(self.object_move_dist_))
            while self.object_move_dist_ >15:
                toc = time.time()
                time.sleep(0.5)
                if (toc - tic > seconds_until_next_reset):
                    break
            if self.object_move_dist_ <=15:
                time_to_break=True

        #Now we can close the fingers and wait for the grasp
        print("Getting gripper ready")
        initialize_hand_srv_(GET_GRIPPER_READY)
        time.sleep(8)
        print("lowering crane object ready")
        reset_obj_srv_(LOWER_CRANE_OBJECT_READY)
        time.sleep(1.5)
        #time.sleep(10)
        print("allowing gripper to move")
        initialize_hand_srv_(ALLOW_GRIPPER_TO_MOVE)
        self.enable_fake_keyboard_ = True
        self.enable = True
        self.master_tic = time.time()

if __name__ == "__main__":

    rospy.init_node('AutoDataCollectorNode')

    #try:
    AutoDataCollectorNode()
    #except:
    #    rospy.logerr('Could not instantiate AutoCollectorNode')