#!/usr/bin/env python

import os
import sys
import serial
import math, numpy as np
import roslib; roslib.load_manifest('hrl_fabric_based_tactile_sensor')
import hrl_lib.util as ut
#import hrl_fabric_based_tactile_sensor.adc_publisher_node as apn
import rospy
import matplotlib.pyplot as plt

plt.ion()

import time
from scipy.signal import savgol_filter, lfilter, butter
from scipy.interpolate import interp1d
#from  hrl_fabric_based_tactile_sensor.map_thermistor_to_temperature import temperature

fB,fA = butter(2, 0.1, analog=False)
temp_dev_nm = '/dev/cu.teensy.s2673410' # thermal teensy serial number
baudrate = 115200
temp_dev = []

def setup_serial(dev_name, baudrate):
    try:
        serial_dev = serial.Serial(dev_name)
        if(serial_dev is None):
            raise RuntimeError("[%s]: Serial port %s not found!\n" % (rospy.get_name(), dev_name))

        serial_dev.setBaudrate(baudrate)
        serial_dev.setParity('N')
        serial_dev.setStopbits(1)
        serial_dev.write_timeout = .1
        serial_dev.timeout= 1

        serial_dev.flushOutput()
        serial_dev.flushInput()
        return serial_dev

    except serial.serialutil.SerialException as e:
        rospy.logwarn("[%s] Error initializing serial port %s", rospy.get_name(), dev_name)
        return []

def send_string(serial_dev, message):
    try:
        
        serial_dev.write(message) 
        serial_dev.flushOutput()
    except serial.serialutil.SerialException as e:
        print "Error sending string"

def start_heating(temp_dev):
    send_string(temp_dev, 'G')

def stop_heating(temp_dev):
    send_string(temp_dev, 'X')

if __name__ == '__main__':
    while temp_dev == []:
        print "Setting up serial...",
        temp_dev = setup_serial(temp_dev_nm, baudrate)

        time.sleep(.05)
    print "done"
    print ' '

    start_heating(temp_dev)
    time.sleep(5)
    stop_heating(temp_dev)
    time.sleep(1)
    start_heating(temp_dev)
    time.sleep(5)
    stop_heating(temp_dev)
    time.sleep(1)
