import time
from dronekit import connect, VehicleMode, Command,LocationGlobalRelative,LocationGlobal
from pymavlink import mavutil
import argparse
import cv2
import numpy as np
from threading import Thread
import math
import airsim
import datetime
from depthFuncs import Depth
from threading import Thread
import sys
import torch
"""
sim_vehicle.py -v ArduCopter -f airsim-copter -w --no-mavproxy

python mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:14550 --out udp:127.0.0.1:14551

"""

#----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--connect', default='tcp:127.0.0.1:5762')
args = parser.parse_args()

connection_string = args.connect
vehicle = connect(connection_string,wait_ready=False)

vehicle.mode = VehicleMode("GUIDED")
#----------------------------------------------

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print ("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print (" Waiting for vehicle to initialise...")
        time.sleep(1)

    print ("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print (" Waiting for arming...")
        time.sleep(1)

    time.sleep(2)
    print ("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).

    while True:
        print (" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
            print ("Reached target altitude")
            break
        time.sleep(1)

def send_body_velocity_rate(velocity_f, velocity_r, velocity_down, rate,duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b0000010111000111,  # use only speeds and yaw_rate
        0, 0, 0,  # x, y, z positions (not used)
        velocity_f, velocity_r, velocity_down,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, rate)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(0.007)

def movement():
    print("başladık herhalde")

def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors and
    for the specified duration.

    This uses the SET_POSITION_TARGET_LOCAL_NED command with a type mask enabling only 
    velocity components 
    (http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_local_ned).
    
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version 
    (sending the message multiple times does not cause problems).
    
    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 

    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(0.007)

def display(img):
    pt1 = (240,180)
    pt2 = (400,300)
    # height2=int(frameHeight / 2)
    # width2=int(frameWidth/2)
    # cv2.line(img,(int(frameWidth/2),0),(int(frameWidth/2),frameHeight),(110, 57, 4),2)
    # cv2.line(img, (0, int(frameHeight / 2) ), (frameWidth, int(frameHeight / 2) ), (110, 57, 4), 2)
    for i in range(1,8):
        cv2.line(img,(i*80,0),(i*80,480),(110, 57, 4),2)

    for i in range(1,8):
        cv2.line(img,(0,i*60),(640,i*60),(110, 57, 4),2)

    cv2.rectangle(img, pt1, pt2, (110, 57, 4), 2)


def cam_record(device,midas,transform):
    while True:
        rawImage1 = client.simGetImage("bottom_center", airsim.ImageType.Scene)
        bottom = cv2.imdecode(airsim.string_to_uint8_array(rawImage1), cv2.IMREAD_UNCHANGED)
        rawImage2=client.simGetImage("front_center",airsim.ImageType.Scene)
        front = cv2.imdecode(airsim.string_to_uint8_array(rawImage2), cv2.IMREAD_UNCHANGED)
        detect_letter(bottom)

        img = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        depth_normalized = output / (output.max() /255)
        depth_normalized = depth_normalized.astype(np.uint8)
        _,depth_normalized=cv2.threshold(depth_normalized,150,255,cv2.THRESH_BINARY)
        innerCenterAvg=np.average(depth_normalized[240:400,180:300])
        outerCenterAvg=np.average(depth_normalized[160:480,180:300])
        centerLeftAvg=np.average(depth_normalized[0:240,0:300])
        centerRightAvg=np.average(depth_normalized[400:640,0:300])
        print("*************************")
        print(f"CenterAvg:{innerCenterAvg}")
        print(f"CenterLeftAvg:{centerLeftAvg}")
        print(f"CenterRightAvg:{centerRightAvg}")
        print("************************")
        
        currentLat=vehicle.location._lat
        currentLon=vehicle.location._lon
        targetLat,targetLon=39.872248, 32.731967
        
        latDiff=targetLat-currentLat
        lonDiff=targetLon-currentLon
        latVel=latDiff*1000
        lonVel=lonDiff*1000
        send_ned_velocity(latVel,latDiff,0,1)
        if innerCenterAvg>190:
            send_body_velocity_rate(0,0,0,0,1)
            if centerLeftAvg>centerRightAvg:
                send_body_velocity_rate(0,3,0,0,1)
            else:
                send_body_velocity_rate(0,-3,0,0,1)
        else:
            send_body_velocity_rate(0.3,0,0,0,1)
        display(depth_normalized)


        cv2.imshow("img_copy", depth_normalized)
        cv2.imshow("bottom",bottom)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_letter(img):
    kernel = np.ones((1,1),np.uint8)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgBlur = cv2.GaussianBlur(imgBlur, (7, 7), 1)
    imgBlur = cv2.GaussianBlur(imgBlur, (7, 7), 1)

    # se=cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))

    imgCanny = cv2.Canny(imgBlur,100,150)

    # imgEx =cv2.morphologyEx(imgCanny, cv2.MORPH_DILATE, se)

    imgDilated = cv2.dilate(imgCanny, kernel, iterations=2)

    contours,hierarchy = cv2.findContours(imgDilated,cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        length = int(len(contours))
        area = cv2.contourArea(cnt)
        print(f"the length is :{length}")
        if area > 3000 :
            for i in range(length):
                if  hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1 :
                    contour_position = i
                    peri = cv2.arcLength(contours[contour_position],True)
                    approx = cv2.approxPolyDP(contours[contour_position], 0.01 * peri, True)
                    objCor = len(approx)         
                    print(f"obj cor: {objCor}")

                    if objCor == 12:
                        print("FOUND IT BOIS")
                        x, y, w, h = cv2.boundingRect(contours[contour_position])
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                        cv2.circle(img,(int((x+w/2)),int((y+h/2))),6,(255,0,255),thickness=3)
                        cv2.drawContours(img,contours[contour_position],-1,(255,0,0),2)
                        cv2.putText(img,'Successfull',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1,cv2.LINE_AA)
                        print(contour_position)
                        vehicle.mode=VehicleMode("LAND")
                    else:
                        print("you have smth else as inner contour")
                        cv2.drawContours(img,contours[contour_position],-1,(255,0,0),2)
                else:
                    pass


if __name__=="__main__":
  
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()


    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # letterThread=Thread(target=detect_letter)
    camThread=Thread(target=cam_record,args=(device,midas,transform),daemon=True)
    movementThread=Thread(target=movement,daemon=True)

    arm_and_takeoff(3)
    time.sleep(2)
    camThread.start()
    # letterThread.start()
    movementThread.start()
    time.sleep(180)
    exit()