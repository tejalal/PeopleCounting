import numpy as np
import cv2
import Helper
import time
import argparse
import datetime
import sys
import urllib.request

status = True
cnt_up = 0
cnt_down = 0
w = 1280
h = 720
frameArea = h*w
areaTH = frameArea/250
print('Area Threshold: ', areaTH)

# Entry / exit lines
line_up = int(2*(h/5)-75)
line_down = int(3*(h/5)+75)

up_limit = int(1*(h/5))
down_limit = int(4*(h/5))

print("Red line :", str(line_down))
print("Blue line :", str(line_up))

line_down_color = (255, 0, 0)  # Blue color line
line_up_color = (0, 0, 255)  # Red color line

pt1 = [0, line_down]
pt2 = [w, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))

pt3 = [0, line_up]
pt4 = [w, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [w, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt7 = [0, down_limit]
pt8 = [w, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

# Structuring elements for morphographic filters
kernelOp = np.ones((3, 3), np.uint8)
kernelOp2 = np.ones((5, 5), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

url = 'http://10.12.8.246:8080/shot.jpg'  # Mobile camera URL
ret = True
while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)

    for i in persons:
        i.age_one()

    # Apply background subtraction
    fgmask2 = fgbg.apply(frame)
    # eliminate shadows
    try:
        ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print('UP:', cnt_up)
        print('DOWN:', cnt_down)
        break

    #  Contours
    contours0, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:

            #  Tracking

            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit, down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        # the object is close to one that has already been detected before
                        new = False
                        i.updateCoords(cx, cy)   # update coordinates in the object and resets age
                        if i.going_UP(line_down, line_up) == True:
                            cnt_up += 1;
                            print("ID:", i.getId(), 'crossed, going out at', time.strftime("%c"))
                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1;
                            print("ID:", i.getId(), 'crossed, coming in at', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        # remove persons from the list
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # free the memory
                if new == True:
                    p = Helper.MyPerson(pid, cx, cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            #   DRAWINGS

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # END for cnt in contours0

    # Draw tracking line
    for i in persons:
        if status:  # run only when status is True
            if len(i.getTracks()) >= 2:
                pts = np.array(i.getTracks(), np.int32)
                pts = pts.reshape((-1, 1, 2))
                frame = cv2.polylines(frame, [pts], False, i.getRGB())
        if i.getId() == 9:
            print(str(i.getX()), ',', str(i.getY()))

    # display  all the info
    str_up = 'Outgoing: ' + str(cnt_up)
    cv2.line(frame, (10, 10), (10, 30), (255, 0, 0), 2)
    cv2.line(frame, (10, 10), (5, 20), (255, 0, 0), 2)
    cv2.line(frame, (10, 10), (15, 20), (255, 0, 0), 2)

    str_down = 'Incoming: ' + str(cnt_down)
    cv2.line(frame, (10, 35), (10, 55), (0, 0, 255), 2)
    cv2.line(frame, (10, 55), (5, 45), (0, 0, 255), 2)
    cv2.line(frame, (10, 55), (15, 45), (0, 0, 255), 2)

    str_diff = 'Difference: ' + str(int(cnt_down)-int(cnt_up))

    frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=1)
    frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=1)
    #white line
    frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

    cv2.putText(frame, str_up, (20, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_down, (20, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    cv2.imshow('Live Streaming', frame)  # display Live Streaming
    cv2.imshow('Masked Video', mask2)  # display B & W video
    
    # press ESC to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# End of while(cap.isOpened())

# close all windows
cv2.destroyAllWindows()
