# -*- coding: utf8 -*-
import cv2
import numpy as np
import time

def mouse_callback(event, x, y, flags, param) :
    print("마우스 위치 X : ", x, " Y : ", y) #이벤트 발생한 마우스 위치 출력

cap = cv2.VideoCapture("C:/Users/LISA/Desktop/rec1.avi")

while(1) :
    ret, frame = cap.read()

    resized_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('resized_frame', resized_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    # cv2.setMouseCallback('resized_frame', mouse_callback)
   
    key = cv2.waitKey(30)
    if key == 27 : #esc
        print("ESC")
        break
    # if key == 26 : #ctrl + z
    #     break

cv2.destroyAllWindows()
cap.release()