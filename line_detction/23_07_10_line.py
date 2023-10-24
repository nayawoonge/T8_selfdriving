# -*- coding: utf-8 -*-
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import datetime

#표시할 비디오의 출력 형식과 크기
dispW=1280
dispH=720

#비디오 프레임의 방향
flip=2
camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=20/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv2.VideoCapture(camSet)

record = False

def make_cordinates(image, line_parameters) :
  slope, intercept = line_parameters
  print(image.shape)
  y1 = image.shape[0]
  y2 = int(y1*(3/5))
  x1 = int((y1-intercept)/slope)
  x2 = int((y2-intercept)/slope)
  return np.array([x1, y1, x2, y2])
  
def averae_slope_intercept(image, lines) :
  left_fit = []
  right_fit = []
  for line in lines :
    x1, y1, x2, y2 = line.reshape(4)
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    slope = parameters[0]
    intercept = parameters[1]
    if slope < 0 :
      left_fit.append((slope, intercept))
    else :
      right_fit.append((slope, intercept))
  left_fit_average = np.average(left_fit, axis=0)
  right_fit_average = np.average(right_fit, axis=0)
  left_line = make_cordinates(image, left_fit_average)
  right_line = make_cordinates(image, right_fit_average)
  return np.array([left_line, right_line])
  
def canny(image) :
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  canny = cv2.Canny(blur, 50, 150)
  return canny

def display_lines(image, lines) :
  line_image = np.zeros_like(image)
  if lines is not None :
    for x1, y1, x2, y2 in lines :
      # print(line)
      # x1, y1, x2, y2 = line.reshape(4)
      cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
  return line_image

def region_of_interest(image) :
  # square = np.array([
  #   [(0, 600), (300, 150), (900, 150), (1280, 600)]
  #   ])
  polygons = np.array([
  [(0, 600), (1280, 600), (660, 0)]
  ])
  mask = np.zeros_like(image)
  cv2.fillPoly(mask, polygons, 255)
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

while True:
    ret, frame= cap.read()
    # cv2.imshow('Jetson nano camera', frame)
    
    now = datetime.datetime.now().strftime("%d_%H_%M_%S") 
    
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_line = averae_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_line)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    
    key = cv2.waitKey(1)
    if key == 27 :  #esc
        break
    elif key == 99 :  #c
        print("Capture")
        cv2.imwrite("videos/" + str(now) + ".png", frame)
    elif key == 114 :  #r
        print("Video Record")
        record = True
        video = cv2.VideoWriter("videos/" + str(now) + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 60.0, (dispW, dispH))
    elif key == 115 : #s
        print("Video Record Stop")
        record = False
        video.release()
        
    if record == True :
        print("Video Recording...")
        video.write(frame)
        
        
cap.release()
cv2.destroyAllWindows()
