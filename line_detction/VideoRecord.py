import datetime
import cv2

capture = cv2.VideoCapture("/camera/video1.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while (1) :
  if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)) :
    capture.open("/camera/video1.mp4")
    
  ret, frame = capture.read()
  cv2.imshow('VideoFrame', frame)
  
  now = datetime.datetime.now().strftime("%d_%H_%M_%S") #filename : date_hour_minutes_seconds
  key = cv2.waitkey(33)
  
  if key == 27 :  #esc
    break
  elif key == 26 :  #ctrl + z
    print("Capture")
    cv2.imwrite("/camera" + str(now) + ".png", frame)
  elif key == 24 :  #ctrl + x
    print("Video Record")
    record = True
    video = cv2.VideoWriter("/camera" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
  elif key == 3 : #ctrl + c
    print("Video Record Stop")
    record = False
    video.release()

  if record == True :
    print("Video Recording...")
    video.write(frame)
    
capture.release()
cv2.destroyAllWindows()