# -*- coding: utf-8 -*-
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=[0, 255, 0], thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect_lane(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    height, width = edges.shape
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)

    return masked

def roi(image):
    # 이미지 크기 가져오기
    x = int(image.shape[1])
    y = int(image.shape[0])
    
    # print(x, y)

    _shape = np.array([
          [int(0.1*x), int(y)], 
          [int(0.1*x), int(0.1*y)], 
          [int(0.4*x), int(0.1*y)], 
          [int(0.4*x), int(y)], 
          [int(0.6*x), int(y)], 
          [int(0.6*x), int(0.1*y)], 
          [int(0.9*x), int(0.1*y)], 
          [int(0.9*x), int(y)], 
          [int(0.2*x), int(y)]
          ])
    # print(_shape)
    cv2.polylines(image, [_shape], 1, (255,0,0))
    # cv2.imshow('image', image)
    
    mask = np.zeros_like(image)
    # image 채널 개수 = image.shape[2] = 3
    # print(image.shape[2])

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])

    # source = np.float32([[w // 2 - 30, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    # destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])
    source = np.float32([[0.1*w, 0.1*h], [0.9*w, 0.1*h], [w * 0.1, h*0.9], [0.9*w, h*0.9]])
    destination = np.float32([[0, 0], [w-250, 0], [400, h], [w-650, h]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)    
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))
    cv2.imshow('img', _image)
    
    plt.imshow(_image),plt.title('Perspective')
    plt.show()
    
    return _image, minv

def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    
    # print(leftbase, rightbase)

    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 4
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        #네모 박스
        cv2.imshow("oo", out_img)

        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color = 'yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret

def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result

def process_frame(frame):
    color_filtered = color_filter(frame)    
    # cv2.imshow('color_filtered', color_filtered)
    
    roied = roi(color_filtered)
    # cv2.imshow('roi', roied)
    
    warped, minv = wrapping(roied)
    # warped, minv = wrapping(roied)
    
    left_base, right_base = plothistogram(warped)
    
    draw_info = slide_window_search(warped, left_base, right_base)
    _, result = draw_lane_lines(frame, warped, minv, draw_info)
    return result
  



# #표시할 비디오의 출력 형식과 크기
# dispW=1280
# dispH=720
# #비디오 프레임의 방향
# flip=2
# camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cap = cv2.VideoCapture(camSet)

# if not cap.isOpened():
#     print('영상 파일을 열 수 없습니다.')
#     exit()

# record = False

# while True:
#     ret, frame= cap.read()
#     if not ret:
#         break
#     # cv2.imshow('Jetson nano camera', frame)
    
#     now = datetime.datetime.now().strftime("%d_%H_%M_%S") 
#     #filename : date_hour_minutes_seconds
#     # out = cv2.VideoWriter("videos/" + str(now) + ".avi",  cv2.VideoWriter_fourcc(*'XVID'), 60, (dispW, dispH))
#     # out.write(frame)
#     result_frame = process_frame(frame)
#     # cv2.imshow('Frame', result_frame)
    
#     key = cv2.waitKey(30)
#     if key == 27 :  #esc
#         break
#     elif key == 99 :  #c
#         print("Capture")
#         cv2.imwrite("videos/" + str(now) + ".png", frame)
#     elif key == 114 :  #r
#         print("Video Record")
#         record = True
#         video = cv2.VideoWriter("videos/" + str(now) + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 60.0, (dispW, dispH))
#     elif key == 115 : #s
#         print("Video Record Stop")
#         record = False
#         video.release()
        
#     if record == True :
#         print("Video Recording...")
#         video.write(frame)
        
# cap.release()
# cv2.destroyAllWindows()

image = cv2.imread('C:/Users/jiung/Downloads/17_16_07_53.png')
result_frame = process_frame(image)
cv2.imshow('Frame', result_frame)
cv2.waitKey(0)