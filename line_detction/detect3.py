# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

# 비디오 프레임의 방향
flip=2
# 표시할 비디오의 출력 형식과 크기
dispW=1280
dispH=720

# camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cap = cv2.VideoCapture(camSet)

cap = cv2.VideoCapture("./rec1.avi")
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if not cap.isOpened():
    print('영상 파일을 열 수 없습니다.')
    exit()

record = False

while True:
    retval, img = cap.read()
    if not retval:
        break
    
    # 조감도 wrapped img
    (h, w) = (img.shape[0], img.shape[1])

    # 좌상, 우상, 좌하, 우하
    source = np.float32([[300,500],[550,500],
                         [100,600],[700,600]])
    destination = np.float32([[0, 0], [w-50, 0], 
                              [100, h], [w-150, h]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minverse = cv2.getPerspectiveTransform(destination, source)
    wrapped_img = cv2.warpPerspective(img, transform_matrix, (w, h))
    
    #조감도 imshow
    cv2.imshow("wrapped_img", wrapped_img)
    # plt.imshow(img)
    # plt.show()
    
    canny = cv2.Canny(wrapped_img,50,150)
    cv2.imshow("canny",canny)
    canny_mask = np.zeros_like(canny)
    canny_mask[canny > 0] = 255
    cv2.imshow("canny_mask", canny_mask)
    
    # 조감도 필터링
    hls = cv2.cvtColor(wrapped_img, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(wrapped_img, wrapped_img, mask = mask)
    
    print(type(masked))
    

    # 조감도 필터링 자르기
    x = int(masked.shape[1])
    y = int(masked.shape[0])

    # ROI 영역 좌표
    _shape = np.array(
        [[0, int(y)], [0, 0], [int(0.4*x), 0], [int(0.4*x), int(y)], [int(0.6*x), int(y)], [int(0.6*x), 0],[int(x), 0], [int(x), int(y)], [0, int(y)]])

    cv2.polylines(masked, [_shape], 1, (255,0,0))
    
    # cv2.imshow("polylines", masked)
    
    new_mask = np.zeros_like(masked)

    if len(masked.shape) > 2:
        channel_count = masked.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(masked, new_mask)
    
    cv2.imshow('masked_image', masked_image)

    # 조감도 선 따기 wrapped img threshold
    _gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', thresh)

    # 선 분포도 조사 histogram
    histogram = np.sum(thresh[thresh.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    # plt.plot(hist)
    # plt.show()

    # histogram 기반 window roi 영역
    out_img = np.dstack((thresh, thresh, thresh))

    nwindows = 4
    window_height = np.int(thresh.shape[0] / nwindows)
    nonzero = thresh.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = thresh.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = thresh.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = leftbase - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = leftbase + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = rightbase - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = rightbase + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        
        cv2.imshow("out_img", out_img)

        if len(good_left) > minpix:
            leftbase = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            rightbase = np.int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    
    #if not leftx :
      

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, thresh.shape[0] - 1, thresh.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}
    # plt.plot(left_fit)
    # plt.show()

    # 원본 이미지에 라인 넣기
    left_fitx = ret['left_fitx']
    right_fitx = ret['right_fitx']
    ploty = ret['ploty']

    warp_zero = np.zeros_like(thresh).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, minverse, (img.shape[1], img.shape[0]))
    
    result = cv2.addWeighted(img, 1, newwarp, 0.4, 0)
    cv2.imshow("result", result)
 
    # 저장 파일 이름 형식 : date_hour_minutes_seconds
    now = datetime.datetime.now().strftime("%d_%H_%M_%S") 

    key = cv2.waitKey()
    if key == 27 :  #esc
        break
    elif key == 99 :  # c 캡처
        print("Capture")
        cv2.imwrite("videos/" + str(now) + ".png", result)
    elif key == 114 :  # r 영상 녹화
        print("Video Record")
        record = True
        video = cv2.VideoWriter("videos/" + str(now) + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, frame_size)
    elif key == 115 : # s 영상 녹화 종료
        print("Video Record Stop")
        record = False
        video.release()
        
    if record == True :
        print("Video Recording...")
        video.write(result)

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()
