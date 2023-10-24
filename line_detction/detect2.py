import cv2
import numpy as np

def sliding_window_lane_detection(binary_img, n_windows=9, margin=100, min_pix=50):
    # 이미지 하단 부분에서 윈도우 시작 지점 계산
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # 윈도우 높이 계산
    window_height = binary_img.shape[0] // n_windows

    # 차선 좌표를 저장할 리스트 초기화
    left_lane_inds = []
    right_lane_inds = []

    # 현재 차선 추적 지점
    left_current = left_base
    right_current = right_base

    # 슬라이딩 윈도우 탐색
    for window in range(n_windows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height

        win_x_left_low = left_current - margin
        win_x_left_high = left_current + margin
        win_x_right_low = right_current - margin
        win_x_right_high = right_current + margin

        # 윈도우 내에서 차선 픽셀을 찾기 위해 비트마스크 사용
        good_left_inds = ((binary_img[win_y_low:win_y_high, win_x_left_low:win_x_left_high]).nonzero()[0] + win_y_low,
                          (binary_img[win_y_low:win_y_high, win_x_left_low:win_x_left_high]).nonzero()[1] + win_x_left_low)

        good_right_inds = ((binary_img[win_y_low:win_y_high, win_x_right_low:win_x_right_high]).nonzero()[0] + win_y_low,
                           (binary_img[win_y_low:win_y_high, win_x_right_low:win_x_right_high]).nonzero()[1] + win_x_right_low)

        # 차선 픽셀 저장
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 다음 차선 추적 지점 업데이트
        if len(good_left_inds[0]) > min_pix:
            left_current = np.int(np.mean(good_left_inds[1]))
        if len(good_right_inds[0]) > min_pix:
            right_current = np.int(np.mean(good_right_inds[1]))

    # 리스트를 배열로 변환하여 차선 픽셀들을 얻음
    left_lane_inds = np.concatenate(left_lane_inds, axis=1)
    right_lane_inds = np.concatenate(right_lane_inds, axis=1)

    # 좌표 추출
    leftx, lefty = left_lane_inds[1], left_lane_inds[0]
    rightx, righty = right_lane_inds[1], right_lane_inds[0]

    # 다항식으로 차선 피팅
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

cap = cv2.VideoCapture("./rec1.avi")
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if not cap.isOpened():
    print('영상 파일을 열 수 없습니다.')
    exit()

record = False

while True:
    retval, image = cap.read()
    if not retval:
        break

    # 이미지를 HSV로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 흰색 차선 필터링
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # 노란색 차선 필터링
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 흰색과 노란색 차선을 합침
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 흰색과 노란색 차선을 이진화 이미지로 변환
    binary_img = np.zeros_like(combined_mask)
    binary_img[combined_mask > 0] = 1

    # 슬라이딩 윈도우 기반 차선 인식
    left_fit, right_fit = sliding_window_lane_detection(binary_img)

    # 시각화
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # 왼쪽 차선 피팅 결과 시각화
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    cv2.polylines(out_img, np.int_([pts_left]), isClosed=False, color=(255, 0, 0), thickness=5)

    # 오른쪽 차선 피팅 결과 시각화
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    cv2.polylines(out_img, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=5)

    # 이미지와 차선 시각화
    result = cv2.addWeighted(image, 1, out_img, 0.3, 0)
    cv2.imshow('Lane Detection', result)


    key = cv2.waitKey(30)
    if key == 27 :  #esc
        break



if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()


