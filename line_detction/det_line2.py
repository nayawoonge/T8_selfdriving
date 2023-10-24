import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def filter_yellow_white(img):
    # 색상 범위를 지정하여 노란색과 흰색을 필터링합니다.
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 노란색 필터링
    yellow_lower = np.array([15, 30, 115], dtype=np.uint8)
    yellow_upper = np.array([35, 204, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

    # 흰색 필터링
    white_lower = np.array([0, 200, 0], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, white_lower, white_upper)

    return yellow_mask, white_mask

def detect_lane(image):
    yellow_mask, white_mask = filter_yellow_white(image)

    edges = cv2.Canny(cv2.bitwise_or(yellow_mask, white_mask), 50, 150)

    height, width = edges.shape
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    left_lane_points = []
    right_lane_points = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)

        if slope < 0:  # 왼쪽 차선
            left_lane_points.append((x1, y1))
            left_lane_points.append((x2, y2))
        else:  # 오른쪽 차선
            right_lane_points.append((x1, y1))
            right_lane_points.append((x2, y2))

    if len(left_lane_points) > 0 and len(right_lane_points) > 0:
        left_bottom = min(left_lane_points, key=lambda x: x[1])
        left_top = max(left_lane_points, key=lambda x: x[1])
        right_bottom = min(right_lane_points, key=lambda x: x[1])
        right_top = max(right_lane_points, key=lambda x: x[1])

        # 왼쪽 차선과 오른쪽 차선의 끝점 연결
        cv2.line(image, left_bottom, right_bottom, (0, 255, 0), 3)
        cv2.line(image, left_top, right_top, (0, 255, 0), 3)
        cv2.line(image, left_bottom, left_top, (0, 255, 0), 3)
        cv2.line(image, right_bottom, right_top, (0, 255, 0), 3)

        # 끝점 연결된 영역 색칠
        pts = np.array([left_bottom, left_top, right_top, right_bottom], np.int32)
        cv2.fillPoly(image, [pts], (0, 100, 255))

    return image

# cap = cv2.VideoCapture('C:/Users/LISA/Desktop/test.mp4')  # 카메라 장치를 사용할 경우
cap = cv2.VideoCapture('rec2.avi')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_lane(frame)

    cv2.imshow('Lane Detection', result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
