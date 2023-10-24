import cv2
import numpy as np

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

    cv2.imshow('edges', edges)

    height, width = edges.shape
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result


cap = cv2.VideoCapture('rec2.avi')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_lane(frame)

    cv2.imshow('Lane Detection', result)
    if cv2.waitKey(1) == ord('q'):
        break

    #waitKey(33) : 한장의 사진을 0.033초 동안 띄움 
    k = cv2.waitKey(33)

    #esc를 누르면 동영상 종료
    if k == 27 :
        print('동영상 종료')
        cap.realse()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()

