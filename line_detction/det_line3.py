import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

cap = cv2.VideoCapture('rec2.avi')  # 카메라 장치를 사용할 경우

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 회색으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    height, width = edges.shape

    # 관심 영역 설정
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # 허프 변환을 통한 선 검출
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    # 예측 선을 원래 이미지에 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 결과 출력
    cv2.imshow('Lane Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
