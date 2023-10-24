import cv2
import numpy as np

def region_of_interest(img, vertices):
    # ROI를 생성하기 위한 마스크를 생성합니다.
    mask = np.zeros_like(img)
    
    # 다각형 영역을 채웁니다.
    cv2.fillPoly(mask, vertices, 255)
    
    # ROI 영역을 만듭니다.
    masked_img = cv2.bitwise_and(img, mask)
    
    return masked_img

def detect_lanes(img):
    # 영상을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러를 적용합니다.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 엣지 검출을 수행합니다.
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI를 설정합니다.
    height, width = img.shape[:2]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # 허프 변환을 사용하여 차선을 검출합니다.
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # 검출된 차선을 영상에 그립니다.
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    
    # 원본 영상과 검출된 차선을 합성하여 반환합니다.
    
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    return result

def draw_lines(img, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 도로 주행 영상을 읽어옵니다.
cap = cv2.VideoCapture('C:/Users/LISA/Desktop/test.mp4')

while True:
    # 프레임을 읽어옵니다.
    ret, frame = cap.read()

    # 영상을 처리하여 차선을 인식합니다.
    lane_detected = detect_lanes(frame)

    # 인식된 차선을 화면에 표시합니다.
    cv2.imshow('Lane Detection', lane_detected)

    # 'q' 키를 누르면 반복문을 종료합니다.
    if cv2.waitKey(1) == ord('q'):
        break

# 사용한 자원을 해제합니다.
cap.release()
cv2.destroyAllWindows()
