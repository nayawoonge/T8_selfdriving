import cv2
import numpy as np

def preprocess(image):
    # 1) 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2) 가우시안 블러를 적용하여 노이즈 제거
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3) 캐니 엣지 알고리즘을 적용하여 엣지 추출
    canny = cv2.Canny(blur, 50, 150)
    
    return canny

def get_roi(image):
    mask = np.zeros_like(image)
    height, width = image.shape

    # ROI 설정
    roi = np.array([[
        (width * 0.05, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.95, height)
    ]], dtype=np.int32)

    # ROI에 해당하는 부분에 마스크 적용
    cv2.fillPoly(mask, roi, 255)

    return cv2.bitwise_and(image, mask)

def get_lines(image, rho=1, theta=np.pi/180, threshold=40, min_line_len=100, max_line_gap=50):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def draw_lines(image, lines):
    result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return result


cap = cv2.VideoCapture("rec2.avi")

while cap.isOpened():
    ret, frame = cap.read()
        
    if not ret:
        break
        
    processed = preprocess(frame)
    cv2.imshow('processed', processed)

    roi = get_roi(processed)
    cv2.imshow('roi', roi)

    lines = get_lines(roi)

    drawn = draw_lines(roi, lines)
    cv2.imshow('drawn', drawn)
        
    combined = cv2.addWeighted(frame, 0.8, drawn, 1, 1)
    cv2.imshow("combined", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
