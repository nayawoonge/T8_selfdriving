import cv2
import numpy as np
import time

class Car:
    def control_car(self, left_speed, right_speed):
        print(f"Controlling car with left speed: {left_speed} and right speed: {right_speed}")

    def car_run(self, speed1, speed2):
        print(f"Running car with speed1: {speed1} and speed2: {speed2}")

    def forward(self):
        print("Moving forward")
        # You can adjust these speeds or use different logic for forward movement
        self.car_run(50, 50)

    def backward(self):
        print("Moving backward")
        # Adjust these speeds or logic as required for backward movement
        self.car_run(-50, -50)

    def left(self):
        print("Turning left")
        # Adjust these speeds or logic for left turn
        self.control_car(25, 50)

    def right(self):
        print("Turning right")
        # Adjust these speeds or logic for right turn
        self.control_car(50, 25)
car = Car()
car.forward()
car.backward()
car.left()
car.right()
def resize_image(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width, height // 2), (0, height // 2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    left_points, right_points = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            midpoint = (x1 + x2) // 2
            if midpoint < width // 2:
                left_points.append(midpoint)
            else:
                right_points.append(midpoint)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    left_lane = int(np.mean(left_points)) if left_points else 0
    right_lane = int(np.mean(right_points)) if right_points else width

    cv2.line(frame, (left_lane, height), (left_lane, height // 2), (0, 0, 255), 5)
    cv2.line(frame, (right_lane, height), (right_lane, height // 2), (0, 0, 255), 5)

    return left_lane, right_lane, frame

def main():
	
    car.forward()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        left_lane, right_lane, detected_frame = detect_lane(frame)
        detected_frame = resize_image(detected_frame, 50)

        center = detected_frame.shape[1] // 2
        lane_center = (left_lane + right_lane) // 2
        Bias = center - lane_center

        if abs(Bias) > 50:
            if Bias > 0:
                car.control_car(30, 45)
            else:
                car.control_car(45, 30)
        else:
            car.car_run(45, 45)

        cv2.imshow('Lane Detection', detected_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()