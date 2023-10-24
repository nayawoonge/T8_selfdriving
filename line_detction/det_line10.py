from code import interact
from turtle import right
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import null

def edge_generator(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # grayscale conversion
    blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0) # blurring using gaussian matrices
    #Edge detection
    canny_image = cv2.Canny(blur_image, 50, 150) #detects the edges given a range of accepted jump/change in intensity values
    return canny_image

def region_of_interest(image):
    height = image.shape[0] #2d array length = number of rows = y
    polygons = np.array([[(200, height), (1100, height), (550,250)]]) # manually checked and saw in matplotlib
    mask = np.zeros_like(image) # black image as likezeroes makes array of same dimensions with zero (here: intensity)
    cv2.fillPoly(mask, polygons, 255) # 255 - white color
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1) # makes a linear fit
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
        if len(left_fit) and len(right_fit):
            left_fit_average  = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)
            left_line  = make_coordinates(image, left_fit_average)
            right_line = make_coordinates(image, right_fit_average)
            averaged_lines = [left_line, right_line]
            return averaged_lines

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

image = cv2.imread('finding-lanes/test_image.jpeg') # get the image
lane_image = np.copy(image) # making a copyof image array -- deep copy

canny_image = edge_generator(lane_image) # use the defined function to get canny edge image
cropped_image = region_of_interest(canny_image)
#HOUGH TRANFORM
lines_detected = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) # running hugh transform to get bin with best r, theta value
averaged_lines = average_slope_intercept(lane_image, lines_detected)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('result', combo_image) # shows only region of interest
cv2.waitKey(0)
#-------------OR----------------#
# plt.imshow(canny_image)
# plt.show()