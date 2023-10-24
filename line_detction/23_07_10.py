import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_cordinates(image, line_parameters) :
  slope, intercept = line_parameters
  print(image.shape)
  y1 = image.shape[0]
  y2 = int(y1*(3/5))
  x1 = int((y1-intercept)/slope)
  x2 = int((y2-intercept)/slope)
  return np.array([x1, y1, x2, y2])
  
def averae_slope_intercept(image, lines) :
  left_fit = []
  right_fit = []
  for line in lines :
    x1, y1, x2, y2 = line.reshape(4)
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    slope = parameters[0]
    intercept = parameters[1]
    if slope < 0 :
      left_fit.append((slope, intercept))
    else :
      right_fit.append((slope, intercept))
  left_fit_average = np.average(left_fit, axis=0)
  right_fit_average = np.average(right_fit, axis=0)
  left_line = make_cordinates(image, left_fit_average)
  right_line = make_cordinates(image, right_fit_average)
  return np.array([left_line, right_line])
  
def canny(image) :
  gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  canny = cv2.Canny(blur, 50, 150)
  return canny

def display_lines(image, lines) :
  line_image = np.zeros_like(image)
  if lines is not None :
    for x1, y1, x2, y2 in lines :
      # print(line)
      # x1, y1, x2, y2 = line.reshape(4)
      cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
  return line_image

def region_of_interest(image) :
  # square = np.array([
  #   [(0, 600), (300, 150), (900, 150), (1280, 600)]
  #   ])
  polygons = np.array([
  [(0, 600), (1280, 600), (660, 0)]
  ])
  mask = np.zeros_like(image)
  cv2.fillPoly(mask, polygons, 255)
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

image = cv2.imread('C:/Users/jiung/Downloads/10_17_22_58.png')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
plt.imshow(canny)
plt.show()
cv2.imshow("result", cropped_image)
cv2.waitKey(0)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_line = averae_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_line)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow("result", combo_image)
cv2.waitKey(0)