import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im = cv.imread('temp_plot_clean.png')
assert im is not None, "file could not be read, check with os.path.exists()"

# Convert the image to grayscale
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 80, 255, 0)

cv.imshow('imgray ', imgray)
print(ret)
cv.imshow('thresh', thresh)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

output_image = np.zeros_like(im)

# Create a list to store selected contours
selected_contours = []

# Iterate over all contours
for cnt in contours:
    # Calculate the contour area
    area = cv.contourArea(cnt)
    
    # Approximate the contour to a polygon
    approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)
    
    # Get the number of corners
    num_corners = len(approx)
    
    # Filter based on area and exclude contours with 3 corners (triangles)
    if area > 10000 and area < 35000 and num_corners != 3:
        # Add the contour to the list of selected contours
        selected_contours.append(cnt)
        
        print(f"Area: {area}")
        print(f"Number of corners: {num_corners}")
        
# drawing the contours and displaying image
cv.drawContours(output_image, selected_contours, -1, (220, 152, 91), -1)
cv.imshow('Selected Contours', output_image)
cv.waitKey(0)
cv.destroyAllWindows()