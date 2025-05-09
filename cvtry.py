import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


im = cv.imread('temp_plot.png')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 85, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv.approxPolyDP(cnt, .03 * cv.arcLength(cnt, True), True)
    if len(approx)==8:
        largest_contour = max(contours, key=cv.contourArea)

        if cv.contourArea(largest_contour) > 1000:
            cv.drawContours(im, [cnt], 0, (220, 152, 91), -1)

cv.imshow('image',im)
cv.waitKey(0)
cv.waitKey(1000)
cv.destroyAllWindows()