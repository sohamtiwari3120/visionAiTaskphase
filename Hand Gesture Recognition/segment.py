import cv2
import numpy as np
import imutils
kernel = np.ones((5, 5),np.uint8)
bg = None

def run_avg(curr_image, aWeight):
    global bg
    if bg is None:
        bg = curr_image.copy().astype('float')
        return
    cv2.accumulateWeighted(src = curr_image, dst = bg, alpha=aWeight)

def segmentImage(curr_image, threshold=25):
    global bg
    kernel = np.ones((5, 5),np.uint8)
    diff = cv2.absdiff(bg.astype('uint8'), curr_image)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    contours, heirarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print('contours')
    # print(contours[0])
    if(len(contours)==0):
        return
    else:
        segmented = max(contours, key = cv2.contourArea)
        return thresholded, segmented, diff