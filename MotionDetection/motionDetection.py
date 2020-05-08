import cv2
import numpy as np
from pyImageSearchNMS import *
cap = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2
    frame = cv2.flip(frame, 1)
    fgmask = fgbg.apply(frame)
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel, iterations=2)
    _, dil = cv2.threshold(dilation, 240, 255, cv2.THRESH_BINARY)
    contours, heirarchy = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # im2 = np.zeros(frame.shape)
    # print(contours)
    # cv2.drawContours(im2, contours, -1, (0,0, 255), 3)
    minDim = 50
    # contours2 = non_max_suppression_fast(contours, 0.4)
    # print(type(contours), len(contours), contours[0].shape)
    boxes=[]
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        boxes+=[[x*1.0, y*1.0, w*1.0, h*1.0]]
    # print(boxes)
    # print(np.array(boxes))
    boxes2 = non_max_suppression_fast(np.array(boxes), 0.4)
    for box in boxes2:
        (x, y, w, h)=box
        if((w>minDim) and (h>minDim)):
         cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
    out.write(frame)
    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    cv2.imshow('dil', dil)
    # cv2.imshow('im2', im2)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()