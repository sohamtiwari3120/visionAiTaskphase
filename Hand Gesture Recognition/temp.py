import cv2
import numpy as np
from segment import *

kernel = np.ones((5,5), np.uint8)
cap = cv2.VideoCapture(0)
num_frames = 0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cpy = frame.copy()
    num_frames+=1
    (h, w) = frame.shape[:2]

    roi = frame[50:300, 50:300]
    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # ret, gray2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # lower_skin = np.array([64, 30, 28])
    # upper_skin = np.array([205, 174, 172])
    # mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # erosion = cv2.erode(opening, kernel, iterations=3)
    if num_frames<30:
        run_avg(blurred, 0.5)
    else:
        # run_avg(blurred, 0.002)
        hand = segmentImage(blurred, threshold=25)
        if hand is not None:
            (thresholded, segmented)=hand
            cv2.drawContours(frame, [segmented+(50, 50)], -1, (0, 0, 255))
            hull = cv2.convexHull(segmented)
            # hull = cv2.convexHull(segmented, returnPoints=False)
            # print('segmented')
            # print(segmented[0].shape)
            # print(segmented)
            # print(segmented.shape)
            # print('hull')
            # print(hull, hull.shape)
            # defects = cv2.convexityDefects(segmented, hull)
            # print(hull)
            # if defects is not None:
            #     for i in range(defects.shape[0]):
            #         s,e,f,d = defects[i,0]
            #         start = tuple(segmented[s][0]+(50, 50))
            #         end = tuple(segmented[e][0]+(50, 50))
            #         far = tuple(segmented[f][0]+(50, 50))
            #         # print(start)
            #         # print(far)
            #         cv2.line(frame,start,end,[10,255,0],1)
            #         cv2.circle(frame,far,5,[0,0,255],-1)

            cv2.drawContours(frame, [hull+(50, 50)], -1, (10, 255, 0))
            cv2.imshow('Thresholded', thresholded)
    cv2.rectangle(frame, (50, 50), (300, 300), (10, 255, 0), 3)
    cv2.imshow('original', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('gray', gray)
    # cv2.imshow('opening', opening)
    # cv2.imshow('erosion', erosion)
    # cv2.imshow('gray2', gray2)
    k = cv2.waitKey(1)
    if k & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()