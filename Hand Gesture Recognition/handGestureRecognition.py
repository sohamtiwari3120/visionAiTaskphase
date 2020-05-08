import cv2
import numpy as np
from segment import *
import math
from sklearn.metrics.pairwise import euclidean_distances
kernel = np.ones((5,5), np.uint8)
cap = cv2.VideoCapture(0)
num_frames = 0
cX, cY = None, None
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))
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
        hand = segmentImage(blurred, 12)
        if hand is not None:
            (thresholded, segmented, diff)=hand
            cv2.drawContours(frame, [segmented+(50, 50)], -1, (0, 0, 255))
            hull = cv2.convexHull(segmented, returnPoints=False)
            chull = cv2.convexHull(segmented)
            defects = cv2.convexityDefects(segmented, hull)
            count_defects = 0
            try:
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(segmented[s][0]+[50, 50])
                    end = tuple(segmented[e][0]+[50, 50])
                    far = tuple(segmented[f][0]+[50, 50])

                    a = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
                    b = ((end[0]-far[0])**2 + (end[1]-far[1])**2)**0.5
                    c = ((far[0]-start[0])**2 + (far[1]-start[1])**2)**0.5

                #have computed the length of the sides of the triangle a,b,c
                # we will now use the cosine rule to compute the angle
                    theta = math.acos((b**2 + c**2 - a **2)/(2*b*c))*180/math.pi
                    if(theta<=90):
                        count_defects+=1
                        cv2.circle(frame, far, 5, (255, 0, 255), -1)
                    cv2.line(frame, start, end, [10, 255, 0], 2)
            except:
                a=0
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Fingers: '+str(count_defects+1), (50,50), font, 1, (250, 150, 255), 2, cv2.LINE_AA)
            # if(count_defects>=4):
            #     extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0]+[50, 50])
            #     extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0]+[50, 50])
            #     extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0]+[50, 50])
            #     extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0]+[50, 50])
            #     cX = int((extreme_left[0] + extreme_right[0]) / 2)
            #     cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
            if(count_defects == 0 ):
                extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0]+[50, 50])
                extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0]+[50, 50])
                extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0]+[50, 50])
                extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0]+[50, 50])
                cv2.circle(frame, extreme_top, 5, (50, 50, 255), -1)
                cv2.circle(frame, extreme_bottom, 5, (50, 50, 255), -1)
                cv2.circle(frame, extreme_left, 5, (50, 50, 255), -1)
                cv2.circle(frame, extreme_right, 5, (50, 50, 255), -1)

                # find the center of the palm
                cX = int((extreme_left[0] + extreme_right[0]) / 2)
                cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
                cv2.circle(frame, (cX, cY), 5, (10, 100, 255), -1)
                Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
                distance = euclidean_distances([(cX, cY)], Y=Y)[0]
                maximum_distance_index = distance.argmax()
                cv2.circle(frame, Y[maximum_distance_index], 5, (255, 255, 255), -1)
                directions = ['Left', 'Right', 'Top', 'Bottom']
                cv2.putText(frame, 'Dir: '+directions[maximum_distance_index], (300,350), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                print(distance)
                print(maximum_distance_index)
                print('-------------------------------------------------')
            # cv2.drawContours(frame, [hull+(50, 50)], -1, (10, 255, 0))
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)#erosion followed by dilation
            cv2.imshow('diff', diff)
            cv2.imshow('Thresholded', opening)
    cv2.rectangle(frame, (50, 50), (300, 300), (10, 255, 0), 3)
    cv2.imshow('original', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('gray', gray)
    out.write(frame)
    # cv2.imshow('opening', opening)
    # cv2.imshow('erosion', erosion)
    # cv2.imshow('gray2', gray2)
    k = cv2.waitKey(1)
    if k & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()