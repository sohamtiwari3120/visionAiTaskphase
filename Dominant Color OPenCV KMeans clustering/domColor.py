import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

img = cv2.imread('img1.jpg', 1)
print(img.shape)
print('KMeans clustering started...')
km = KMeans(n_clusters=6)
X = img.copy().reshape(-1, img.shape[2])
print(X.shape)
km.fit(X)
print('Fitting of data over...')
X_comp = km.cluster_centers_[km.labels_]
X_comp = np.clip(X_comp.astype('uint8'), 0, 255)
print(km.cluster_centers_)
print(X_comp.shape)
X_comp = X_comp.reshape(img.shape[0], img.shape[1], img.shape[2])
print(X_comp.shape)
colors, freq = np.unique(km.labels_, return_counts=True)
maxInd = np.argmax(freq)
domColor = km.cluster_centers_[colors[maxInd]]
domColor = np.clip(domColor.astype('uint8'), 0, 255)
domImg = img.copy()
domImg[:,:]=domColor
cv2.putText(domImg, str(domColor), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(domImg, str(domColor), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

print("Dominant color: ", domColor)
cv2.imshow('original', img)
cv2.imshow('comp', X_comp)
cv2.imshow('dominant color', domImg)
cv2.waitKey(0)
cv2.destroyAllWindows()