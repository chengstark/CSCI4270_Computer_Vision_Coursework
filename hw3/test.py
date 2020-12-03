import numpy as np
import cv2
import math

np.random.seed(1)
img = cv2.imread('whistler.png', 0)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel = np.absolute(sobel_x) + np.absolute(sobel_y)
a = np.asarray([ 0. , 8. , 8., 8.,  2., 24.,  8., 14., 20.,  2.])
last_local_min = np.zeros_like(a)
left = a[:-2]
right = a[2:]
center = a[1:-1]
last_local_min[1:-1] = (1-((center == right)*(center == left))) * (right <= center) * (left <= center) * center + ((1-((center == right)*(center == left)))*(center <= right) * (left <= right)) \
                       * right + ((1-((center == right)*(center == left)))*(right <= left) * (center <= left)) * left + ((center == right) * (center == left)) * left
# last_local_min[1:-1] = (left<=center)*(right<=center)*center + ((center == right) * (center == left)) * left
last_local_min[0] = max(a[0], a[1])
last_local_min[-1] = max(a[-1], a[-2])
print(a.shape, last_local_min.shape)
print(a)
print(last_local_min)
# print(sobel[0, 0:10])
# print(sobel[1, 0:10])
# sobel[1, 0:10] += last_local_min
# print(sobel[1, 0:10])
# print(sobel[:, 0:10])

# test = sobel[0, 0:10]
# print(test)
# L = test[:-2]
# R = test[2:]
# C = test[1:-1]
# V = [L, C, R]
# min_vals = np.zeros_like(test)
# min_vals[1:-1] = np.asarray(V).min(0)[:]  # middle
# min_vals[0] = np.amin(np.asarray([test[0], test[1]]))  # left border
# min_vals[-1] = np.amin(np.asarray([test[-1], test[-2]]))  # right border
# print(min_vals)
# import os
# files = os.listdir('carving')
# for file in files:
#     if file.endswith('.jpg'):
#         cv2.imread()