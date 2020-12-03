import cv2
import numpy as np
import sys

# read image
img = cv2.imread(sys.argv[1])
# create matrix of pixel coordinates
coords = np.meshgrid(range(img.shape[0]), range(img.shape[1]), indexing='ij')

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
im_grd = (np.absolute(sobelx) + np.absolute(sobely)) / 20
print(im_grd.shape)

xs = coords[0].reshape((coords[0].shape[0], coords[0].shape[1], 1))/10
ys = coords[1].reshape((coords[1].shape[0], coords[1].shape[1], 1))/10

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(img_hsv.shape)

# combine features
features = img
features = np.concatenate((img, xs), axis=2)
features = np.concatenate((features, ys), axis=2)
# features = np.concatenate((features, img_hsv), axis=2)
# reshape features
Z = features.reshape((-1, 5))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
print(res.shape)
res2 = res.reshape((img.shape[0], img.shape[1], 5))
# write result to output
print(res2.shape)
cv2.imwrite('res.jpg', res2[:, :, :3])
