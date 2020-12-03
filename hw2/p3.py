import os
import numpy as np
import cv2
import sys

# parse argument
folder = sys.argv[1]
f = os.listdir(folder)

# recorders for recording the results
rec1 = []
rec2 = []
# loop through the folder
for filename in f:
    img = cv2.imread(folder+'/'+filename, 0)
    # using sobel for images
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # calculate magnitudes for gradients
    magx = np.linalg.norm(sobelx)
    magy = np.linalg.norm(sobely)
    # calculate average
    magx = magx*magx/(img.shape[0]*img.shape[1])
    magy = magy*magy/(img.shape[0]*img.shape[1])
    mag = magx+magy
    # append into containers
    rec1.append([filename, mag])
    rec2.append([mag, filename])

# sort for name and magnitude and
rec1 = sorted(rec1)
rec2 = sorted(rec2, reverse=True)

# print to console
for info in rec1:
    print('{}: {:.2f}'.format(info[0], info[1]))
print('Image {} is best focused.'.format(rec2[0][1]))




