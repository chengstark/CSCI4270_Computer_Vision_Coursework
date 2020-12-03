import cv2
import numpy as np
import copy
import math
from os import listdir
from os.path import isfile, join
from os import walk
import sys

# for testing only
# img_name1 = 'hw1_data/four_images/central_park.jpg'
# img_name2 = 'hw1_data/four_images/hop.jpg'
# s = 150
# M = 4
# N = 6

'''parse from input'''
img_name1 = sys.argv[1]
img_name2 = sys.argv[2]
out_img = sys.argv[3]
M = int(sys.argv[4])
N = int(sys.argv[5])
s = int(sys.argv[6])


'''crop the image into square'''
def squarify(img, img_name):
    col, row = img.shape[0], img.shape[1]
    if col < row:
        tmp = (row - col) / 2
        '''crop image and print output'''
        rt = img[0:col, int(tmp):int(row-tmp)]
        print('Image {} cropped at ({},{}) and ({},{})'.format(img_name, 0, int(tmp), col-1, int(row-tmp-1)))
    else:
        tmp = (col - row) / 2
        '''crop image and print output'''
        rt = img[int(tmp): int(col-tmp), 0:row]
        print('Image {} cropped at ({},{}) and ({},{})'.format(img_name, int(tmp), 0, int(col-tmp-1), row-1))
    return rt


'''read in image'''
img1 = cv2.imread(img_name1)
img2 = cv2.imread(img_name2)

'''make a image into square and resize into s*s'''
square1_ori = squarify(img1, img_name1)
square1 = cv2.resize(square1_ori, (s, s))
print('Resized from {} to {}'.format(square1_ori.shape, square1.shape))
square2_ori = squarify(img2, img_name2)
square2 = cv2.resize(square2_ori, (s, s))
print('Resized from {} to {}'.format(square2_ori.shape, square2.shape))

starter = copy.deepcopy(square1)
'''concatenate images into one row'''
for i in range(1, N):
    if i % 2 == 0:
        starter = np.concatenate((starter, square1), axis=1)
    else:
        starter = np.concatenate((starter, square2), axis=1)
'''repeat that one row'''
finished = np.tile(starter, (M, 1, 1))
'''save the output'''
cv2.imwrite('{}'.format(out_img), finished)
print('The checkerboard with dimensions {} X {} was output to {}'.format(M*s, N*s, out_img))



