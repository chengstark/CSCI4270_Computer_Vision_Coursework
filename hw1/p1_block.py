import cv2
import numpy as np
import copy
import sys

# for testing only
# m = 25
# n = 18
# b = 15
# filename = 'hw1_data/lincoln1.jpg'

'''parsing inputs'''
filename = sys.argv[1]
m = int(sys.argv[2])
n = int(sys.argv[3])
b = int(sys.argv[4])

'''read files and convert to gray scale'''
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''calculate image dimensions'''
M, N = gray.shape
sm = float(M/m)
sn = float(N/n)

x = 0
y = 0
intensity_map = np.zeros((m, n), dtype=np.float64)
filled_img = np.zeros((m * b, n * b), dtype=np.float64)
for j in range(m):
    for i in range(n):
        '''calculate intensity average of a chunk of image'''
        tmp_matrix = gray[int(j*sm):int((j+1)*sm), int(i*sn):int((i+1)*sn)]
        mean = np.mean(tmp_matrix, dtype=np.float64)
        '''store the mean to a intensity map and fill in the output image chunk by chunk'''
        filled_img[j * b:(j + 1) * b, i * b:(i + 1) * b] = mean
        intensity_map[j, i] = mean

'''write to output'''
cv2.imwrite(filename[:-4] + '_g.jpg', filled_img)

'''calculate median'''
mid = np.median(filled_img)

'''convert to binary image'''
binary_filled = copy.deepcopy(filled_img)
binary_filled[filled_img < mid] = 0
binary_filled[filled_img >= mid] = 255

'''write to output'''
cv2.imwrite(filename[:-4] + '_b.jpg', binary_filled)

'''write log to console'''
print('Downsized images are ({}, {})'.format(m, n))
print('Block images are ({}, {})'.format(filled_img.shape[0], filled_img.shape[1]))
print('Average intensity at ({}, {}) is {:.2f}'.format(m // 3, n // 3, intensity_map[m // 3, n // 3]))
print('Average intensity at ({}, {}) is {:.2f}'.format(m // 3, 2*n // 3, intensity_map[m // 3, 2*n // 3]))
print('Average intensity at ({}, {}) is {:.2f}'.format(2*m // 3, n // 3, intensity_map[2*m // 3, n // 3]))
print('Average intensity at ({}, {}) is {:.2f}'.format(2*m // 3, 2*n // 3, intensity_map[2*m // 3, 2*n // 3]))
print('Binary threshold: {:.2f}'.format(mid))
print('Wrote image {}'.format(filename[:-4] + '_g.jpg'))
print('Wrote image {}'.format(filename[:-4] + '_b.jpg'))
