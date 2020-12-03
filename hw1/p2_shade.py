import cv2
import numpy as np
import copy
import math
import sys

# for testing only
# dir = 'center'
# filename = 'hw1_data/four_images/central_park.jpg'
# filename = 'hw1_data/myimg/19201080.jpg'
# output_name = 'p2_{}.jpg'.format(dir)

'''parse input'''
filename = sys.argv[1]
output_name = sys.argv[2]
dir = sys.argv[3]

'''check required direction'''
if dir in ['left', 'top', 'right', 'bottom', 'center']:
    '''read image and create a empty multiplier map with the same dimension'''
    img = cv2.imread(filename)
    multi_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    M = img.shape[0]
    N = img.shape[1]

    if dir == 'center':
        '''create row-dimension map'''
        one_row1 = np.arange(M//2, -1, -1)
        one_row2 = np.arange(1, M//2, 1)
        one_row = np.concatenate((one_row1, one_row2))
        row_dim = np.repeat(one_row[:, np.newaxis], N, axis=1)

        '''create column dimension map'''
        one_col1 = np.arange(N//2, -1, -1)[:, np.newaxis]
        one_col2 = np.arange(1, N//2, 1)[:, np.newaxis]
        one_col = np.concatenate((one_col1, one_col2), axis=0).reshape(1, N)
        col_dim = np.repeat(one_col[:, ], M, axis=0)

        '''combine two maps with calculation of euclidean distance and normalize result'''
        multi_map = np.sqrt(np.multiply(row_dim, row_dim) + np.multiply(col_dim, col_dim), )
        max_val = np.max(multi_map)
        multi_map = np.divide(multi_map, max_val)
        multi_map = np.subtract(1, multi_map)

    elif dir == 'top':
        '''make multiplier map by creating a 1D array and covert to 2D with repeat and newaxis'''
        one_row = np.linspace(0, 1, M)
        multi_map = np.repeat(one_row[:, np.newaxis], N, axis=1)
        multi_map = np.subtract(1, multi_map)

    elif dir == 'bottom':
        '''make multiplier map by creating a 1D array and covert to 2D with repeat and newaxis'''
        one_row = np.linspace(0, 1, M)
        multi_map = np.repeat(one_row[:, np.newaxis], N, axis=1)

    elif dir == 'left':
        '''make multiplier map by creating 1D array and covert to 2D by tile'''
        one_row = np.linspace(0, 1, N)
        multi_map = np.tile(one_row, (M, 1))
        multi_map = np.subtract(1, multi_map)

    elif dir == 'right':
        '''make multiplier map by creating 1D array and covert to 2D by tile'''
        one_row = np.linspace(0, 1, N)
        multi_map = np.tile(one_row, (M, 1))

    np.set_printoptions(precision=3)
    '''print log to console'''
    for i in [0, M//2, M-1]:
        for j in [0, N//2, N-1]:
            print('({},{}) {:.3f}'.format(i, j, multi_map[i, j]))

    '''multiply the multiplier map to the original image'''
    multi_map = np.subtract(1, multi_map)
    repeated_map = np.repeat(multi_map[:, :, np.newaxis], 3, axis=2)
    cpy_img = copy.deepcopy(img)
    cpy_img = cpy_img * repeated_map
    '''concatenate original image and shaded image and save output'''
    out_image = np.concatenate((img, cpy_img), axis=1)
    cv2.imwrite(output_name, out_image)

