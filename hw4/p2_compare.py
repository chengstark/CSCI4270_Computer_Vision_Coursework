import cv2
import math
import numpy as np
import os
import sys
import copy
import math as m
import os
# worth to mention, cv <col, row>, np <row, col>
pi = math.pi
# sigma = 1
import cv2
import numpy as np
from scipy import spatial
# import image_plot_utilities as ipu
# import resize_img

# python p2_compare.py sigma img

sigma = float(sys.argv[1])
sigma = int(sigma)
img = sys.argv[2]

# calculate response map of harris and normalization
def my_get_harris():
    ksize = (4 * sigma + 1, 4 * sigma + 1)
    im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)

    '''  Derivative kernels '''
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    '''  Derivatives '''
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)

    ''' Components of the outer product '''
    im_dx_sq = im_dx * im_dx
    im_dy_sq = im_dy * im_dy
    im_dx_dy = im_dx * im_dy

    ''' Convolution of the outer product with the Gaussian kernel
        gives the summed values desired '''
    h_sigma = 2 * sigma
    h_ksize = (4 * h_sigma + 1, 4 * h_sigma + 1)
    im_dx_sq = cv2.GaussianBlur(im_dx_sq, h_ksize, h_sigma)
    im_dy_sq = cv2.GaussianBlur(im_dy_sq, h_ksize, h_sigma)
    im_dx_dy = cv2.GaussianBlur(im_dx_dy, h_ksize, h_sigma)

    ''' Compute the Harris measure '''
    kappa = 0.004
    im_det = im_dx_sq * im_dy_sq - im_dx_dy * im_dx_dy
    im_trace = im_dx_sq + im_dy_sq
    im_harris = im_det - kappa * im_trace * im_trace  # harris measure for each pixel

    ''' Normalized to 0..255 '''
    i_min = np.min(im_harris)
    i_max = np.max(im_harris)
    im_harris = 255 * (im_harris - i_min) / (i_max - i_min)
    return im_harris

# get the input from im_harris, a map of response
# using eight same size translation maps to do the 8-connected NMS
def map_nms(input_2darray):
    im_center = input_2darray[1:-1, 1:-1]
    im_up = input_2darray[:-2, 1:-1]
    im_down = input_2darray[2:, 1:-1]
    im_left = input_2darray[1:-1, :-2]
    im_right = input_2darray[1:-1, 2:]
    im_upleft = input_2darray[:-2, :-2]
    im_upright = input_2darray[2:, :-2]
    im_downleft = input_2darray[:-2, 2:]
    im_downright = input_2darray[2:, 2:]

    temp = cv2.compare(im_center, im_up, cv2.CMP_GE)
    temp = (temp == 255) * im_center + (temp == 0) * np.zeros_like(im_center)

    for map in [im_up, im_down, im_left, im_right, im_upleft, im_upright, im_downleft, im_downright]:
        temp = cv2.compare(temp, map, cv2.CMP_GE)
        temp = (temp == 255) * im_center + (temp == 0) * np.zeros_like(im_center)
    new_harris_map = np.zeros_like(input_2darray)
    new_harris_map[1:-1, 1:-1] = temp
    im_harris = new_harris_map
    return im_harris

# draw points of harris keypoints
def draw_points(im, im_harris, number):
    kp_size = 4 * sigma
    harris_keypoints = []
    for i in range(0, im_harris.shape[0]):
        for j in range(0, im_harris.shape[1]):
            harris_keypoints.append(cv2.KeyPoint(j, i, _size=kp_size, _response=im_harris[i][j]))

    harris_keypoints.sort(key=lambda k: k.response, reverse=True)
    selected = harris_keypoints[:number]
    out_img = cv2.drawKeypoints(im.astype(np.uint8), selected, None)
    cv2.imwrite('{}_harris.jpg'.format(im_name[:-4]), out_img)
    return selected

# the series of operation to complete harrris detection
def get_harris(number):
    response_map = my_get_harris()
    im_harris = map_nms(response_map)
    print('HARRIS: {}'.format(np.sum(im_harris)))
    harris_points = draw_points(im, im_harris, number)
    return harris_points

# detect orb keypoints from image
# place in order and save in a list
def get_orb(number):
    orb = cv2.ORB_create()
    orb_lst = orb.detect(im, None)
    orb_lst, des = orb.compute(im, orb_lst)
    orb_lst = sorted(orb_lst, key=lambda keypoint: keypoint.response, reverse=True)
    orb_lst = orb_lst[:200]
    return orb_lst

# as the list has been sorted
# print out the top 10 information
def get_10tops(harris, orb):
    file.write('\nTop 10 Harris keypoints:\n')
    for i in range(0, 10):
        file.write(
            '{}: ({:.2f}, {:.2f}) {:.4f} {:.2f}\n'.format(i,
                                                          harris[i].pt[0],
                                                          harris[i].pt[1],
                                                          harris[i].response,
                                                          harris[i].size))
    file.write('\nTop 10 orb keypoints:\n')
    for i in range(0, 10):
        file.write(
            '{}: ({:.2f}, {:.2f}) {:.4f} {:.2f}\n'.format(i,
                                                          orb[i].pt[0],
                                                          orb[i].pt[1],
                                                          orb[i].response,
                                                          orb[i].size))
# get list of harris and orb points
# forming two KDtrees to find mutually closest points
# save as an array to find the average and median
def compare_points(harris, orb):
    list_harris = []
    list_orb = []

    for i in range(0, 200):
        list_harris.append(harris[i].pt)
        list_orb.append(orb[i].pt)

    tree_harris = spatial.KDTree(list_harris)
    tree_orb = spatial.KDTree(list_orb)

    result_harris = []
    result_orb = []
    for i in range(0, 100):
        harris_query = tree_orb.query(list_harris[i])
        result_harris.append(harris_query)
        orb_query = tree_harris.query(list_orb[i])
        result_orb.append(orb_query)

    harris_result_array = np.asarray(result_harris)
    orb_result_array = np.asarray(result_orb)
    print(harris_result_array[:, 0])
    print(np.average(harris_result_array[:, 0]))

    rank = np.arange(100)
    rank = np.reshape(rank, (100, 1))
    harris_rank = np.reshape(harris_result_array[:, 1], (100, 1))
    orb_rank = np.reshape(orb_result_array[:, 1], (100, 1))

    harris_difference = np.absolute(harris_rank - rank)
    harris_result_array = np.concatenate((harris_result_array, harris_difference), axis=1)
    orb_difference = np.absolute(orb_rank - rank)
    orb_result_array = np.concatenate((orb_result_array, orb_difference), axis=1)

    median_distance = np.median(harris_result_array[:, 0])
    average_distance = np.average(harris_result_array[:, 0])
    median_index_difference = np.median(harris_result_array[:, 2])
    average_index_difference = np.average(harris_result_array[:, 2])

    print('Harris keypoint to orb distances:')
    print('num_from 100 num_to 200')
    print('Median distance: {:.1f}'.format(median_distance))
    print('Average distance: {:.1f}'.format(average_distance))
    print('Median index difference: {:.1f}'.format(median_index_difference))
    print('Average index difference: {:.1f}'.format(average_index_difference))

    file.write('\nHarris keypoint to orb distances:\n')
    file.write('num_from 100 num_to 200\n')
    file.write('Median distance: {:.1f}\n'.format(median_distance))
    file.write('Average distance: {:.1f}\n'.format(average_distance))
    file.write('Median index difference: {:.1f}\n'.format(median_index_difference))
    file.write('Average index difference: {:.1f}\n'.format(average_index_difference))

    median_distance = np.median(orb_result_array[:, 0])
    average_distance = np.average(orb_result_array[:, 0])
    median_index_difference = np.median(orb_result_array[:, 2])
    average_index_difference = np.average(orb_result_array[:, 2])

    print('orb keypoint to Harris distances:')
    print('num_from 100 num_to 200')
    print('Median distance: {:.1f}'.format(median_distance))
    print('Average distance: {:.1f}'.format(average_distance))
    print('Median index difference: {:.1f}'.format(median_index_difference))
    print('Average index difference: {:.1f}'.format(average_index_difference))

    file.write('\norb keypoint to Harris distances:\n')
    file.write('num_from 100 num_to 200\n')
    file.write('Median distance: {:.1f}\n'.format(median_distance))
    file.write('Average distance: {:.1f}\n'.format(average_distance))
    file.write('Median index difference: {:.1f}\n'.format(median_index_difference))
    file.write('Average index difference: {:.1f}\n'.format(average_index_difference))


if __name__ == '__main__':
    if img[-3:] == 'jpg':
        im_name = img
    else:
        im_name = img + '.jpg'

    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)

    file = open('hisout.txt'.format(im_name[:-4]), 'a')

    harris = get_harris(200)
    orb = get_orb(200)

    get_10tops(harris, orb)

    compare_points(harris, orb)

    file.close()









