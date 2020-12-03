import cv2
import numpy as np
import sys

sigma = int(sys.argv[1])
file = sys.argv[2]
name = sys.argv[2][:-4]
extension = file[-4:len(file)]

'''function to generate harris image'''
def generate_harris(im):
    ''' Gaussian smoothing '''
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
    im_harris_ = im_det - kappa * im_trace * im_trace
    return im_harris_


'''function to apply NMS'''
def nms(im_):
    i_min = np.min(im_)
    i_max = np.max(im_)
    im_harris_ = 255 * (im_ - i_min) / (i_max - i_min)

    '''
    Apply non-maximum thresholding using dilation, which requires the image
    to be uint8.  Comparing the dilated image to the Harris image will preserve
    only those locations that are peaks.
    '''
    max_dist = 2 * sigma
    kernel = np.ones((2 * max_dist + 1, 2 * max_dist + 1), np.uint8)
    im_harris_dilate = cv2.dilate(im_harris_, kernel)
    im_harris_[np.where(im_harris_ < im_harris_dilate)] = 0

    return im_harris_


'''plot harris keypoints'''
def harris_plot(im_, harris_map_, limiter):
    harris_keypoints = []
    keypoint_size = 4 * sigma
    '''loop through the harris map and create keypoints'''
    for index, x in np.ndenumerate(harris_map_):
        harris_keypoints.append(cv2.KeyPoint(index[1], index[0], _size=keypoint_size,
                                             _response=harris_map_[index[0]][index[1]]))
    '''sort the harris keypoints'''
    harris_keypoints = sorted(harris_keypoints, key=lambda keypoint: keypoint.response, reverse=True)
    harris_keyed = cv2.drawKeypoints(im_.astype(np.uint8), harris_keypoints[:limiter], None)
    cv2.imwrite('{}_harris{}'.format(name, extension), harris_keyed)
    return harris_keypoints


'''function to find the closest point in the other list'''
def find_closest_dist(pt0, pts):
    '''calculate distance between a point an array of points'''
    dist = np.sqrt(np.square((pt0[0] - pts[:, 0])) + np.square((pt0[1] - pts[:, 1])))
    '''return the closest point's distance and index'''
    return [dist[np.argmin(dist)], np.argmin(dist)]


'''helper function'''
def pretty_print(lst):
    line = ''
    for i in range(10):
        if i < 9:
            line += '{}: ({:.2f}, {:.2f}) {:.4f}\n'.format(i, lst[i].pt[0], lst[i].pt[1], lst[i].response)
        else:
            line += '{}: ({:.2f}, {:.2f}) {:.4f}'.format(i, lst[i].pt[0], lst[i].pt[1], lst[i].response)
    return line


'''get closest points and calc rank/image distance difference'''
def get_closest(harris_lst_, orb_lst_):
    '''load coordinates'''
    h_coords = []
    o_coords = []
    for i in range(harris_lst_.__len__()):
        x, y = harris_lst_[i].pt
        h_coords.append([x, y])
    for i in range(orb_lst_.__len__()):
        x, y = orb_lst_[i].pt
        o_coords.append([x, y])

    h_coords = np.asarray(h_coords)
    o_coords = np.asarray(o_coords)
    '''find  closest points in the other list'''
    h_closest = np.apply_along_axis(find_closest_dist, 1, h_coords[:min(len(harris_lst_), 100)],
                                    o_coords[:min(len(orb_lst), 200)])
    o_closest = np.apply_along_axis(find_closest_dist, 1,
                                    o_coords[:min(len(orb_lst), 100)], h_coords[:min(len(harris_lst_), 200)])

    '''get current list's rank'''
    h_ori_rank = np.arange(min(len(harris_lst_), 100))
    o_ori_rank = np.arange(min(len(orb_lst), 100))
    '''calculate rank difference'''
    h_rank = np.abs(h_closest[:, 1] - h_ori_rank)
    o_rank = np.abs(o_closest[:, 1] - o_ori_rank)

    '''print outputs'''
    print('\nHarris keypoint to ORB distances:')
    print('Median distance: {:.1f}'.format(np.median(h_closest[:, 0])))
    print('Average distance: {:.1f}'.format(np.average(h_closest[:, 0])))
    print('Median index distance: {:.1f}'.format(np.median(h_rank)))
    print('Average index distance: {:.1f}'.format(np.average(h_rank)))

    print('\nORB keypoint to Harris distances:')
    print('Median distance: {:.1f}'.format(np.median(o_closest[:, 0])))
    print('Average distance: {:.1f}'.format(np.average(o_closest[:, 0])))
    print('Median index distance: {:.1f}'.format(np.median(o_rank)))
    print('Average index distance: {:.1f}'.format(np.average(o_rank)))


def print_top10(harris_lst_, orb_lst_):
    '''print top 10 of the lists'''
    h_top10 = harris_lst_[:10]
    o_top10 = orb_lst_[:10]
    print('\nTop 10 Harris keypoints:')
    print(pretty_print(h_top10))
    print('\nTop 10 ORB keypoints:')
    print(pretty_print(o_top10))

if __name__ == '__main__':
    '''harris key point detection'''
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    im_harris = generate_harris(im)
    harris_map = nms(im_harris)
    harris_lst = harris_plot(im, harris_map, 200)

    '''orb key point detection'''
    orb = cv2.ORB_create(1000)
    orb_lst, des = orb.detectAndCompute(im, None)
    '''filter out key point size bigger than 45'''
    orb_lst = [i for i in orb_lst if i.size <= 45]
    '''sort the orb key point list'''
    orb_lst = sorted(orb_lst, key=lambda keypoint: keypoint.response, reverse=True)
    orb_lst = orb_lst[:200]
    '''pick top 200 and draw'''
    orb_keyed = cv2.drawKeypoints(im, orb_lst, None)
    cv2.imwrite('{}_orb{}'.format(name, extension), orb_keyed)
    '''print top 10 of the both key point lists'''
    print_top10(harris_lst, orb_lst)
    '''calculate closest points and print average and median coordinate distance/ rank distance'''
    get_closest(harris_lst, orb_lst)



