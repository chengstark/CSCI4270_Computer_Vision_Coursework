import cv2
import numpy as np

im = cv2.imread('examples/wisconsin.jpg', cv2.IMREAD_GRAYSCALE)
sigma = 2
ksize = (4*sigma+1,4*sigma+1)
im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)

'''  Derivative kernels '''
kx,ky = cv2.getDerivKernels(1,1,3)
kx = np.transpose(kx/2)
ky = ky/2

'''  Derivatives '''
im_dx = cv2.filter2D(im_s,-1,kx)
im_dy = cv2.filter2D(im_s,-1,ky)

''' Components of the outer product '''
im_dx_sq = im_dx * im_dx
im_dy_sq = im_dy * im_dy
im_dx_dy = im_dx * im_dy

''' Convolution of the outer product with the Gaussian kernel
    gives the summed values desired '''
h_sigma = 2*sigma
h_ksize = (4*h_sigma+1,4*h_sigma+1)
im_dx_sq = cv2.GaussianBlur(im_dx_sq, h_ksize, h_sigma)
im_dy_sq = cv2.GaussianBlur(im_dy_sq, h_ksize, h_sigma)
im_dx_dy = cv2.GaussianBlur(im_dx_dy, h_ksize, h_sigma)

''' Compute the Harris measure '''
kappa = 0.004
im_det = im_dx_sq * im_dy_sq - im_dx_dy * im_dx_dy
im_trace = im_dx_sq + im_dy_sq
im_harris = im_det - kappa * im_trace*im_trace

cv2.imwrite('res/harris.jpg', im_harris)

''' Renormalize the intensities into the 0..255 range '''
i_min = np.min(im_harris)
i_max = np.max(im_harris)
print("Before normalization the minimum and maximum harris measures are",
     i_min, i_max)
im_harris = 255 * (im_harris - i_min) / (i_max-i_min)

'''
Apply non-maximum thresholding using dilation, which requires the image
to be uint8.  Comparing the dilated image to the Harris image will preserve
only those locations that are peaks.
'''
max_dist = 2*sigma
kernel = np.ones((2*max_dist+1, 2*max_dist+1), np.uint8)
im_harris_dilate = cv2.dilate(im_harris, kernel)
im_harris[np.where(im_harris < im_harris_dilate)] = 0

cv2.imwrite('res/thresh.jpg', im_harris)


def harris_plot(im, harris_map, limiter):
    harris_keypoints = []
    keypoint_size = 4 * sigma
    for index, x in np.ndenumerate(harris_map):
        harris_keypoints.append(cv2.KeyPoint(index[1], index[0], _size=keypoint_size,
                                             _response=harris_map[index[0]][index[1]]))

    harris_keypoints = sorted(harris_keypoints, key=lambda keypoint: keypoint.response, reverse=True)
    harris_keypoints = harris_keypoints[:limiter]
    harris_keyed = cv2.drawKeypoints(im, harris_keypoints, None)
    cv2.imwrite('res/harris.jpg', harris_keyed)
    return harris_keypoints


def get_10tops(harris, orb):
    print('\nTop 10 Harris keypoints:\n')
    for i in range(0, 10):
        print(
            '{}: ({:.2f}, {:.2f}) {:.4f} {:.2f}\n'.format(i,
                                                          harris[i].pt[0],
                                                          harris[i].pt[1],
                                                          harris[i].response,
                                                          harris[i].size))
    print('\nTop 10 orb keypoints:\n')
    for i in range(0, 10):
        print(
            '{}: ({:.2f}, {:.2f}) {:.4f} {:.2f}\n'.format(i,
                                                          orb[i].pt[0],
                                                          orb[i].pt[1],
                                                          orb[i].response,
                                                          orb[i].size))
        
        
harris_keypoints = harris_plot(im, im_harris, 200)

orb = cv2.ORB_create()
orb_lst, des = orb.detectAndCompute(im, None)
orb_lst = sorted(orb_lst, key=lambda keypoint: keypoint.response, reverse=True)
orb_lst = orb_lst[:200]
orb_keyed = cv2.drawKeypoints(im, orb_lst, None)

get_10tops(harris_keypoints, orb_lst)





