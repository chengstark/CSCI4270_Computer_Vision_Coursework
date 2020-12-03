import numpy as np
import cv2
import sys


'''apply threshold to the image'''
def threshold(mag_, thresh_):
    mag_[np.where(mag_ < thresh_)] = 0
    mag_[np.where(mag_ >= thresh_)] = 255
    return mag_


'''NMS operations'''
def nms(img_, dir_map_, im_mag_):
    mark_helper = np.zeros((dir_map_.shape[0], dir_map_.shape[1]))
    '''the center of the original direction matrix'''
    new_dir_map = dir_map_[1:-1, 1:-1]
    new_mag = im_mag_[1:-1, 1:-1]
    ''''This section is to use matrix slicing and compare these slices in different directions to determine local max'''
    '''mark the max element with 1'''
    '''direction code 1'''
    mark_helper[1:-1, 1:-1][(new_dir_map == 1) & (im_mag_[0:-2, 1:-1] <= new_mag) & (im_mag_[2:im_mag_.shape[0], 1:-1]
                                                                                     <= new_mag)] = 1
    mark_helper[0:-2, 1:-1][(new_dir_map == 1) & (im_mag_[0:-2, 1:-1] == new_mag) & (im_mag_[2:im_mag_.shape[0], 1:-1]
                                                                                     <= new_mag)] = 1
    mark_helper[2:im_mag_.shape[0], 1:-1][(new_dir_map == 1) & (im_mag_[0:-2, 1:-1] <= new_mag) &
                                          (im_mag_[2:im_mag_.shape[0], 1:-1] == new_mag)] = 1
    '''direction code 2'''
    mark_helper[1:-1, 1:-1][(new_dir_map == 2) & (im_mag_[0:-2, 0:-2] <= new_mag) &
                            (im_mag_[2:im_mag_.shape[0], 2:im_mag_.shape[1]] <= new_mag)] = 1
    mark_helper[0:-2, 0:-2][(new_dir_map == 2) & (im_mag_[0:-2, 0:-2] == new_mag) &
                            (im_mag_[2:im_mag_.shape[0], 2:im_mag_.shape[1]] <= new_mag)] = 1
    mark_helper[2:im_mag_.shape[0], 2:im_mag_.shape[1]][(new_dir_map == 2) & (im_mag_[0:-2, 0:-2] <= new_mag) &
                                                    (im_mag_[2:im_mag_.shape[0], 2:im_mag_.shape[1]] == new_mag)] = 1
    '''direction code 3'''
    mark_helper[1:-1, 1:-1][(new_dir_map == 3) & (im_mag_[1:-1, 0:-2] <= new_mag) & (im_mag_[1:-1, 2: im_mag_.shape[1]]
                                                                                     <= new_mag)] = 1
    mark_helper[1:-1, 2: im_mag_.shape[1]][(new_dir_map == 3) & (im_mag_[1:-1, 0:-2] <= new_mag) &
                                           (im_mag_[1:-1, 2: im_mag_.shape[1]] == new_mag)] = 1
    mark_helper[1:-1, 0:-2][(new_dir_map == 3) & (im_mag_[1:-1, 0:-2] == new_mag) &
                            (im_mag_[1:-1, 2: im_mag_.shape[1]] <= new_mag)] = 1
    '''direction code 4'''
    mark_helper[1:-1, 1:-1][(new_dir_map == 4) & (im_mag_[2:im_mag_.shape[0], 0:-2] <= new_mag) &
                            (im_mag_[0:-2, 2:im_mag_.shape[1]] <= new_mag)] = 1
    mark_helper[0:-2, 2:im_mag_.shape[1]][(new_dir_map == 4) & (im_mag_[2:im_mag_.shape[0], 0:-2] <= new_mag) &
                                          (im_mag_[0:-2, 2:im_mag_.shape[1]] == new_mag)] = 1
    mark_helper[2:im_mag_.shape[0], 0:-2][(new_dir_map == 4) & (im_mag_[2:im_mag_.shape[0], 0:-2] == new_mag) &
                                          (im_mag_[0:-2, 2:im_mag_.shape[1]] <= new_mag)] = 1

    '''translate the magnitude from original image magnitude by using the markers, now we have a map with NMSed'''
    mag = np.zeros_like(img_)
    mag[mark_helper == 1] = im_mag_[mark_helper == 1]

    '''apply the > 1 threshold'''
    aft1_mag = np.zeros_like(mag)
    aft1_mag[mag > 1] = mag[mag > 1]
    num_aft_non_max = len(np.where(mag != 0)[0])
    num_aft_1 = len(np.where(aft1_mag != 0)[0])

    '''calculate parameters'''
    avg = np.average(aft1_mag[aft1_mag != 0])
    std = np.std(aft1_mag[aft1_mag != 0])
    thresh = min(avg+0.5*std, 30/sigma)

    '''apply threshold'''
    result = threshold(aft1_mag, thresh)
    num_aft_thresh = len(np.where(result != 0)[0])

    print("Number after non-maximum: {}".format(num_aft_non_max))
    print("Number after 1.0 threshold: {}".format(num_aft_1))
    print("mu: {:.2f}".format(avg))
    print("s: {:.2f}".format(std))
    print("Threshold: {:.2f}".format(thresh))
    print("Number after threshold: {}".format(num_aft_thresh))

    '''write to output'''
    cv2.imwrite('{}_thr{}'.format(img_name, extension), result)


if __name__ == '__main__':
    sigma = float(sys.argv[1])
    file_name = sys.argv[2]
    img_name = file_name[:-4]
    extension = file_name[-4:]

    '''read in image as gray scale'''
    img = cv2.imread(file_name, 0)
    img = img.astype(np.float)
    kernel_size = (int(4 * sigma + 1), int(4 * sigma + 1))

    '''apply gaussian blur'''
    blur = cv2.GaussianBlur(img, kernel_size, sigma)
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    im_dx = cv2.filter2D(blur, -1, kx)
    im_dy = cv2.filter2D(blur, -1, ky)

    '''calculate magnitude'''
    im_mag = np.sqrt(im_dx ** 2 + im_dy ** 2)
    '''normalize to 0-255'''
    im_grd = im_mag / np.max(im_mag) * 255
    im_deg_dir = np.arctan2(im_dy, im_dx)
    cv2.imwrite('{}_grd{}'.format(img_name, extension), im_grd)

    '''containers for record color and direction'''
    color_map = np.zeros((img.shape[0], img.shape[1], 3))
    dir_map = np.zeros((img.shape[0], img.shape[1]))

    '''record color with radian conditions'''
    '''red'''
    color_map[(im_deg_dir >= - np.pi) & (im_deg_dir < -7 * np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (0, 0, 255)
    color_map[(im_deg_dir >= - np.pi / 8) & (im_deg_dir < np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (0, 0, 255)
    color_map[(im_deg_dir >= 7 * np.pi / 8) & (im_deg_dir <= np.pi) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (0, 0, 255)
    '''green'''
    color_map[(im_deg_dir >= -7 * np.pi / 8) & (im_deg_dir < -5 * np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (0, 255, 0)
    color_map[(im_deg_dir >= np.pi / 8) & (im_deg_dir < 3 * np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (0, 255, 0)
    '''blue'''
    color_map[(im_deg_dir >= -5 * np.pi / 8) & (im_deg_dir < -3 * np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (255, 0, 0)
    color_map[(im_deg_dir >= 3 * np.pi / 8) & (im_deg_dir < 5 * np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (255, 0, 0)
    '''white'''
    color_map[(im_deg_dir >= 5 * np.pi / 8) & (im_deg_dir < 7 * np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (255, 255, 255)
    color_map[(im_deg_dir >= -3 * np.pi / 8) & (im_deg_dir <= - np.pi / 8) & ((im_dx + im_dy) != 0) & (im_mag > 1)] \
        = (255, 255, 255)

    '''translate direction into 1, 2, 3, 4'''
    dir_map[(im_deg_dir >= -5 * np.pi / 8) & (im_deg_dir < -3 * np.pi / 8) & ((im_dx + im_dy) != 0)] = 1
    dir_map[(im_deg_dir >= 3 * np.pi / 8) & (im_deg_dir < 5 * np.pi / 8) & ((im_dx + im_dy) != 0)] = 1
    dir_map[(im_deg_dir >= -7 * np.pi / 8) & (im_deg_dir < -5 * np.pi / 8) & ((im_dx + im_dy) != 0)] = 2
    dir_map[(im_deg_dir >= np.pi / 8) & (im_deg_dir < 3 * np.pi / 8) & ((im_dx + im_dy) != 0)] = 2
    dir_map[(im_deg_dir >= - np.pi) & (im_deg_dir < -7 * np.pi / 8) & ((im_dx + im_dy) != 0)] = 3
    dir_map[(im_deg_dir >= - np.pi / 8) & (im_deg_dir < np.pi / 8) & ((im_dx + im_dy) != 0)] = 3
    dir_map[(im_deg_dir >= 7 * np.pi / 8) & (im_deg_dir <= np.pi) & ((im_dx + im_dy) != 0)] = 3
    dir_map[(im_deg_dir >= 5 * np.pi / 8) & (im_deg_dir < 7 * np.pi / 8) & ((im_dx + im_dy) != 0)] = 4
    dir_map[(im_deg_dir >= -3 * np.pi / 8) & (im_deg_dir < - np.pi / 8) & ((im_dx + im_dy) != 0)] = 4

    '''write the direction image'''
    cv2.imwrite('{}_dir{}'.format(img_name, extension), color_map)

    '''apply NMS'''
    nms(img, dir_map, im_mag)

