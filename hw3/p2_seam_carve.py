import cv2
import numpy as np
import sys


'''function to calculate sobel values of the image'''
def calc_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel = np.absolute(sobel_x) + np.absolute(sobel_y)
    return sobel


'''function for tracing back up'''
def trace_back(last_min, M_, curr_row, rec):
    if curr_row < 0:
        return rec
    rec.append(last_min)
    '''special case on the side of the matrix'''
    if last_min == 0:
        idx_ = np.argmin(M_[curr_row - 1, 0:last_min + 2])
        idx_ = last_min + idx_
        trace_back(idx_, M_, curr_row - 1, rec)
    elif last_min == M_.shape[1] - 1:
        idx_ = np.argmin(M_[curr_row - 1, last_min - 1: last_min + 1])
        idx_ = last_min + (1 - idx_)
        trace_back(idx_, M_, curr_row - 1, rec)
    else:
        '''normal case'''
        idx_ = np.argmin(M_[curr_row - 1, last_min-1:last_min+2])
        idx_ = last_min + (idx_ - 1)
        trace_back(idx_, M_, curr_row - 1, rec)


'''function to apply the seam carv'''
def carv(sobel, ori_img, img, id, dir_):
    '''prepare container in_placer_record for storing updated weights'''
    '''from W [i, j] = e[i, j] + min􏰀W [i − 1, j − 1], W [i − 1, j], W [i − 1, j + 1]􏰁'''
    in_placer_record = sobel.copy()
    in_placer_record[:, 0] *= 0
    in_placer_record[:, ori_img.shape[1] - 1] *= 0
    in_placer_record[:, 0] += sys.maxsize
    in_placer_record[:, -1] += sys.maxsize
    '''loop through the rows'''
    for i in range(1, ori_img.shape[0]):
        last_line_weight = in_placer_record[i - 1, :]
        last_line_min = np.zeros((img.shape[1],))
        '''slice out the left/center/right part of the current row'''
        left = last_line_weight[:-2]
        right = last_line_weight[2:]
        center = last_line_weight[1:-1]
        '''apply boolean asthmatics for calculate the local minima of the current row'''
        last_line_min[1:-1] = (1-((center == right)*(center == left))) * (center <= right) * (center <= left) * center \
                              + ((1-((center == right)*(center == left)))*(right <= center) * (right <= left)) * right \
                              + ((1-((center == right)*(center == left)))*(left <= right) * (left <= center)) * left \
                              + ((center == right) * (center == left)) * left
        '''handle special cases of the first and the last column'''
        last_line_min[0] = min(last_line_weight[0], last_line_weight[1])
        last_line_min[-1] = min(last_line_weight[-1], last_line_weight[-2])
        '''apply the update'''
        in_placer_record[i, :] += last_line_min

    '''back track to the top of the image'''
    backtrack = []
    last_min_ = np.argmin(in_placer_record[ori_img.shape[0] - 1, :])
    trace_back(last_min_, in_placer_record, ori_img.shape[0] - 1, backtrack)
    '''prepare the mask'''
    mask = np.empty_like(img, dtype=np.bool)
    np.ndarray.fill(mask, True)
    '''calculate energy and prepare for printing outputs'''
    energy_sum = 0
    backtrack.reverse()
    tmp = ori_img.copy()
    for idx in range(backtrack.__len__()):
        x = idx
        y = backtrack[idx]
        '''mark the seam on the image'''
        tmp[x, y] = [0, 0, 255]
        mask[x, y] = False
        energy_sum += sobel_[x, y]
        if id == 0 or id == 1 or id == diff - 1:
            if idx == 0 or idx == backtrack.__len__()//2 or idx == backtrack.__len__()-1:
                if dir_ == 0:
                    print(x, ',', y)
                else:
                    print(y, ',', x)
    if id == 0:
        if dir_ == 0:
            cv2.imwrite('{}_seam.jpg'.format(img_name), tmp)
        else:
            tmp = np.rot90(tmp)
            tmp = np.rot90(tmp)
            tmp = np.rot90(tmp)
            cv2.imwrite('{}_seam.jpg'.format(img_name), tmp)
    '''convert single channel mask to 3-channel mask'''
    color_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    '''print energy'''
    if id == 0 or id == 1 or id == diff - 1:
        print('Energy of seam {}: {:.2f}'.format(id, energy_sum/backtrack.__len__()))

    '''apply mask and return for the next iteration'''
    '''apply on grayscale image'''
    new_gray_img = img.copy()
    new_gray_img = new_gray_img[mask]
    new_gray_img = new_gray_img.reshape((ori_img.shape[0], ori_img.shape[1] - 1))
    '''apply on color image'''
    new_color_img = ori_img.copy()
    new_color_img = new_color_img[color_mask]
    new_color_img = new_color_img.reshape((ori_img.shape[0], ori_img.shape[1] - 1, 3))

    return new_color_img, new_gray_img


if __name__ == '__main__':
    '''parse input arguments'''
    file_name = sys.argv[1]
    img_name = file_name[:-4]
    '''read in image and convert to grayscale'''
    ori_img_ = cv2.imread(file_name).astype(np.float32)
    gray = cv2.cvtColor(ori_img_, cv2.COLOR_BGR2GRAY)
    col, row = gray.shape
    '''determine direction of the image and calculate how many iteration is needed'''
    diff = 0
    if row > col:
        diff = row - col
        gray_for_carv = gray
        color_for_carv = ori_img_
        dir_ = 0
    else:
        diff = col - row
        '''rotate the image to landscape if it is portrait'''
        gray_for_carv = np.rot90(gray)
        color_for_carv = np.rot90(ori_img_)
        dir_ = 1
    '''loop through the diff and get the result'''
    count = 0
    for x in range(diff):
        '''print points to output'''
        if count == 0 or count == 1 or count == diff-1:
            print('\nPoints on seam {}:'.format(count))
            if dir_ == 0: print('vertical')
            else: print('horizontal')
        '''calculate sobel and apply seam carve'''
        sobel_ = calc_sobel(gray_for_carv)
        color_for_carv, gray_for_carv = carv(sobel_, color_for_carv, gray_for_carv, count, dir_)
        count += 1
    '''rotate the image back to portrait if it is portrait'''
    if row < col:
        color_for_carv = np.rot90(color_for_carv)
        color_for_carv = np.rot90(color_for_carv)
        color_for_carv = np.rot90(color_for_carv)

    '''write final result of the carve'''
    cv2.imwrite('{}_final.jpg'.format(img_name), color_for_carv)
