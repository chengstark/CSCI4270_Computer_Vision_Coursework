import cv2
import numpy as np
import math
import time
import sys
import os


t = 4
bw = 4
bh = 4

# all labels for later iteration
labels = os.listdir('hw6_data/train')[1:]


# calculate histogram
def calc_hist(img, bins):
    w, h, _ = img.shape
    pixels = img.reshape(w * h, 3)
    hist, _ = np.histogramdd(pixels, (bins, bins, bins))
    hist = hist.reshape((1, bins * bins * bins))
    return hist


# calculate feature vector
def calc_feature_vector(img):
    w, h, d = img.shape
    # calculate delta
    delta_w = math.floor(w / (bw + 1))
    delta_h = math.floor(h / (bh + 1))
    # prepare the grid of coordinates
    xs = np.asarray(range(0, bw))
    ys = np.asarray(range(0, bh))
    xs *= delta_w
    ys *= delta_h
    y_grid, x_grid = np.meshgrid(xs, ys)
    x_grid = x_grid.flatten().reshape(x_grid.shape[0]*x_grid.shape[1], 1).astype(np.int64)
    y_grid = y_grid.flatten().reshape(y_grid.shape[0]*y_grid.shape[1], 1).astype(np.int64)
    grid = np.hstack((x_grid, y_grid))
    # calculate feature of the first slice of the image
    first_slice = img[grid[0][0]:grid[0][0] + 2 * delta_w, grid[0][1]:grid[0][1] + 2 * delta_h, :]
    first_slice_hist = calc_hist(first_slice, t)
    feature_vector = first_slice_hist
    # calculate feature of the rest of the image
    for i in range(1, len(grid)):
        point = grid[i]
        slice = img[point[0]:point[0] + 2 * delta_w, point[1]:point[1] + 2 * delta_h, :]
        slice_hist = calc_hist(slice, t)
        feature_vector = np.concatenate((feature_vector, slice_hist), axis=1)
    # return the feature vector of the image
    return feature_vector


# calculate all feature vectors and combining feature vectors with their label
# labels are converted from string to correspond int index
def combine_label(folder_name):
    all_feature_vectors = []
    # loop through all labels
    for label_idx in range(len(labels)):
        print('starting label {} - {}'.format(folder_name, labels[label_idx]))
        folder_path = 'hw6_data/{}/{}/'.format(folder_name, labels[label_idx])
        image_names = os.listdir(folder_path)
        # loop through all images in the folder
        for image_name in image_names:
            img = cv2.imread(folder_path + image_name)
            feature_vector = calc_feature_vector(img)
            feature_vector = feature_vector.tolist()[0]
            # features + label
            feature_vector.append(label_idx)
            all_feature_vectors.append(feature_vector)
        print('finished label {} - {}'.format(folder_name, labels[label_idx]))

    all_feature_vectors = np.asarray(all_feature_vectors)
    np.savetxt("{}_vectors.txt".format(folder_name), all_feature_vectors)
    print('{} feature vector saved'.format(folder_name))


if __name__ == '__main__':
    # generate feature vectors + label for train set and test set
    combine_label("train")
    combine_label("test")
