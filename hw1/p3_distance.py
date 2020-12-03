import cv2
import numpy as np
import copy
import math
from os import listdir
from os.path import isfile, join
from os import walk
import os
from scipy.spatial import distance_matrix
import sys

'''parameters'''
S = 32
m = 256/S

'''parse inputs'''
image_folder = sys.argv[1]
f = os.listdir(image_folder)

color_stat = set()
hist_stat = set()

hists = dict()

'''read in images and calculate them for avg color vector and histograms'''
for filename in f:
    img = cv2.imread(image_folder + '/' + filename)
    reds = np.ravel(img[:, :, 0])
    greens = np.ravel(img[:, :, 1])
    blues = np.ravel(img[:, :, 2])
    '''calculate histogram and avg color vector'''
    one_hist, endpoints = np.histogramdd([reds, greens, blues], range=((0, 256), (0, 256), (0, 256)), bins=8)
    '''normalize the histograms'''
    one_hist = np.divide(one_hist, img.shape[0] * img.shape[1])
    avg = np.mean(img, axis=(0, 1))
    '''store results in a dict'''
    hists[filename] = [one_hist, avg]

'''compare between pairs of images'''
for i in range(len(f)):
    for j in range(len(f)):
        if i == j:
            continue
        filename1 = f[i]
        filename2 = f[j]
        tmp = sorted([filename1, filename2])
        hist1 = hists[filename1][0]
        hist2 = hists[filename2][0]
        avg1 = hists[filename1][1]
        avg2 = hists[filename2][1]
        '''calculate the L2norm between pairs of images and store them in sets'''
        hist_dist = np.linalg.norm(hist1 - hist2)
        hist_stat.add((hist_dist, tmp[0], tmp[1]))
        color_dist = np.linalg.norm(avg1 - avg2)
        color_stat.add((color_dist, tmp[0], tmp[1]))

'''sort the results from min to max distances'''
color_stat = sorted(color_stat)
hist_stat = sorted(hist_stat)

'''print log to console'''
print('Using distance between color averages.')
print('Closest pair is ({}, {})'.format(color_stat[0][1], color_stat[0][2]))
print('Minimum distance is {:.3f}'.format(color_stat[0][0]))
print('Furthest pair is ({}, {})'.format(color_stat[-1][1], color_stat[-1][2]))
print('Maximmum distance is {:.3f}'.format(color_stat[-1][0]))
print()
print('Using distance between histograms.')
print('Closest pair is ({}, {})'.format(hist_stat[0][1], hist_stat[0][2]))
print('Minimum distance is {:.3f}'.format(hist_stat[0][0]))
print('Furthest pair is ({}, {})'.format(hist_stat[-1][1], hist_stat[-1][2]))
print('Maximmum distance is {:.3f}'.format(hist_stat[-1][0]))
