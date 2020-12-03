import numpy as np
import random
import math
import sys

# parse input arguments
file_name = sys.argv[1]
samples = int(sys.argv[2])
tau = float(sys.argv[3])

# check if the seed is present
if sys.argv.__len__() == 5:
    seed = int(sys.argv[4])
    np.random.seed(seed)

# load input file with numpy
f = np.loadtxt(file_name, dtype=np.float64)


# calculate distance from a point to a line (line is basically two points)
def calc_dist(p3, p1, p2):
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)


# evaluate how many point are inliers and how many are outliers and calculate their distances to the current line
def evaluate(f_, tau_, p1, p2):
    tmp_dist = np.apply_along_axis(calc_dist, 1, f_, p1, p2)
    tmp_eval = np.abs(tmp_dist)
    tmp_eval = np.resize(tmp_eval, (tmp_eval.shape[0], 1))
    tmp_dist = np.resize(tmp_dist, (tmp_dist.shape[0], 1))
    eval_dist = np.hstack((tmp_eval, tmp_dist))
    inliers_ = np.where(abs(eval_dist[:, 0]) < abs(tau_))
    outliers_ = np.where(abs(eval_dist[:, 0]) >= abs(tau_))
    in_dists = np.take(tmp_dist, inliers_)
    out_dists = np.take(tmp_dist, outliers_)
    return np.size(inliers_), np.sum(in_dists), np.sum(out_dists)


# necessary recorders and containers
best_inlier_count = 0
best_abc = []
best_in_dist = 0
best_out_dist = 0
# loop through number of samples
for i in range(0, samples):
    sample = np.random.randint(0, f.shape[0], 2)
    idx0, idx1 = sample
    # check if indices are the same, if so skip
    if idx0 == idx1:
        continue

    # calculate the line expression from two points, simple algebra
    coord0, coord1 = f[idx0], f[idx1]

    a = coord1[1] - coord0[1]
    b = coord0[0] - coord1[0]
    c = coord1[0]*coord0[1] - coord0[0]*coord1[1]

    norm_factor = math.sqrt(a**2 + b**2)
    a /= norm_factor
    b /= norm_factor
    c /= norm_factor

    if c >= 0:
        a *= -1
        b *= -1
        c *= -1

    # call evaluate function to get evaluation of the current line
    inlier_count, in_dist_sum, out_dist_sum = evaluate(f, tau, coord0, coord1)
    if inlier_count > best_inlier_count:
        best_inlier_count = inlier_count
        best_in_dist = in_dist_sum
        best_out_dist = out_dist_sum
        best_abc = [a, b, c]
        print('Sample {}:'.format(i))
        print('indices ({},{})'.format(idx0, idx1))
        print('line ({:.3f},{:.3f},{:.3f})'.format(a, b, c))
        print('inliers {}'.format(inlier_count))
        print()

# print out the avg distances
print('avg inlier dist {:.3f}'.format(best_in_dist/best_inlier_count))
print('avg outlier dist {:.3f}'.format(best_out_dist/(f.shape[0] - best_inlier_count)))


