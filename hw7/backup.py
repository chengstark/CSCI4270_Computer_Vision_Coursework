import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from sklearn.cluster import KMeans
import shutil
from mrcnn import utils
import mrcnn.model as modellib
import coco
import skimage
import itertools


np.random.seed(100)
clusters = 2
number_of_kpts = 500
f_name = '000017'
grid_step = 50
use_grid = False
dist_thresh = 4


if os.path.exists(f_name):
    shutil.rmtree(f_name)
os.mkdir(f_name)


def SIFT_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    out = cv2.drawKeypoints(img, kp, None)
    cv2.imwrite('{}/sift_{}.jpg'.format(f_name, f_name), out)
    return kp


def spread_points(img, imname, step):
    xs = np.arange(step // 2, img.shape[0] + step // 2, step)
    ys = np.arange(step // 2, img.shape[1] + step // 2, step)
    y_grid, x_grid = np.meshgrid(xs, ys)
    x_grid = x_grid.flatten().reshape(x_grid.shape[0] * x_grid.shape[1], 1).astype(np.int64)
    y_grid = y_grid.flatten().reshape(y_grid.shape[0] * y_grid.shape[1], 1).astype(np.int64)
    grid = np.hstack((x_grid, y_grid))
    out = img.copy()
    for row in grid:
        x, y = row
        out = cv2.circle(out, (x, y), 1, (255, 0, 0), -1)
    cv2.imwrite('{}/dots.jpg'.format(f_name), out)
    return grid


def calc_optical_flow(prev_frame, curr_frame, kp_0, spread_grid=False):
    lk_params = dict(winSize=(30, 30),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if not spread_grid:
        pts_0 = np.asarray([kp_0[idx].pt for idx in range(0, len(kp_0))]).reshape(-1, 2).astype(np.float32)
    else:
        pts_0 = kp_0
        pts_0 = np.asarray(pts_0).astype(np.float32)
    pts_1, msk, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, pts_0, None, **lk_params)
    msk = msk.flatten()
    filtered_pts1 = pts_1[msk == 1]
    filtered_pts0 = pts_0[msk == 1]
    mask = np.zeros_like(prev_frame)
    color = np.random.randint(0, 255, (len(filtered_pts1), 3))

    curr_frame_cpy = curr_frame.copy()
    for i, (new, old) in enumerate(zip(filtered_pts1, filtered_pts0)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(curr_frame_cpy, (a, b), 1, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imwrite('{}/flow_{}.jpg'.format(f_name, f_name), img)

    return filtered_pts0, filtered_pts1


def calc_dist(row):
    x0, y0, x1, y1 = row
    dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    return dist


def check_camera_move(prev_pts, curr_pts, thresh=10):
    combined_pts = np.concatenate((prev_pts, curr_pts), axis=1)
    dists = np.apply_along_axis(calc_dist, 1, combined_pts)
    avg_dist = np.sum(dists) / combined_pts.shape[0]
    print(avg_dist)
    return avg_dist > thresh, avg_dist, dists


# calculate distance from a point to a line (line is basically two points)
def calc_point2line_dist(row, p3):
    p1 = np.asarray([row[0], row[1]])
    p2 = np.asarray([row[2], row[3]])
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)


# evaluate how many point are inliers and how many are outliers and calculate their distances to the current line
def evaluate(combined_pts, tau_, intersect):
    tmp_dist = np.apply_along_axis(calc_point2line_dist, 1, combined_pts, intersect)
    tmp_eval = np.abs(tmp_dist)
    tmp_eval = np.resize(tmp_eval, (tmp_eval.shape[0], 1))
    tmp_dist = np.resize(tmp_dist, (tmp_dist.shape[0], 1))
    eval_dist = np.hstack((tmp_eval, tmp_dist))
    inliers_ = np.where(abs(eval_dist[:, 0]) < abs(tau_))
    outliers_ = np.where(abs(eval_dist[:, 0]) >= abs(tau_))
    in_dists = np.take(tmp_dist, inliers_)
    out_dists = np.take(tmp_dist, outliers_)
    return np.size(inliers_), np.sum(in_dists), np.sum(out_dists)


def calc_line_expression(row):
    coord0 = [row[0], row[1]]
    coord1 = [row[2], row[3]]
    a = coord1[1] - coord0[1]
    b = coord0[0] - coord1[0]
    c = coord1[0] * coord0[1] - coord0[0] * coord1[1]

    norm_factor = math.sqrt(a ** 2 + b ** 2)
    a /= norm_factor
    b /= norm_factor
    c /= norm_factor

    if c >= 0:
        a *= -1
        b *= -1
        c *= -1
    return a, b, c


def calc_intersect(a0, b0, c0, a1, b1, c1):
    if a0 * b1 - a1 * b0 == 0:
        return [False]
    else:
        y = (a1 * c0 - a0 * c1) / (a0 * b1 - a1 * b0)
        x = -(b1 * y + c1) / a1
        return [True, int(x), int(y)]


def draw_foe(img1, intersect, pts):
    curr_out = img1.copy()
    color = np.random.randint(0, 255, (pts.shape[0], 3))
    for i in range(pts.shape[0]):
        row = pts[i]
        # cv2.line(curr_out, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), color[i].tolist(), 2)
    cv2.circle(curr_out, (int(intersect[0]), int(intersect[1])), 10, (255, 0, 0), -1)
    cv2.imwrite('{}/conected_curr_lines.jpg'.format(f_name), curr_out)


def ransac_foe(prev_frame, curr_frame, prev_pts, curr_pts, samples=500, bound=100):
    combined_pts = np.concatenate((prev_pts, curr_pts), axis=1)
    all_lines = np.apply_along_axis(calc_line_expression, 1, combined_pts)
    # necessary recorders and containers
    best_inlier_count = 0
    best_foe = None
    best_in_dist = 0
    best_out_dist = 0
    # loop through number of samples
    for i in range(0, samples):
        sample = np.random.randint(0, combined_pts.shape[0], 2)
        idx0, idx1 = sample
        # check if indices are the same, if so skip
        if idx0 == idx1:
            continue

        a0, b0, c0 = all_lines[idx0]
        a1, b1, c1 = all_lines[idx1]
        res = calc_intersect(a0, b0, c0, a1, b1, c1)
        if not res[0]:
            continue
        _, intersect_x, intersect_y = res
        intersect = np.asarray([intersect_x, intersect_y])

        # call evaluate function to get evaluation of the current line
        inlier_count, in_dist_sum, out_dist_sum = evaluate(combined_pts, bound, intersect)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_in_dist = in_dist_sum
            best_out_dist = out_dist_sum
            best_foe = intersect
            print('Sample {}:'.format(i))
            print('indices ({},{})'.format(idx0, idx1))
            print('intersect {}'.format(intersect))
            print('inliers {}'.format(inlier_count))
            print()

    # print out the avg distances
    print('avg inlier dist {:.3f}'.format(best_in_dist/best_inlier_count))
    print('avg outlier dist {:.3f}'.format(best_out_dist / (combined_pts.shape[0] - best_inlier_count)))
    print('best intersect {}'.format(intersect))
    draw_foe(curr_frame, intersect, combined_pts)


def calc_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_8UC1, 0, 1)
    sobel = np.absolute(sobel_x) + np.absolute(sobel_y)
    return sobel


def get_rgb(row, img):
    return img[row[0], row[1]]


def clustering(prev_img, curr_img, move_dists, avg_move_dist, prev_pts, curr_pts, filename):
    thresh = avg_move_dist
    # thresh = 20

    filtered_pts = prev_pts[move_dists > thresh]
    filtered_pts_int = filtered_pts.astype(np.int8)
    # filtered_pts_int *= 10

    filtered_dists = move_dists[move_dists > thresh]
    filtered_dists = filtered_dists.reshape((filtered_dists.shape[0], 1))
    featrue_vec = np.concatenate((filtered_pts*100, filtered_dists/20), axis=1)

    # color_pts = np.apply_along_axis(get_rgb, 1, filtered_pts_int, prev_img)
    # featrue_vec = np.concatenate((featrue_vec, color_pts), axis=1)
    print(featrue_vec.shape)

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(featrue_vec)
    color = np.random.randint(0, 255, (clusters, 3))
    prediction = kmeans.labels_
    cluster_out = prev_img.copy()
    # cluster_out = cnt_msk

    for group in range(0, clusters):
        this_group = filtered_pts[prediction == group]
        ul_x, ul_y = min(this_group[:, 0]), min(this_group[:, 1])
        lr_x, lr_y = max(this_group[:, 0]), max(this_group[:, 1])
        for i in range(0, this_group.shape[0]):
            cv2.circle(cluster_out, (this_group[i][0], this_group[i][1]), 5, color[group].tolist(), -1)
        cv2.rectangle(cluster_out,
                      (ul_x, ul_y),
                      (lr_x, lr_y),
                      color[group].tolist(), 2)

    cv2.imwrite('{}/cluster.jpg'.format(f_name), cluster_out)


def check_if_inside_box(row, box):
    x, y = row
    y0, x0, y1, x1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
    if ((x1+w1) < x2) or ((x2+w2) < x1):  # one on the left of another
        return False
    if (y1 < (y2-h2)) or (y2 < (y1-h1)):  # one on the upper of another
        return False
    return True


def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    area_inter = (min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2))
    area_r1 = w1*h1
    area_r2 = w2*h2
    area_union = area_r1 + area_r2 - area_inter
    iou = area_inter / area_union * 100
    return iou


def filter_roi_boxes(boxes):
    print('Originally had {} ROIs'.format(boxes.shape[0]))
    lst_boxes = boxes.tolist()
    merged_boxes = []
    failed_box_idx = set()
    combs = itertools.combinations(range(len(lst_boxes)), 2)
    for i, j, in combs:
        a = lst_boxes[i]
        b = lst_boxes[j]
        ay0, ax0, ay1, ax1 = a
        by0, bx0, by1, bx1 = b
        if overlap_check(ax0, ay0, ax1-ax0, ay1-ay0, bx0, by0, bx1-bx0, by1-by0):
            iou_ = iou(ax0, ay0, ax1-ax0, ay1-ay0, bx0, by0, bx1-bx0, by1-by0)
            if iou_ > 15:
                failed_box_idx.add(i)
                failed_box_idx.add(j)
                merged_boxes.append([min(ay0, by0), min(ax0, bx0), max(ay1, by1), max(ax1, bx1)])

    offset = 0
    failed_box_idx = sorted(failed_box_idx)
    for idx in failed_box_idx:
        lst_boxes.pop(idx - offset)
        offset += 1
    for box in merged_boxes:
        lst_boxes.append(box)
    print('Removed {} overlapped ROIs'.format(offset))
    print('{} ROIs left'.format(len(lst_boxes)))
    return lst_boxes


def mrcnn_clustering(prev_img, move_dists, avg_move_dist, prev_pts, boxes):
    filtered_pts = prev_pts[move_dists > dist_thresh]
    filtered_dists = move_dists[move_dists > dist_thresh]
    colors = np.random.randint(0, 255, (boxes.shape[0], 3))
    clustered = prev_img.copy()
    for i in range(boxes.shape[0]):
        box = boxes[i]
        y0, x0, y1, x1 = box
        pts_msk = np.apply_along_axis(check_if_inside_box, 1, filtered_pts, box)
        if len(np.where(pts_msk == 1)[0]) == 0:
            continue
        pts_in_box = filtered_pts[pts_msk]
        dists_in_box = filtered_dists[pts_msk]
        if np.size(dists_in_box) >= 1:
            clustered = cv2.rectangle(clustered, (x0, y0), (x1, y1), colors[i].tolist(), 2)
            for pt in pts_in_box:
                clustered = cv2.circle(clustered, (pt[0], pt[1]), 2, colors[i].tolist(), -1)

    cv2.imwrite('{}/mrcnn_cluster.jpg'.format(f_name), clustered)


def mrcnn_segmentation(im_path):
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir='', config=config)
    model.load_weights("mask_rcnn_coco.h5", by_name=True)
    image = skimage.io.imread(im_path)
    results = model.detect([image], verbose=1)
    r = results[0]

    masks = r['masks']
    boxes = r['rois']
    N = r['rois'].shape[0]
    masked_image = image.astype(np.uint32).copy()
    colors = np.random.randint(0, 255, (N, 3))
    alpha = 0.5
    for i in range(N):
        mask = masks[:, :, i]
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1,
                                             masked_image[:, :, c] * (1 - alpha) + alpha * colors[i][c] * 255,
                                             masked_image[:, :, c])
    masked_image = masked_image.astype(np.uint8)
    cv_masked_image = skimage.img_as_ubyte(masked_image)

    for box in boxes:
        y0, x0, y1, x1 = box
        cv_masked_image = cv2.rectangle(cv_masked_image, (x0, y0), (x1, y1), (255, 0, 0), 2)

    skimage.io.imsave('{}/seg.png'.format(f_name), masked_image)
    cv2.imwrite('{}/boxed_out.png'.format(f_name), cv_masked_image)

    return cv_masked_image, boxes


if __name__ == '__main__':
    im0 = cv2.imread('hw7data/{}_10.png'.format(f_name))
    im1 = cv2.imread('hw7data/{}_11.png'.format(f_name))
    # im0 = cv2.imread('hw7data/000169_10_0.png'.format(f_name))
    # im1 = cv2.imread('hw7data/000169_10.png'.format(f_name))

    cv2.imwrite('{}/im0.jpg'.format(f_name), im0)
    cv2.imwrite('{}/im1.jpg'.format(f_name), im1)

    if not use_grid:

        # gray = im0.copy()
        # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (13, 13), 0)
        # sobel_ = calc_sobel(gray)
        # ret, thresh = cv2.threshold(sobel_, 127, 255, 0)
        # contours, hierarchy = cv2.findContours(sobel_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # black_msk = np.zeros_like(im0)
        # cnt_msk = cv2.drawContours(black_msk, contours, -1, (0, 255, 0), 1)

        pts = SIFT_points(im0)
        pts = sorted(pts, key=lambda keypoint: keypoint.response, reverse=True)
        pts = pts[: number_of_kpts]
    else:
        pts = spread_points(im0, f_name, grid_step)

    prev_pts, curr_pts = calc_optical_flow(im0, im1, pts, use_grid)

    camera_moving, avg_move_dist, move_dists = check_camera_move(prev_pts, curr_pts, 7)

    if camera_moving:
        print('Camera Moved')
        # ransac_foe(im0, im1, prev_pts, curr_pts)
    else:
        print('Camera is not moving')

    clustering(im0, im1, move_dists, avg_move_dist, prev_pts, curr_pts, f_name)

    masked_im, roi_boxes = mrcnn_segmentation('hw7data/{}_10.png'.format(f_name))
    filtered_boxes = filter_roi_boxes(roi_boxes)
    mrcnn_clustering(im0, move_dists, avg_move_dist, prev_pts, np.asarray(filtered_boxes))






