import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import hdbscan
import shutil
import mrcnn.model as modellib
import coco
import skimage
import itertools
import tensorflow as tf
# from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(1)
clusters = 3
number_of_kpts = 1000
f_name = '000151'
grid_step = 20
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
        out = cv2.circle(out, (int(x), int(y)), 1, (255, 0, 0), -1)
    cv2.imwrite('{}/dots.jpg'.format(f_name), out)
    return grid


def draw_flow(curr_frame, combined_pts):
    curr_frame_cpy = curr_frame.copy()
    color = np.random.randint(0, 255, (len(combined_pts), 3))
    for i in range(combined_pts.shape[0]):
        row = combined_pts[i]
        cv2.line(curr_frame_cpy, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), color[i].tolist(), 2)
    return curr_frame_cpy


def calc_optical_flow(prev_frame, curr_frame, kp_0, draw=False):
    lk_params = dict(winSize=(30, 30),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    pts_0 = np.asarray(kp_0).astype(np.float32)
    pts_1, msk, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, pts_0, None, **lk_params)
    msk = msk.flatten()
    filtered_pts1 = pts_1[msk == 1]
    filtered_pts0 = pts_0[msk == 1]
    combined_pts = np.concatenate((filtered_pts0, filtered_pts1), axis=1)
    if draw:
        out = draw_flow(curr_frame, combined_pts)
        cv2.imwrite('{}/flow_{}.jpg'.format(f_name, f_name), out)

    return filtered_pts0, filtered_pts1


def calc_dist(row):
    x0, y0, x1, y1 = row
    dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    return dist


def check_camera_move(prev_pts, curr_pts, thresh=10, boxes=None):
    combined_pts = np.concatenate((prev_pts, curr_pts), axis=1)
    dists = np.apply_along_axis(calc_dist, 1, combined_pts)
    avg_dist = np.sum(dists) / dists.shape[0]
    if boxes is not None:
        out_msk = np.zeros((dists.shape[0]))
        for box in boxes:
            msk = np.apply_along_axis(check_if_inside_box, 1, prev_pts, box)
            out_msk[msk == 1] = 1
        out_box_dists = dists[out_msk == 0]
        out_avg_dist = np.sum(out_box_dists) / out_box_dists.shape[0]
        return out_avg_dist > thresh, out_avg_dist, dists
    else:
        return False, avg_dist, dists


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

    return a, b, c


def calc_intersect(a0, b0, c0, a1, b1, c1):
    if a0 * b1 - a1 * b0 == 0:
        return [False]
    rhs = np.asarray([[a0, b0], [a1, b1]])
    lhs = np.asarray([-c0, -c1])
    intersect = np.linalg.solve(rhs, lhs)
    return [True, intersect[0], intersect[1]]


def draw_foe(img1, intersect, pts):
    curr_out = img1.copy()
    color = np.random.randint(0, 255, (pts.shape[0], 3))
    for i in range(pts.shape[0]):
        row = pts[i]
        cv2.line(curr_out, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), color[i].tolist(), 2)
    cv2.circle(curr_out, (int(intersect[0]), int(intersect[1])), 10, (255, 0, 0), -1)
    cv2.imwrite('{}/conected_curr_lines.jpg'.format(f_name), curr_out)


def surface_fit(feature):
    def poly_matrix(x, y, order=2):
        """ generate Matrix use with lstsq """
        ncols = (order + 1) ** 2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order + 1), range(order + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x ** i * y ** j
        return G
    ordr = 1  # order of polynomial
    x, y, z = feature.T
    x, y = x - x[0], y - y[0]  # this improves accuracy

    # make Matrix:
    G = poly_matrix(x, y, ordr)
    # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, z)[0]

    # Evaluate it on a grid...
    nx, ny = 30, 30
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                         np.linspace(y.min(), y.max(), ny))
    GG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    zz = np.reshape(np.dot(GG, m), xx.shape)



    # Plotting (see http://matplotlib.org/examples/mplot3d/custom_shaded_3d_surface.html):
    fg, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    rgb = ls.shade(zz, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    ax.plot3D(x, y, z, "o")

    fg.canvas.draw()
    plt.show()


def ransac_foe(prev_frame, curr_frame, prev_pts, curr_pts, samples=500, bound=10):
    combined_pts = np.concatenate((prev_pts, curr_pts), axis=1)
    slopes = np.apply_along_axis(calc_slope, 1, combined_pts)
    slopes = slopes.reshape((slopes.shape[0], 1))

    # coord_and_slope = np.concatenate((prev_pts, slopes), axis=1)
    # surface_fit(coord_and_slope)

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
            print('intersect {}'.format(best_foe))
            print('inliers {}'.format(inlier_count))
            print()

    # print out the avg distances
    print('avg inlier dist {:.3f}'.format(best_in_dist/best_inlier_count))
    print('avg outlier dist {:.3f}'.format(best_out_dist / (combined_pts.shape[0] - best_inlier_count)))
    print('best intersect {}'.format(best_foe))
    draw_foe(curr_frame, best_foe, combined_pts)
    return best_foe


def filter_by_slope(row, foe, slope_diff_thresh=0.2):
    foe_x, foe_y = foe
    x0, y0, x1, y1 = row
    actual_slope = (y1 - y0) / (x1 - x0)
    expected_slope = (foe_y - y0) / (foe_x - x0)
    diff = abs(expected_slope - actual_slope)
    if diff > slope_diff_thresh:
        return False
    else:
        return True


def calc_slope(row):
    x0, y0, x1, y1 = row
    actual_slope = (y1 - y0) / (x1 - x0)
    return actual_slope


def calc_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_8UC1, 0, 1)
    sobel = np.absolute(sobel_x) + np.absolute(sobel_y)
    return sobel


def get_rgb(row, img):
    return img[row[0], row[1]]


def generate_heat_map_color(row, max_):
    return [row / max_ * 255, row / max_ * 255, 0]


def kmean_clustering(prev_img, curr_img, move_dists, grid_dists, grid_pts, prev_pts, curr_pts, foe):
    combined_pts = np.concatenate((prev_pts, curr_pts), axis=1)
    # slope_msk = np.apply_along_axis(filter_by_slope, 1, combined_pts, foe)
    p2l_dists = np.apply_along_axis(calc_point2line_dist, 1, combined_pts, foe)
    slope_msk = p2l_dists < 50
    filtered_pts = prev_pts[slope_msk]
    filtered_combined = combined_pts[slope_msk]
    filtered_move_dists = move_dists[slope_msk]

    aft_slope_flow = draw_flow(curr_img, filtered_combined)
    cv2.imwrite('{}/aft_slope.jpg'.format(f_name), aft_slope_flow)

    print('Slope filtering removed {} points, now {} left'.
          format(prev_pts.shape[0] - filtered_pts.shape[0], filtered_pts.shape[0]))

    dist_from_foe = np.sqrt(np.square(filtered_pts[:, 0] - foe[0] + np.square((filtered_pts[:, 1] - foe[1]))))
    grid_pts = np.asarray(grid_pts)
    grid_dist_from_foe = np.sqrt(np.square(grid_pts[:, 0] - foe[0] + np.square((grid_pts[:, 1] - foe[1]))))
    grid_normazlied_move_dists = grid_dists / grid_dist_from_foe * 100
    normazlied_move_dists = filtered_move_dists / dist_from_foe * 100
    normazlied_move_dists = normazlied_move_dists.reshape((normazlied_move_dists.shape[0], 1))
    normalized_colors = np.apply_along_axis(generate_heat_map_color, 1, normazlied_move_dists, np.max(normazlied_move_dists))

    print('Average values SIFT {}, GRID {}'.format(np.average(normazlied_move_dists), np.average(grid_normazlied_move_dists)))

    plt.plot(normazlied_move_dists)
    plt.savefig('{}/plot.jpg'.format(f_name))
    plt.clf()

    coefs = np.polyfit(grid_dist_from_foe, grid_move_dists, 2)
    tmp_X = np.arange(0, int(np.max(grid_dist_from_foe)), 1)
    ffit = np.poly1d(coefs)
    plt.plot(tmp_X, ffit(tmp_X), 'b')
    expected_move_dists = ffit(dist_from_foe)
    diff = filtered_move_dists - expected_move_dists
    diff = np.abs(diff)
    plt.scatter(dist_from_foe, filtered_move_dists, c='r')
    plt.savefig('{}/fit.jpg'.format(f_name))

    tmp = curr_img.copy()
    for i in range(filtered_combined.shape[0]):
        row = filtered_combined[i]
        if normazlied_move_dists[i][0] > np.average(normazlied_move_dists):
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        # color = normalized_colors[i]
        cv2.line(tmp, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (int(color[0]), int(color[1]), int(color[2])), 2)
    cv2.imwrite('{}/heat.jpg'.format(f_name), tmp)

    print('Filtering with moving distance threshold {}'.format(np.average(grid_move_dists)))

    msk = normazlied_move_dists > np.average(normazlied_move_dists)
    # msk = normazlied_move_dists > np.average(grid_normazlied_move_dists)
    msk = msk.flatten()
    # msk = diff > np.average(diff)
    # msk = filtered_move_dists > np.average(move_dists)
    filtered_pts = filtered_pts[msk]
    filtered_combined = filtered_combined[msk]
    filtered_move_dists = filtered_move_dists[msk]
    print('After threshold, {} pts left'.format(filtered_pts.shape[0]))
    aft_thresh = draw_flow(prev_img, filtered_combined)
    cv2.imwrite('{}/aft_thresh.jpg'.format(f_name), aft_thresh)

    # slopes = np.apply_along_axis(calc_slope, 1, filtered_combined)
    filtered_move_dists = filtered_move_dists.reshape((filtered_move_dists.shape[0], 1))
    slopes = np.apply_along_axis(calc_point2line_dist, 1, filtered_combined, foe)
    slopes = slopes.reshape((slopes.shape[0], 1))

    feature_vec = np.concatenate((filtered_pts, filtered_move_dists), axis=1)
    feature_vec = np.concatenate((feature_vec, slopes), axis=1)

    # dbscan = DBSCAN(min_samples=2, eps=8)
    # dbscan_clusters = dbscan.fit_predict(feature_vec)
    # prediction = dbscan.labels_

    # kmeans = KMeans(n_clusters=clusters, random_state=0).fit(feature_vec)
    # prediction = kmeans.labels_

    print('HDBSCAN')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    prediction = clusterer.fit_predict(feature_vec)

    unique_clusters = np.unique(prediction)
    # cluster_out = cnt_msk
    boxes = []
    for cluster_class in unique_clusters:
        if cluster_class == -1:
            continue
        this_cluster = filtered_pts[prediction == cluster_class]
        ul_x, ul_y = min(this_cluster[:, 0]), min(this_cluster[:, 1])
        lr_x, lr_y = max(this_cluster[:, 0]), max(this_cluster[:, 1])
        if ul_x == lr_x or ul_y == lr_y:
            continue
        boxes.append([ul_y, ul_x, lr_y, lr_x])
    return boxes, filtered_pts, filtered_move_dists.flatten()


def check_if_inside_box(row, box):
    x, y = row
    y0, x0, y1, x1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def is_overlapped(x1, y1, w1, h1, x2, y2, w2, h2):
    center_1 = [x1 + (w1 / 2), y1 + (h1 / 2)]
    center_2 = [x2 + (w2 / 2), y2 + (h2 / 2)]
    diff = [abs(center_1[0] - center_2[0]), abs(center_1[1] - center_2[1])]
    if diff[0] > ((w1 / 2) + (w2 / 2)) or diff[1] > ((h1 / 2) + (h2 / 2)):
        return False
    else:
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
    combos = itertools.combinations(range(len(lst_boxes)), 2)
    for i, j, in combos:
        a = lst_boxes[i]
        b = lst_boxes[j]
        ay0, ax0, ay1, ax1 = a
        by0, bx0, by1, bx1 = b
        if is_overlapped(ax0, ay0, ax1 - ax0, ay1 - ay0, bx0, by0, bx1 - bx0, by1 - by0):
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


def combine_boxes(kmean_boxes, mrcnn_boxes):
    tmp_kmean_boxes = kmean_boxes.copy()
    kmean_overlapped_idxes = set()
    for i in range(len(kmean_boxes)):
        a = kmean_boxes[i]
        for b in mrcnn_boxes:
            ay0, ax0, ay1, ax1 = a
            by0, bx0, by1, bx1 = b
            if is_overlapped(ax0, ay0, ax1 - ax0, ay1 - ay0, bx0, by0, bx1 - bx0, by1 - by0):
                kmean_overlapped_idxes.add(i)

    offset = 0
    for idx in kmean_overlapped_idxes:
        tmp_kmean_boxes.pop(idx - offset)
        offset += 1
    return tmp_kmean_boxes + mrcnn_boxes


def mrcnn_clustering(prev_img, move_dists, prev_pts, boxes):
    print('MRCNN clustering with {} pts'.format(prev_pts.shape[0]))
    colors = np.random.randint(0, 254, (boxes.shape[0], 3))
    clustered = prev_img.copy()
    count = 0

    for i in range(boxes.shape[0]):
        box = boxes[i]
        y0, x0, y1, x1 = box
        pts_msk = np.apply_along_axis(check_if_inside_box, 1, prev_pts, box)
        pts_in_box = prev_pts[pts_msk]
        if pts_in_box.shape[0] >= 5:
            clustered = cv2.rectangle(clustered, (int(x0), int(y0)), (int(x1), int(y1)), colors[i].tolist(), 2)
            clustered = cv2.putText(clustered, 'ROI IDX {}'.format(count),
                                    (int(x0)+2, int(y0)+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)
            clustered = cv2.putText(clustered, '{} PTS'.format(pts_in_box.shape[0]),
                                    (int(x0)+2, int(y0)+24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)

            for pt in pts_in_box:
                clustered = cv2.circle(clustered, (int(pt[0]), int(pt[1])), 3, colors[i].tolist(), -1)
            count += 1
        else:
            clustered = cv2.rectangle(clustered, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)
            clustered = cv2.putText(clustered, 'ROI IDX {}'.format(count),
                                    (int(x0) + 2, int(y0) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            clustered = cv2.putText(clustered, 'MRCNN',
                                    (int(x0) + 2, int(y0) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            for pt in pts_in_box:
                clustered = cv2.circle(clustered, (int(pt[0]), int(pt[1])), 3, colors[i].tolist(), -1)
            count += 1
    print('FINAL result contains {} ROIs'.format(count))
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
    boxes_ = r['rois']
    ids = r['class_ids']
    print('IDS: ', ids)

    id_pool = [1, 2, 3, 4, 6, 8]
    boxes = []
    for i in range(len(ids)):
        id_ = ids[i]
        if id_ in id_pool:
            boxes.append(boxes_[i])

    N = r['rois'].shape[0]
    masked_image = image.astype(np.uint32).copy()
    colors = np.random.randint(0, 255, (N, 3))
    alpha = 0.5
    all_masks = np.zeros((image.shape[0], image.shape[1]))
    for i in range(N):
        mask_ = masks[:, :, i]
        all_masks[mask_ == 1] = 1
        for c in range(3):
            masked_image[:, :, c] = np.where(mask_ == 1,
                                             masked_image[:, :, c] * (1 - alpha) + alpha * colors[i][c] * 255,
                                             masked_image[:, :, c])
    masked_image = masked_image.astype(np.uint8)
    cv_masked_image = skimage.img_as_ubyte(masked_image)

    for box in boxes:
        y0, x0, y1, x1 = box
        cv_masked_image = cv2.rectangle(cv_masked_image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

    skimage.io.imsave('{}/seg.png'.format(f_name), masked_image)

    return cv_masked_image, all_masks, np.asarray(boxes)


# def add_dense_pts(existing_pts):


if __name__ == '__main__':
    im0 = cv2.imread('hw7data/{}_10.png'.format(f_name))
    im1 = cv2.imread('hw7data/{}_11.png'.format(f_name))

    # im0 = cv2.imread('hw7data/000169_10_0.png'.format(f_name))
    # im1 = cv2.imread('hw7data/000169_10.png'.format(f_name))

    cv2.imwrite('{}/im0.jpg'.format(f_name), im0)
    cv2.imwrite('{}/im1.jpg'.format(f_name), im1)

    masked_im, mask, mrcnn_boxes = mrcnn_segmentation('hw7data/{}_10.png'.format(f_name))
    cv2.imwrite('{}/msk.jpg'.format(f_name), mask*255)

    mask_xs = np.where(mask == 1)[1]
    mask_ys = np.where(mask == 1)[0]
    mask_xs = mask_xs.reshape((mask_xs.shape[0], 1))
    mask_ys = mask_ys.reshape((mask_ys.shape[0], 1))
    mask_pts = np.concatenate((mask_xs, mask_ys), axis=1)
    down_sampler = np.random.choice([True, False], mask_pts.shape[0], p=[0.01, 0.99])
    mask_pts = mask_pts[down_sampler]

    if not use_grid:
        pts = SIFT_points(im0)
        pts = sorted(pts, key=lambda keypoint: keypoint.response, reverse=True)
        pts = pts[: number_of_kpts]
        pts = [pts[idx].pt for idx in range(0, len(pts))]
    else:
        pts = spread_points(im0, f_name, grid_step)
        pts = pts.tolist()

    # pts = mask_pts.tolist() + pts
    prev_pts, curr_pts = calc_optical_flow(im0, im1, pts, draw=True)

    _, avg_move_dist, move_dists = check_camera_move(prev_pts, curr_pts, 7)

    grid_pts = spread_points(im0, f_name, grid_step)
    grid_pts = grid_pts.tolist()
    grid_move_prev_pts, grid_move_curr_pts = calc_optical_flow(im0, im1, grid_pts, draw=False)
    camera_moving, grid_avg_move_dist, grid_move_dists = check_camera_move(grid_move_prev_pts, grid_move_curr_pts, 7, mrcnn_boxes)

    mask_prev_pts, mask_curr_pts = calc_optical_flow(im0, im1, mask_pts, draw=False)

    if camera_moving:
        print('Camera Moved')
    else:
        print('Camera is not moving')

    FOE = ransac_foe(im0, im1, prev_pts, curr_pts)
    kmean_boxes, filtered_pts, filtered_dists = kmean_clustering(im0, im1, move_dists, grid_move_dists, grid_move_prev_pts, prev_pts, curr_pts, FOE)

    filtered_boxes = filter_roi_boxes(mrcnn_boxes)
    combined_boxes = combine_boxes(kmean_boxes, filtered_boxes)

    mrcnn_clustering(im0, filtered_dists, filtered_pts, np.asarray(combined_boxes))






