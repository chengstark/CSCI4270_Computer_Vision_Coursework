import cv2
import numpy as np
import os
import itertools
import sys
import shutil

dir_ = sys.argv[1]
out_path = sys.argv[2]

decision_dict = dict()      # dictionary to store all matched pairs
path_dict = dict()          # dictionary to store all connected edges of the graph (will explain later)
                            # solving multi-image mosaic as travelling sales man problem
matched_record = set()      # record all images that are matched with other images
multi_mosaic = True        # flag to determine whether it is a multi image mosaic

# prepare the log file (I use if for my own output folders)
if multi_mosaic:
    file = open('{}/log.txt'.format(out_path), 'w+')
    file.truncate()
else:
    file = open('{}/log.txt'.format(out_path), 'w+')
    file.truncate()


# helper function to load all images in a directory
def load_helper(dir):
    im_lst = os.listdir(dir)
    im_lst = [k for k in im_lst if k.lower().endswith('jpg') or k.lower().endswith('png')]
    return im_lst


# draw matched images with SIFT lines side by side
def draw_side_by_side(im0, im1, sift_lst0, sift_lst1, pair, mask, good, name):
    matchesMask = mask.ravel().tolist()
    # parameters for drawing, filter out outlier matches
    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    linked = cv2.drawMatches(im0, sift_lst0, im1, sift_lst1, good, None, **draw_params)
    # write image to output
    file.write('Written side by side match image to {}\n'.format('{}/{}_{}_{}.jpg'.format(out_path, pair[0][:-4], pair[1][:-4], name)))
    output_linked = cv2.resize(linked, None, fx=0.5, fy=0.5)
    if multi_mosaic:
        cv2.imwrite('{}/{}_{}_{}.jpg'.format(out_path, pair[0][:-4], pair[1][:-4], name), output_linked)
    else:
        cv2.imwrite('{}/{}_{}_{}.jpg'.format(out_path, pair[0][:-4], pair[1][:-4], name), output_linked)


# function to stitch matched images
def stitch(img0, img1, H, use_blend):
    h, w = img1.shape[0], img1.shape[1]
    # coordinates of four points of img1
    ul = np.asarray([0, 0, 1]).reshape((3, 1))
    ur = np.asarray([w, 0, 1]).reshape((3, 1))
    ll = np.asarray([0, h, 1]).reshape((3, 1))
    lr = np.asarray([w, h, 1]).reshape((3, 1))
    P = np.hstack((ul, ur))
    P = np.hstack((P, ll))
    P = np.hstack((P, lr))

    # calculate mapped coordinates
    P_ = np.dot(H, P)
    P_ /= P_[2]

    # check if there are out of bound coordinates
    im1_xs = P_[0]
    im1_ys = P_[1]
    im0_xs = np.asarray([0, im0.shape[1]])
    im0_ys = np.asarray([0, im0.shape[0]])
    min_x = np.min(im1_xs)
    min_y = np.min(im1_ys)
    # compose a temporary translation matrix
    T = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    offset_x = 0
    offset_y = 0

    # calculate offset
    if min_x < 0:
        offset_x = int(np.abs(min_x))
        T[0, 2] = offset_x
    if min_y < 0:
        offset_y = int(np.abs(min_y))
        T[1, 2] = offset_y

    # calculate new H
    H = np.dot(T, H)
    im0_xs = np.asarray([0, im0.shape[1]])
    im0_ys = np.asarray([0, im0.shape[0]])
    xs = np.concatenate((im0_xs, im1_xs))
    ys = np.concatenate((im0_ys, im1_ys))

    file.write('Offsets are offset_x: {}, offset_y: {}\n'.format(offset_x, offset_y))

    # compensate these out of bound coordinates with offset
    xs += offset_x
    ys += offset_y

    # warpPerspective
    file.write('Generated warp perspective\n')
    result = cv2.warpPerspective(img1, H, (int(np.max(xs)), int(np.max(ys))))
    to_be_mapped = result[offset_y:offset_y + img0.shape[0], offset_x:offset_x + img0.shape[1]]

    # create a mask of the region where im0 will be masked to
    im0_mask = np.zeros_like(to_be_mapped)
    # paste only the overlap region onto the mask
    im0_mask[to_be_mapped != [0, 0, 0]] = img0[to_be_mapped != [0, 0, 0]]
    # fill in the blank of img0 with img1
    im0_mask[im0_mask == [0, 0, 0]] = to_be_mapped[im0_mask == [0, 0, 0]]
    # mask for im1
    im1_mask = to_be_mapped.copy()

    # average the overlapped region
    avgeraged_overlap = cv2.addWeighted(im0_mask, 0.5, im1_mask, 0.5, 0)
    # combine the overlap region and non-overlap regions into a final mask
    mask = result.copy()
    mask[offset_y:offset_y + img0.shape[0], offset_x:offset_x + img0.shape[1]][to_be_mapped != [0, 0, 0]] = avgeraged_overlap[to_be_mapped != [0, 0, 0]]
    mask[offset_y:offset_y + img0.shape[0], offset_x:offset_x + img0.shape[1]][to_be_mapped == [0, 0, 0]] = img0[to_be_mapped == [0, 0, 0]]
    # draw a bounding box on the warped image
    im1_xs += offset_x
    im1_ys += offset_y
    cv2.line(mask, (int(im1_xs[1]), int(im1_ys[1])), (int(im1_xs[0]), int(im1_ys[0])), (255, 255, 255), 3)
    cv2.line(mask, (int(im1_xs[0]), int(im1_ys[0])), (int(im1_xs[2]), int(im1_ys[2])), (255, 255, 255), 3)
    cv2.line(mask, (int(im1_xs[2]), int(im1_ys[2])), (int(im1_xs[3]), int(im1_ys[3])), (255, 255, 255), 3)
    cv2.line(mask, (int(im1_xs[3]), int(im1_ys[3])), (int(im1_xs[1]), int(im1_ys[1])), (255, 255, 255), 3)

    return mask

    # if use_blend:
    #     mask = np.zeros_like(result)
    #     mask[offset_y:offset_y + img0.shape[0], offset_x:offset_x + img0.shape[1]] = img0
    #     result = cv2.addWeighted(result, 0.5, mask, 0.5, 0)
    # else:
    #     result[offset_y:offset_y + img0.shape[0], offset_x:offset_x + img0.shape[1]] = img0
    # return result


# function to get all pairs of matched images
def catch_pairs(dir, im_lst):
    # generate all possible pairs out of all images
    im_pairs = list(itertools.combinations(im_lst, 2))
    id = 0
    for pair in im_pairs:
        print(pair)
        im0 = cv2.imread(dir + '/' + pair[0])
        im1 = cv2.imread(dir + '/' + pair[1])
        # check if they match with each other
        homoM = decide_match(im0, im1, id, True, pair)
        # if there is a homography matrix
        if np.size(homoM) > 0:
            key = pair[0]
            val = pair[1]
            # store them in the decision dictionary
            if not key in decision_dict.keys():
                decision_dict[key] = []
            decision_dict[key].append(val)
            # store them in matched record set
            matched_record.add(pair[0])
            matched_record.add(pair[1])
            file.write('\n')
        id += 1


# check if two image are matched
def decide_match(im0, im1, pair_id, draw_match, pair):
    file.write('\nchecking match and calculating homography matrix for {} and {}\n'.format(pair[0], pair[1]))
    MATCHED = False
    # convert images to gray images
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im0 = im0.astype(np.uint8)
    im1 = im1.astype(np.uint8)
    # detect and compute SIFT features
    sift = cv2.xfeatures2d.SIFT_create()
    sift_lst0, des0 = sift.detectAndCompute(im0, None)
    sift_lst1, des1 = sift.detectAndCompute(im1, None)
    # match the two images' SIFT features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)
    file.write('Before ratio test, there are {} match points\n'.format(len(matches)))
    # apply ratio test
    good = []
    ratio_test_mask = []
    raw_matches = []
    for m, n in matches:
        raw_matches.append(m)
        if m.distance < 0.75 * n.distance:
            good.append(m)
            ratio_test_mask.append(1)
        else:
            ratio_test_mask.append(0)
    file.write('After ratio test, there are {} match points\n'.format(len(good)))
    # store all the good points
    im0_pts = []
    im1_pts = []
    for m in good:
        im0_pts.append(sift_lst0[m.queryIdx].pt)
        im1_pts.append(sift_lst1[m.trainIdx].pt)
    if draw_match:
        # draw matches after ratio test
        draw_side_by_side(im0, im1, sift_lst0, sift_lst1, pair, np.asarray(ratio_test_mask), raw_matches, 'aft_ratio')
    im0_pts = np.float32(im0_pts)
    im1_pts = np.float32(im1_pts)
    # calculate fundamental matrix
    fundaM, fundaMask = cv2.findFundamentalMat(im0_pts, im1_pts, cv2.FM_RANSAC)
    if draw_match:
        # draw matches after fundamental matrix calculation
        draw_side_by_side(im0, im1, sift_lst0, sift_lst1, pair, fundaMask, good, 'aft_funda')
    mask_1d = fundaMask.flatten()
    # filter out inliers after fundamental matrix calculation
    im0_inlier = im0_pts[mask_1d == 1]
    im1_inlier = im1_pts[mask_1d == 1]
    good = np.asarray(good)
    good = good[mask_1d == 1]
    file.write('After calculate fundamental matrix test, there are {} match points\n'.format(len(good)))
    # apply another filter to check if there are at least 75% points left after fundamental matrix calculation
    if len(im0_inlier) / len(im0_pts) > 0.75:
        file.write('\tAfter calculate fundamental matrix, the amount of match points is bigger than 75% of that at previous step\n')
        MATCHED = True
    else:
        file.write('\tAfter calculate fundamental matrix, the amount of match points is smaller than 75% of that at previous step\n')
        MATCHED = False
    # calculate homography matrix
    homoM, mask = cv2.findHomography(im1_inlier, im0_inlier, cv2.RANSAC, 4.0)
    if draw_match:
        # draw matches after homography matrix calculation
        draw_side_by_side(im0, im1, sift_lst0, sift_lst1, pair, mask, good, 'aft_homo')
    file.write('After calculate homography matrix test, there are {} match points\n'.format(len(np.where(mask == 1)[0])))
    # apply another filter to check if there are at least 75% points left after homography matrix calculation
    if len(np.where(mask == 1)[0]) / len(good) > 0.75:
        file.write('\tAfter calculate homography matrix, the amount of match points is bigger than 75% of that at previous step\n')
        MATCHED = True
    else:
        file.write('\tAfter calculate homography matrix, the amount of match points is smaller than 75% of that at previous step\n')
        MATCHED = False
    # if they are matched print to show they are matched
    if MATCHED:
        file.write('{} and {} are matched\n'.format(pair[0], pair[1]))
        print('Yea-Matched\n')
        return homoM
    else:
        file.write('{} and {} are not matched\n'.format(pair[0], pair[1]))
        print('Nah-Matched\n')
        return np.asarray([])


# helper function to find all path between to nodes of a graph
# the graph is represented by dictionary, an image is a node
# if two images are matched they are connected by an edge, graph is undirected
def find_all_paths(dict_, start, end, path=[]):
        path = path + [start]
        # stop condition
        if start == end:
            return [path]
        if not start in dict_.keys():
            return []
        paths = []
        for node in dict_[start]:
            if node not in path:
                # recurse
                newpaths = find_all_paths(dict_, node, end, path)
                # record path
                for newpath in newpaths:
                    paths.append(newpath)
        return paths


# helper function to complete the dictionary
# it is to store all possible matches into the path dictionary, make originally directed path undirected
# e.g. original: {a: [b, c]} -> complete: {a: [b, c], b: [a, c], c: [a, b]}
def complete_dict():
    # loop through decision dictionary
    for key in decision_dict.keys():
        row = [key] + decision_dict[key]
        # complete all connections (like example above)
        for item in row:
            other = [x for x in row if x != item]
            if item not in path_dict.keys():
                path_dict[item] = set(other)
            else:
                path_dict[item] = path_dict[item].union(set(other))


if __name__ == '__main__':
    file.write('reading directory: {}\n'.format(dir_))
    # load images
    im_lst_ = load_helper(dir_)
    file.write('Loaded images: {}\n'.format(im_lst_))
    # check matches
    catch_pairs(dir_, im_lst_)
    im_pairs = list(itertools.combinations(im_lst_, 2))
    # do the multi image mosaic
    if multi_mosaic:
        print('Generating multi image mosaic')
        file.write('Generating multi image mosaic\n')
        all_path = []
        complete_dict()
        # find all possible path
        for pair in im_pairs:
            one_path = find_all_paths(path_dict,  pair[0], pair[1])
            all_path += one_path
        # filter path that passes through all nodes (a path connect all images)
        filtered_path = [path for path in all_path if len(path) == len(matched_record)]
        if len(all_path) != 0:
            if len(filtered_path) == 0:
                # if there is not such path passes through all nodes, print no multi-image mosaic
                print('We do not have multi-image mosaic')
                file.write('We do not have multi-image mosaic\n')
            else:
                optimal_path = filtered_path[0]
                print('We have multi-image matches, follows PATH: {}'.format(optimal_path))
                file.write('We have multi-image matches, follows PATH: {}\n'.format(optimal_path))
                # load the initial image
                name0 = dir_ + '/' + optimal_path[0]
                im0 = cv2.imread(name0)
                result = None
                for i in range(1, len(optimal_path)):
                    # load second image
                    name1 = dir_ + '/' + optimal_path[i]
                    im1 = cv2.imread(name1)
                    # calculate homography matrix for these to images
                    H = decide_match(im0, im1, 'mosaic{}'.format(i), True, [optimal_path[0], optimal_path[1]])
                    # stitch these two images
                    m = stitch(im0, im1, H, True)
                    # write to out put
                    im0 = m
                    result = m
                    file.write('\n')
                file.write('multi-image mosaic is written to {}\n'.format('{}/multi-mosiac.jpg'.format(out_path)))
                output_res =  cv2.resize(result, None, fx=0.5, fy=0.5)
                cv2.imwrite('{}/multi-mosiac.jpg'.format(out_path), output_res)
        else:
            # if no match print no match to output
            file.write('We do not any matches\n')
            print('We do not any matches')
    # do dual-image mosaic
    else:
        file.write('Generating dual image mosaic\n')
        print('Generating dual image mosaic')
        # generate all pairs
        all_path = []
        for pair in im_pairs:
            one_path = find_all_paths(decision_dict,  pair[0], pair[1])
            all_path += one_path
        # check if there are matches
        if len(all_path) != 0:
            # filter to get path that only contains a pair of images
            filtered_path = [path for path in all_path if len(path) == 2]
            for i in range(len(filtered_path)):
                path = filtered_path[i]
                # load two images
                name0 = dir_ + '/' + path[0]
                im0 = cv2.imread(name0)
                name1 = dir_ + '/' + path[1]
                im1 = cv2.imread(name1)
                # calculate homography matrix
                H = decide_match(im0, im1, 'mosaic{}'.format(i), True, [path[0], path[1]])
                # stitch the pair of images
                m = stitch(im0, im1, H, True)
                # save the mosaic
                output_m = cv2.resize(m, None, fx=0.5, fy=0.5)
                file.write('dual-image mosaic of {} and {} is written to {}\n'.format(path[0][:-4], path[1][:-4], '{}/{}_{}.jpg'.format(out_path, path[0][:-4], path[1][:-4])))
                cv2.imwrite('{}/{}_{}.jpg'.format(out_path, path[0][:-4], path[1][:-4]), output_m)
                file.write('\n')
        else:
            # if no match print no match to output
            file.write('We do not any matches\n')
            print('We do not any matches')

    file.close()








