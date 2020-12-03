import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sigma = float(sys.argv[1])
file = sys.argv[2]
pts = sys.argv[3]

'''function to calculate histogram'''
def hist(x, y):
    '''histogram recorder'''
    histogram_ = np.zeros((36,))
    '''get gaussian kernel'''
    w = round(2 * 2 * sigma)
    ksize_w = 2 * w + 1
    Gaus_w = cv2.getGaussianKernel(ksize_w, 2 * sigma)
    Gaus_w = np.dot(Gaus_w, Gaus_w.T)
    '''slice neighbouring area'''
    local_mag = im_gm[int(x - w):int(x + w + 1), int(y - w):int(y + w + 1)]
    local_dir_ = im_gdir[int(x - w):int(x + w + 1), int(y - w):int(y + w + 1)]
    '''apply weight to the local magnitude'''
    local_mag_w = Gaus_w * local_mag
    '''convert degree to index'''
    bin_index_raw = (local_dir_ / (np.pi / 18)) + 18
    bin_index_prev = np.floor(bin_index_raw).astype(np.float32)
    '''calculate distance to the center of the bin'''
    dist = bin_index_raw - bin_index_prev - 0.5
    '''calculate bin index'''
    bin_index = (np.floor((local_dir_ / (np.pi / 18))) + 18).astype(np.int16)
    '''calulate bin index + 1 (next bin index)'''
    bin_next = (np.floor((local_dir_ / (np.pi / 18))) + 18).astype(np.int16)
    bin_next += 1
    bin_next[bin_next == 36] = 0
    '''organize to get each coordinates of the element in the bin index matrix'''
    indicies = np.indices(bin_index.shape, np.int16)
    xs = indicies[0].flatten().reshape(indicies[0].flatten().shape[0], 1)
    ys = indicies[1].flatten().reshape(indicies[1].flatten().shape[0], 1)
    grid = np.hstack((xs, ys))

    for row in grid:
        i, j = row
        '''split the weight'''
        if dist[i, j] < 0:
            histogram_[bin_index[i, j] - 1] += np.abs(dist[i, j]) * local_mag_w[i, j]
            histogram_[bin_index[i, j]] += (1 - np.abs(dist[i, j])) * local_mag_w[i, j]
        else:
            histogram_[bin_next[i, j]] += np.abs(dist[i, j]) * local_mag_w[i, j]
            histogram_[bin_index[i, j]] += (1 - np.abs(dist[i, j])) * local_mag_w[i, j]

    return histogram_, local_dir_


'''function to smooth the histogram'''
def smooth(hist_):
    smoothed_hist_ = hist_.copy()
    '''slice the histogram to left/right/center'''
    left = hist_[0: -2]
    right = hist_[2: hist_.shape[0]]
    center = hist_[1:-1]
    '''perform smoothing'''
    smoothed_hist_[1:-1] = (((left + right) / 2) + center)/2
    smoothed_hist_[0] = (hist_[0] + (hist_[1] + hist_[-1])/2)/2
    smoothed_hist_[-1] = (hist_[-1] + (hist_[-2] + hist_[0])/2)/2
    return smoothed_hist_


'''function to find peak'''
def find_peak(smoothed_hist_):
    is_max = np.zeros_like(smoothed_hist_, dtype=np.bool)
    '''slice the histogram to left/right/center'''
    left = smoothed_hist_[:-2]
    right = smoothed_hist_[2:]
    center = smoothed_hist_[1:-1]
    '''find local maxima'''
    is_max[1:-1] = (center > right) * (center > left)
    is_max[0] = smoothed_hist_[0] > smoothed_hist_[1] and smoothed_hits[0] > smoothed_hits[-1]
    is_max[-1] = smoothed_hist_[-1] > smoothed_hist_[-2] and smoothed_hits[-1] > smoothed_hits[0]
    peak_idx_ = np.where(is_max)[0]
    return peak_idx_


'''apply parabolic fit and find peak'''
def parabolic_fit(row):
    '''perform polynomial fitting'''
    xs = row[0:3]
    ys = row[3:6]
    p = np.polyfit(xs, ys, 2)
    '''calculate derivative for the fitting'''
    deriv = np.polyder(p)
    '''calculate the maxima'''
    root_x = np.roots(deriv)[0]
    root_y = np.polyval(p, root_x)
    return np.asarray([root_x, root_y])


'''function to do the interpolation'''
def interpolate(smoothed_hist_):
    '''find peaks'''
    peak_idx_ = find_peak(smoothed_hist_)
    '''find neighbour index of the peak'''
    left_idx = peak_idx_ - 1
    center_idx = peak_idx_
    right_idx = peak_idx_ + 1
    right_idx[right_idx == 36] = 0
    '''prepare histogram for interpolation'''
    left_hist = smoothed_hist_[left_idx].reshape(peak_idx_.shape[0], 1)
    center_hist = smoothed_hist_[center_idx].reshape(peak_idx_.shape[0], 1)
    right_hist = smoothed_hist_[right_idx].reshape(peak_idx_.shape[0], 1)
    to_be_itp_hist = np.hstack((left_hist, center_hist))
    to_be_itp_hist = np.hstack((to_be_itp_hist, right_hist))
    '''prepare x axis values'''
    xs = np.arange(-175, 185, 10)
    center_xs = xs[center_idx].reshape(peak_idx_.shape[0], 1)
    '''calculate neighbouring x axis values'''
    left_xs = center_xs - 10
    right_xs = center_xs + 10
    to_be_itp_xs = np.hstack((left_xs, center_xs))
    to_be_itp_xs = np.hstack((to_be_itp_xs, right_xs))
    to_be_ipt = np.hstack((to_be_itp_xs, to_be_itp_hist))
    '''do the interpolation'''
    peaks = np.apply_along_axis(parabolic_fit, 1, to_be_ipt)
    filtered_peaks = peaks[peaks[:, 1] > 0]
    '''show graphs'''
    # plt.bar(xs, smoothed_hist_)
    # plt.xticks(xs, rotation=90)
    # plt.show()

    return filtered_peaks

'''print results'''
def print_lst_of_result(peaks, hist_, smoothed_hits_):
    '''find stronger peaks'''
    strong_peaks = peaks[peaks[:, 1]/np.max(peaks[:, 1]) >= 0.8]
    # peaks = peaks[peaks[:, 1]/np.max(peaks[:, 1]) >= 0.01]
    peak_info = peaks[np.argsort(-1 * peaks[:, 1])]
    '''print histograms to outputs'''
    print('Histograms:')
    for i in range(0, 36):
        lower_bound = (i - 18) * 10
        upper_bound = lower_bound + 10
        print('[{},{}]: {:.2f} {:.2f}'.format(lower_bound, upper_bound, hist_[i], smoothed_hits_[i]))
    count = 0
    '''print peaks to outputs'''
    for row in peak_info:
        print('Peak {}: theta {:.1f}, value {:.2f}'.format(count, row[0], row[1]))
        count += 1
    print('Number of strong orientation peaks: {}'.format(strong_peaks.shape[0]))


if __name__ == '__main__':
    '''read in image'''
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    '''read in points'''
    pts = np.loadtxt(pts)
    '''apply gaussian blur'''
    ksize = (int(4 * sigma + 1), int(4 * sigma + 1))
    im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)
    '''calculate derivatives'''
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)
    '''calculate gradient magnitude and degrees'''
    im_gm = np.sqrt(np.square(im_dx) + np.square(im_dy))  # gradient magnitude
    im_gdir = np.arctan2(im_dy, im_dx)  # degrees of magnitude
    '''loop through points'''
    idx = 0
    for row in pts:
        print('\n Point {}: ({},{})'.format(idx, int(row[0]), int(row[1])))
        '''generate histogram'''
        histogram, local_dir = hist(row[0], row[1])
        '''smooth histogram'''
        smoothed_hits = smooth(histogram)
        '''interpolate histogram'''
        peaks_ = interpolate(smoothed_hits)
        '''print outputs'''
        print_lst_of_result(peaks_, histogram, smoothed_hits)
        idx += 1





