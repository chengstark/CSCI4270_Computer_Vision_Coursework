import numpy as np
import sys
import matplotlib.pyplot as plt

# parsing input arguments
file_name = sys.argv[1]
tau = sys.argv[2]
out_name = sys.argv[3]


# avoid ambiguity of direction according to the instruction
def ensure_direction(x_, y_):
    if x_ > 0:
        return np.asarray([x_, y_])
    elif x_ == 0:
        return np.asarray([x_, abs(y_)])
    elif x_ < 0 and y_ > 0:
        return np.asarray([y_, x_])


# function to calculate PCA
def pca(x_):
    m = np.mean(x_, axis=0)
    centered = x_ - m
    cov_matrix = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    eig_vecs = eig_vecs.T
    # sort eigenvectors by eigenvalues
    eig_vecs = eig_vecs[np.argsort(eig_vals)]
    eigvec_0 = eig_vecs[0].T
    eigvec_1 = eig_vecs[1].T
    return eigvec_0, eigvec_1


# read input text
f = np.loadtxt(file_name, dtype=np.float64)

# calculate com, min and max values and write to console
com = (np.mean(f, axis=0)[0], np.mean(f, axis=0)[1])
min_ = (np.min(f, axis=0)[0], np.min(f, axis=0)[1])
max_ = (np.max(f, axis=0)[0], np.max(f, axis=0)[1])
print('min: ({:.3f},{:.3f})'.format(min_[0], min_[1]))
print('max: ({:.3f},{:.3f})'.format(max_[0], max_[1]))
print('com: ({:.3f},{:.3f})'.format(com[0], com[1]))


# calculate PCA
eig_vec_0, eig_vec_1 = pca(f)
# calculate standard deviations
x_pca_0 = np.dot(f, eig_vec_0)
x_pca_1 = np.dot(f, eig_vec_1)
Smin = np.std(x_pca_0)
Smax = np.std(x_pca_1)
# make sure direction is write
directed_0 = ensure_direction(eig_vec_0[0], eig_vec_0[1])
directed_1 = ensure_direction(eig_vec_1[0], eig_vec_1[1])
# write to console# 
print('min axis: ({:.3f},{:.3f}), sd {:.3f}'.format(directed_0[0], directed_0[1], Smin))
print('max axis: ({:.3f},{:.3f}), sd {:.3f}'.format(directed_1[0], directed_1[1], Smax))

# calculate closest form's parameters and write to console
theta = np.arcsin(eig_vec_1[0])
com = np.mean(f, axis=0)
x, y = com
rho = x*np.cos(theta) + y*np.sin(theta)
print('closest point: rho {:.3f}, theta {:.3f}'.format(rho, theta))

# calculate implicit form parameters from the closest form's parameters
a = np.cos(theta)
b = np.sin(theta)
c = 0-(np.cos(theta)*x + np.sin(theta)*y)
print('implicit: a {:.3f}, b {:.3f}, c {:.3f}'
      .format(a, b, c))

# decide whether it is a line or ellipse
if Smin < float(tau) * Smax:
    print('best as line')
else:
    print('best as ellipse')


# ax + by + c = 0
# by = -ax-c
# y = -a/b -c/b


# plot the graph with previously calculated parameters
plt.scatter(f[:, 0], f[:, 1])
plt.plot(com[0], com[1], 'ro')
x = np.linspace(min_[0]-5, max_[0]+5, 100)
y = (-a/b)*x + (-c/b)
plt.plot(x, y, '-g', label='BEST FIT')
plt.show()
plt.savefig(out_name)

