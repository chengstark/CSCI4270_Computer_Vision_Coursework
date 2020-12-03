import numpy as np
import sys

param_file = sys.argv[1]
points_file = sys.argv[2]

'''function to check if a point is within the image dimension'''
def check_inside(a):
    # print(a)
    y, x, _ = a
    if 6000 >= y >= 0 and 4000 >= x >= 0:
        return True
    else:
        return False


'''helper function to pretty print an array'''
def pretty_print(lst):
    s = ''
    for i in range(len(lst) - 1):
        s += str(lst[i])
        s += ' '
    s += str(lst[len(lst) - 1])
    return s


'''read in the parameter file'''
with open(param_file, 'r') as f:
    lines = f.read()
    lines = lines.replace('\n', ' ')
'''split the file contents to each tokens'''
ipts_ = lines.split()
ipts = [float(x) for x in ipts_]
'''assign file content to each variables and change degree to radians'''
rx, ry, rz, tx, ty, tz, f, d, ic, jc = ipts
rx = rx*np.pi/180
ry = ry*np.pi/180
rz = rz*np.pi/180
'''read in the point file'''
points_ = np.loadtxt(points_file)
points = points_.copy()
ones = np.ones((points.shape[0], 1))
points = np.concatenate((points, ones), axis=1)
'''convert to mm unit'''
d /= 1000
sx = f / d
sy = f / d
'''compose the K matrix'''
K = [[sx, 0, jc], [0, sy, ic], [0, 0, 1]]
K = np.asarray(K)
'''compose the R matrix from R = RxRyRz'''
R_x = [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
R_y = [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0 , np.cos(ry)]]
R_z = [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
R_x = np.asarray(R_x)
R_y = np.asarray(R_y)
R_z = np.asarray(R_z)
R = np.dot(R_x, R_y)
R = np.dot(R, R_z)
print(R)
'''compose the t'''
t = np.asarray([[tx], [ty], [tz]])
'''prepare points for future calculation'''
points = points.reshape((points.shape[0], points.shape[1], 1))
'''calculate final R matrix'''
tmp = -np.dot(R.T, t)
processed_R = np.concatenate((R.T, tmp), axis=1)
M = np.dot(K, processed_R)
'''print the M matrix'''
print('Matrix M:')
output = ''
for row in range(M.shape[0]):
    for col in range(M.shape[1]):
        if col != M.shape[1] - 1:
            output += '{:.2f}, '.format(M[row, col])
        else:
            output += '{:.2f}'.format(M[row, col])
    if row != M.shape[0] - 1:
        output += '\n'
print(output)
'''multiply M with points to get the translated coordinates'''
res = np.matmul(M, points)
res = res.reshape((res.shape[0], res.shape[1]))
'''normalize the coordinates to get the actual image coordinate'''
norm_fac = res[:, 2]
norm_fac = norm_fac.reshape((norm_fac.shape[0], 1))
norm_fac = np.tile(norm_fac, (1, 3))
out = np.divide(res, norm_fac)
'''prepare for another output section'''
idx = 0
vis = []
hid = []
print('Projections:')
for point in points_:
    x, y, z = point
    result = out[idx]
    '''check if points are inside'''
    if check_inside(result):
        label = 'inside'
    else:
        label = 'outside'
    print('{}: {:.1f} {:.1f} {:.1f} => {:.1f} {:.1f} {}'.format(idx, x, y, z, result[1], result[0], label))
    idx += 1
'''print visible and hidden indices by checking z axis of each point'''
R_points = np.dot(M, points.T)
vis = np.where(R_points[-1][0] >= 0)[0]
hid = np.where(R_points[-1][0] < 0)[0]
print('visible: {}'.format(pretty_print(vis)))
print('hidden: {}'.format(pretty_print(hid)))




