import numpy as np
import sys

param_file = sys.argv[1]

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
rx1, ry1, rz1, s1, ic1, jc1, rx2, ry2, rz2, s2, ic2, jc2, sample = ipts


'''function to calculate K and R matrix'''
def calc_K_R(rx, ry, rz, s, ic, jc):
    rx = rx*np.pi/180
    ry = ry*np.pi/180
    rz = rz*np.pi/180
    '''compose the K matrix'''
    K = [[s, 0, jc], [0, s, ic], [0, 0, 1]]
    K = np.asarray(K)
    '''compose the R matrix from R = RxRyRz'''
    R_x = [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    R_y = [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0 , np.cos(ry)]]
    R_z = [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    '''combine them all together'''
    R_x = np.asarray(R_x)
    R_y = np.asarray(R_y)
    R_z = np.asarray(R_z)
    R = np.dot(R_x, R_y)
    R = np.dot(R, R_z)
    return R, K


def create_point(p):
    '''create a u point'''
    return np.asarray([p[1], p[0], 1])


def calc_overlap(H, points):
    inside = 0
    '''loop through points and generate the overlap results'''
    for point in points:
        res = np.matmul(H, point)
        res /= res[-1]
        if check_inside(res):
            inside += 1
    return inside


if __name__ == '__main__':
    R1, K1 = calc_K_R(rx1, ry1, rz1, s1, ic1, jc1)
    R2, K2 = calc_K_R(rx2, ry2, rz2, s2, ic2, jc2)

    '''K2R2R⊤1 K−1.'''
    H = np.matmul(K2, R2)
    H = np.matmul(H, R1.T)
    H = np.matmul(H, np.linalg.inv(K1))
    H_21 = H.copy()
    H_21 = H_21 / np.linalg.norm(H_21) * 1000
    print('Matrix: H_21')
    output = ''
    for row in range(H_21.shape[0]):
        for col in range(H_21.shape[1]):
            if col != H_21.shape[1] - 1:
                output += '{:.3f}, '.format(H_21[row, col])
            else:
                output += '{:.3f}'.format(H_21[row, col])
        if row != H_21.shape[0] - 1:
            output += '\n'
    print(output)

    '''create 4 points (0,0), (0,6000), (4000,0) and (4000,6000)'''
    u00 = create_point([0, 0])
    u01 = create_point([0, 6000])
    u02 = create_point([4000, 0])
    u03 = create_point([4000, 6000])

    u10 = np.matmul(H_21, u00)
    u11 = np.matmul(H_21, u01)
    u12 = np.matmul(H_21, u02)
    u13 = np.matmul(H_21, u03)

    u = np.vstack((u10, u11))
    u = np.vstack((u, u12))
    u = np.vstack((u, u13))

    u = u / u[:, -1].reshape(u.shape[0], 1)

    xs = u[:, 0]
    ys = u[:, 1]

    '''print upper left and lower right points'''
    print('Upper left: {:.1f} {:.1f}'.format(np.min(ys), np.min(xs)))
    print('Lower right: {:.1f} {:.1f}'.format(np.max(ys), np.max(xs)))

    ''''generate grid'''
    y_step = 4000 / sample
    x_step = 6000 / sample
    xs = np.arange(x_step/2, 6000+x_step/2, x_step)
    ys = np.arange(y_step/2, 4000+y_step/2, y_step)
    y_grid, x_grid = np.meshgrid(xs, ys)
    x_grid = x_grid.flatten().reshape(x_grid.shape[0]*x_grid.shape[1], 1).astype(np.int64)
    y_grid = y_grid.flatten().reshape(y_grid.shape[0]*y_grid.shape[1], 1).astype(np.int64)
    grid = np.hstack((x_grid, y_grid))
    '''convert grid to u points'''
    points = np.apply_along_axis(create_point, 1, grid)
    '''calculate overlap for H21'''
    H_21_inside = calc_overlap(H_21, points)
    '''calculate H12'''
    H_12 = np.matmul(K1, R1)
    H_12 = np.matmul(H_12, R2.T)
    H_12 = np.matmul(H_12, np.linalg.inv(K2))
    '''calculate overlap for H12'''
    H_12_inside = calc_overlap(H_12, points)
    '''print to output'''
    print('H_21 overlap count {}'.format(H_21_inside))
    print('H_21 overlap fraction {:.3f}'.format(H_21_inside / grid.shape[0]))
    print('H_12 overlap count {}'.format(H_12_inside))
    print('H_12 overlap fraction {:.3f}'.format(H_12_inside / grid.shape[0]))

    '''direction = R.T K^-1 v'''
    v = np.asarray([[3000], [2000], [1]])
    res = np.matmul(R2.T, np.matmul(np.linalg.pinv(K2), v))
    print('Image 2 center direction: ({:.3f}, {:.3f}, {:.3f})'.format(res[0, 0], res[1, 0], res[2, 0]))


