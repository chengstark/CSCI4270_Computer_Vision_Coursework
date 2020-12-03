import numpy as np

# print((np.floor((-0.02082866 / (np.pi / 18))) + 18).astype(np.int8))
# print(np.int(-0.02082866 / (np.pi / 18)))
# print(np.int(-0.02082866 / (np.pi / 18)) + 18)
#
# for i in range(0, 36):
#     lower_bound = (i - 18) * 10
#     upper_bound = lower_bound + 10
#     print('[{}, {}]'.format(lower_bound, upper_bound), i, i-18)
#
# print(((0.93399734 + 1.14741214)/2 + 1.50729924)/2)

x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
x = x.flatten()
y = np.zeros((10,))
idx = np.asarray([0, 1, 2, 3])
y[x[idx]] += 1
print(x[idx])
print(y)

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
res = np.multiply(x1, x2)
print(x1)
print(x2)
print(res)
idx = np.where(x1 == 1)
print(idx)

