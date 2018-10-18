import numpy as np

grid_w = 13
grid_h = 13
grid_wh = np.reshape([13, 13], [1, 1, 1, 1, 2]).astype(np.float32)
# print(grid_wh)
# transpose 가 axis 를 변경하는 것인데,
cxcy = np.transpose([np.tile(np.arange(13), 13), np.repeat(np.arange(grid_h), grid_w)])
#
arr = np.array([1, 2, 3, 4])
rep = np.repeat(arr, 3)
# print(rep)
tras = np.transpose([(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3),
                     (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)])
# print(tras)
cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))
# print(cxcy)

# print(cxcy[..., 0:2])
n = np.array(np.arange(12).reshape(-1, 3, 2))
print(n)
print(n[..., 1])

