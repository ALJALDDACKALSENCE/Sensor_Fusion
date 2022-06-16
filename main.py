import cv2, numpy as np, matplotlib.pyplot as plt
from module import *

img = cv2.imread("data_odometry_velodyne/00/image_0/000000.png", cv2.IMREAD_COLOR)
velo_points = load_from_bin("data_odometry_velodyne/00/velodyne/000000.bin")

projection_mtx = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
                           [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
                           [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
              [1.480249e-02, 7.280733e-04, -9.998902e-01],
              [9.998621e-01, 7.523790e-03, 1.480755e-02]])   
T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

ans, c_ =velo3d_2_camera2d_points(velo_points, (-24.9, 2.0), (-45,45), projection_mtx, R, T)
image = print_projection_cv2(points=ans, color=c_, image=img)

# cv2.imshow("SF", image[156:,:]); cv2.waitKey(0)
# plt.imshow(image); plt.show()
print(image.shape)
# 156~