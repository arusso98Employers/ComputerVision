from scipy.ndimage import map_coordinates as interp2
import numpy as np
import cv2


def UndistortImage(image,LUT):
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

    
    return undistorted.astype(image.dtype)

def ReadCameraModel(models_dir):

    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT

# FIND INTRINSIC MATRIX #########################################################################
fx,fy,cx,cy,_,LUT = ReadCameraModel("./Oxford_dataset_reduced/model")

intrinsic_matrix = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]])

# LOAD AND DEMOSAIC IMAGES #########################################################################
import os
from matplotlib import pyplot as plt
import glob
import matplotlib.cm as cm
import cv2

path = "./Oxford_dataset_reduced/images"
list_of_images = []
counter = 0

list_of_names = []
prev_num = 0

for filename in glob.glob(path + '/*.png'):

    img = cv2.imread(filename,flags = -1)
    color_image = cv2.cvtColor(img,cv2.COLOR_BayerGR2BGR)
    undistorted_image = UndistortImage(color_image,LUT)
    list_of_images.append(undistorted_image)

#plt.imshow(list_of_images[0])#, cmap = "grey")

# EXTRACT KEYS AND FEATURES USING SIFT ##################################################################

list_of_keys = []
list_of_features = []
sift = cv2.SIFT.create()

for i in list_of_images:
    keys, features = sift.detectAndCompute(i, None)
    list_of_keys.append(keys)
    list_of_features.append(features)


# BRUTE FORCE MATCHER TO FIND MATCHES #########################################################################


# create BFMatcher object
bf = cv2.BFMatcher()

frame_first_second = []
# list_frame2_idx = []

draw_features_img_list = []

for i in range(len(list_of_features) - 1):

    # Match descriptors.
    matches = bf.knnMatch(list_of_features[i], list_of_features[i + 1], k=2)

    good = []
    good_non_list = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            good_non_list.append(m)

    img3 = cv2.drawMatchesKnn(list_of_images[i], list_of_keys[i], list_of_images[i + 1], list_of_keys[i + 1], good,
                              None, flags=2)
    draw_features_img_list.append(img3)
    # plt.imshow(img3),plt.show()

    frame_first_second.append(good_non_list)




# GET THE FRAME POINTS #########################################################################

point_frame_first_second = []

for matches, idx in zip(frame_first_second, range(len(frame_first_second))):
    # matches = matches[0:100]
    frame1_points = []
    frame2_points = []
    for match in matches:
        # print(type(match[0]))
        frame1_points.append(list_of_keys[idx][match.queryIdx].pt)
        frame2_points.append(list_of_keys[idx + 1][match.trainIdx].pt)

    point_frame_first_second.append((frame1_points, frame2_points))

# CALCULATE FUNDAMENTAL MATRIX #########################################################################

from cv2 import FM_RANSAC

# cv2.UMat()

fundamental_matrix_list = []
# essential_matrix_list_cv2 = []
list_of_masks = []
point_frame_first_second_final = []

for i, j in point_frame_first_second:
    p1 = np.array(i)  # cv2.UMat(np.array(i, dtype=np.uint8))
    p2 = np.array(j)  # cv2.UMat(np.array(j, dtype=np.uint8))

    # essential_matrix_cv2, _ = cv2.findEssentialMat(p1, p2, intrinsic_matrix)
    # essential_matrix_list_cv2.append(essential_matrix_cv2)

    fundamental_matrix, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, ransacReprojThreshold=1, confidence=0.97)

    # We select only inlier points
    pts1 = p1[mask.ravel() == 1]
    pts2 = p2[mask.ravel() == 1]

    point_frame_first_second_final.append((pts1, pts2))
    fundamental_matrix_list.append(fundamental_matrix)


# CALCULATE ESSENTIAL MATRIX #########################################################################
essential_matrix_list = []

for i in fundamental_matrix_list:
    #print(i)
    #essential_matrix = np.dot(i, intrinsic_matrix))
    essential_matrix =np.dot( np.dot(intrinsic_matrix.T, i), intrinsic_matrix)
    essential_matrix_list.append(essential_matrix)

# FIND ROTATION AND TRANSLATION MATRICES #########################################################################

rotation_matrix_list = []
translation_matrix_list = []

for i, e_mat in enumerate(essential_matrix_list):
    p1 = np.array(point_frame_first_second_final[i][0])
    p2 = np.array(point_frame_first_second_final[i][1])

    points, R_est, t_est, mask_pose = cv2.recoverPose(e_mat, p1, p2,intrinsic_matrix)
    rotation_matrix_list.append(R_est)
    translation_matrix_list.append(t_est)

# PLOT TRAJECTORY #########################################################################

from numpy.linalg import inv
from scipy.linalg import solve

camera_centers = [np.array([[0], [0], [0], [1]])]
list_of_x = [0]
list_of_y = [0]
list_of_z = [0]

# print(len(rotation_matrix_list))
# print(len(translation_matrix_list))
RT_Prev = np.identity(4)

for r, t, idx in zip(rotation_matrix_list, translation_matrix_list, range(len(rotation_matrix_list))):
    # print(r.shape)
    # print(t.shape)

    R_4d = np.vstack((r, np.zeros((1, 3))))
    t_4d = np.array([t[0], t[1], t[2], [1]])
    # t_4d = np.hstack((t, [[1]])
    # print(t_4d.shape)
    # print(R_4d.shape)
    RT_matrix = np.hstack((R_4d, t_4d))
    RT_matrix = np.matmul(RT_matrix, RT_Prev)
    RT_Prev = RT_matrix
    # x_0 = np.identity(3)
    # print(camera_centers.shape)

    x_n = solve(RT_matrix, camera_centers[0])
    # x_k = RT_matrix *

    # x_n = np.dot(inv(RT_matrix), camera_centers[0])
    # print(RT_matrix)

    camera_centers.append(x_n)
    list_of_x.append(x_n[0, 0])
    list_of_y.append(x_n[1, 0])
    list_of_z.append(x_n[2, 0])


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')

ax.plot3D(list_of_x, list_of_y, list_of_z)

plt.show()

plt.figure(figsize=(9, 6))

plt.plot(list_of_x, list_of_z)

plt.show()