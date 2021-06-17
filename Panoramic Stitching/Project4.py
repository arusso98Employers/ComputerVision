import cv2
import os
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import math

path = os.getcwd() + os.sep + "set1"
image_1 = cv2.imread(path + os.sep + "1.jpg")
image_2 = cv2.imread(path + os.sep + "2.jpg")
image_3 = cv2.imread(path + os.sep + "3.jpg")

# print(image_1.shape)
# plt.imshow(image_1),plt.title('Image1')

plt.imshow(image_1),plt.title('Image1')
plt.show()

rows,cols,_ = image_1.shape
theta = math.radians(10)
rotation_matrix = np.float32([[math.cos(theta), -(math.sin(theta)), 0],
                  [math.sin(theta), math.cos(theta), 0],
                  [0, 0, 1]])
dst = cv2.warpPerspective(image_1, rotation_matrix, (1000, 800))
plt.imshow(dst),plt.title('Output Image 1 rotated')
plt.show()

#translate image 2, 100 pixels to the right
plt.imshow(image_2),plt.title('Image2')
plt.show()


rows,cols,_ = image_2.shape
translation_matrix = np.float32([[1, 0, 100],
                  [0, 1, 0],
                  [0, 0, 1]])
dst = cv2.warpPerspective(image_2, translation_matrix, (1000, 800))
plt.imshow(dst),plt.title('Output Image 2 translated')
plt.show()

#Shrink image 3 by a factor of 1/2
plt.imshow(image_3),plt.title('Image3')
plt.show()


rows,cols,_ = image_2.shape
shrink_matrix = np.float32([[0.5, 0, 0],
                  [0, 0.5, 0],
                  [0, 0, 1]])
dst = cv2.warpPerspective(image_3, shrink_matrix, (1000, 800))
plt.imshow(dst),plt.title('Output Image 3 shrank')
plt.show()

sift = cv2.SIFT.create()
keys_1, features_1 = sift.detectAndCompute(image_1, None)
keys_2, features_2 = sift.detectAndCompute(image_2, None)
keys_3, features_3 = sift.detectAndCompute(image_3, None)

# print(descriptors_1.shape)

plt.imshow(cv2.drawKeypoints(image_1, keys_1, None))
plt.suptitle("Image 1 features")
plt.show()

plt.imshow(cv2.drawKeypoints(image_2, keys_2, None))
plt.suptitle("Image 2 features")
plt.show()

plt.imshow(cv2.drawKeypoints(image_3, keys_3, None))
plt.suptitle("Image 3 features")
plt.show()

distance_img2_img1 = distance_matrix(features_2, features_1)

#print(type(distance_img2_img1))

distance_img2_img3 = distance_matrix(features_2, features_3)

#print(distance_img2_img1)
#print(features_1)

distance_img2_img1_1d = distance_img2_img1.copy().flatten()
distance_img2_img3_1d = distance_img2_img3.copy().flatten()

sorted_distance_img2_img1 = np.sort(distance_img2_img1_1d)[0:100]
sorted_distance_img2_img3 = np.sort(distance_img2_img3_1d)[0:100]
#sorted_distance_img2_img3 = np.sort(distance_img2_img3)

indices_img2_img1 = []
indices_img2_img3 = []

for i in range(100):
    indices_img2_img1.append(np.argwhere(distance_img2_img1 == sorted_distance_img2_img1[i]))
    indices_img2_img3.append(np.argwhere(distance_img2_img3 == sorted_distance_img2_img3[i]))

indices_image_2_21 = []
indices_image_1_21 = []

for idx in indices_img2_img1:
    # print(idx)
    # print(type(idx))

    indices_image_2_21.append(idx[0, 0])

    indices_image_1_21.append(idx[0, 1])

indices_image_2 = indices_image_2_21[0:100]
indices_image_1 = indices_image_1_21[0:100]

# print(indices_image_2)
# print(indices_image_1)

best_features_img2_21 = [keys_2[i] for i in indices_image_2]

# [indices_image_2]
best_features_img1_21 = [keys_1[i] for i in indices_image_1]

indices_image_2_23 = []
indices_image_3_23 = []

for idx in indices_img2_img3:
    # print(idx)
    # print(type(idx))

    indices_image_2_23.append(idx[0, 0])

    indices_image_3_23.append(idx[0, 1])

indices_image_2 = indices_image_2_23[0:100]
indices_image_3 = indices_image_3_23[0:100]

# print(indices_image_2)
# print(indices_image_1)

best_features_img2_23 = [keys_2[i] for i in indices_image_2]

# [indices_image_2]
best_features_img3_23 = [keys_3[i] for i in indices_image_3]


src_pts = np.float32([ m.pt for m in best_features_img1_21 ]).reshape(-1,1,2)
dst_pts = np.float32([ m.pt for m in best_features_img2_21 ]).reshape(-1,1,2)

M_21, mask_21 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold = 2)

src_pts = np.float32([ m.pt for m in best_features_img3_23  ]).reshape(-1,1,2)
dst_pts = np.float32([ m.pt for m in best_features_img2_23 ]).reshape(-1,1,2)

M_23, mask_23 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC , ransacReprojThreshold = 2)
#print(M_23)
#print(mask_23)

translation_matrix = np.float32([[1, 0, 300],
                  [0, 1, 350],
                  [0, 0, 1]])

homography_translation_21 = np.dot(M_21, translation_matrix)

dst1 = cv2.warpPerspective(image_1, homography_translation_21, (1000, 800))
plt.imshow(dst1),plt.title('Output')
plt.show()

homography_translation_23 = np.dot( M_23, translation_matrix)

dst2 = cv2.warpPerspective(image_3, homography_translation_23, (1000, 800))
plt.imshow(dst2),plt.title('Output 2')
plt.show()

dst3 = cv2.warpPerspective(image_2, translation_matrix, (1000, 800))
plt.imshow(dst3),plt.title('Output 3')
plt.show()

arr1 = np.maximum(dst1, dst2)
stitched_image = np.maximum(arr1, dst3)
plt.imshow(stitched_image),plt.title('Final Output')
plt.show()