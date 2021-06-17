import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def simplesubplot(index, img, title):
    plt.subplot(index), plt.imshow(img, cmap="gray")
    plt.title(title), plt.xticks([]), plt.yticks([])


def find_sift_features(path, instance):
    b_list = []
    c_list = []
    p_list = []

    b_features_count_per_image = []
    c_features_count_per_image = []
    p_features_count_per_image = []

    # path = os.getcwd() + os.sep + "Project2_data" + os.sep + "TrainingDataset"
    butterfly_files = [f for f in os.listdir(path) if f.find("024_") != -1]
    cowboy_files = [f for f in os.listdir(path) if f.find("051_") != -1]
    plane_files = [f for f in os.listdir(path) if f.find("251_") != -1]

    for i in range(len(butterfly_files)):
        b_list.append(cv2.imread(path + os.sep + butterfly_files[i],
                                 cv2.IMREAD_GRAYSCALE))

    for i in range(len(cowboy_files)):
        c_list.append(cv2.imread(path + os.sep + cowboy_files[i],
                                 cv2.IMREAD_GRAYSCALE))

    for i in range(len(plane_files)):
        p_list.append(cv2.imread(path + os.sep + plane_files[i],
                                 cv2.IMREAD_GRAYSCALE))

    # print(b_list)
    sift = cv2.SIFT.create()

    # I eventually decided to try display all images on 1 page.
    # first number is number of rows. Second is number of colunms and 3rd is the index`

    # f stores the location, size, and orientation of the keypoints
    # d stores the actual (128 × 1) feature descriptors.
    list_of_b_keys = []
    list_of_b_desc = []
    list_of_b_keys_img = []

    if instance == 1:
        figure, axis = plt.subplots(10, 5)
    else:
        figure, axis = plt.subplots(2, 5)

    for i, ax in zip(range(len(b_list)), np.ravel(axis)):
        keys, descriptors = sift.detectAndCompute(b_list[i], None)
        list_of_b_keys.append(keys)
        list_of_b_desc.append(descriptors)

        list_of_b_keys_img.append(cv2.drawKeypoints(b_list[i], keys, None))
        ax.axis('off')
        ax.imshow(list_of_b_keys_img[i], cmap="gray")
        plt.suptitle("butterfly"), plt.xticks([]), plt.yticks([])

    var = [rows.shape[0] for rows in list_of_b_desc]
    # print(sum(var))

    list_of_c_keys = []
    list_of_c_desc = []
    list_of_c_keys_img = []

    if instance == 1:
        figure, axis = plt.subplots(10, 5)
    else:
        figure, axis = plt.subplots(2, 5)

    for i, ax in zip(range(len(c_list)), np.ravel(axis)):
        keys, descriptors = sift.detectAndCompute(c_list[i], None)
        list_of_c_keys.append(keys)

        list_of_c_desc.append(descriptors)
        list_of_c_keys_img.append(cv2.drawKeypoints(c_list[i], keys, None))
        ax.axis('off')
        ax.imshow(list_of_c_keys_img[i], cmap="gray")
        plt.suptitle("cowboy hats"), plt.xticks([]), plt.yticks([])

    list_of_p_keys = []
    list_of_p_desc = []
    list_of_p_keys_img = []

    if instance == 1:
        figure, axis = plt.subplots(10, 6)
    else:
        figure, axis = plt.subplots(4, 4)

    for i, ax in zip(range(len(p_list)), np.ravel(axis)):
        keys, descriptors = sift.detectAndCompute(p_list[i], None)
        list_of_p_keys.append(keys)
        list_of_p_desc.append(descriptors)

        list_of_p_keys_img.append(cv2.drawKeypoints(p_list[i], keys, None))
        ax.axis('off')
        ax.imshow(list_of_p_keys_img[i], cmap="gray")
        plt.suptitle("planes"), plt.xticks([]), plt.yticks([])

    # for i, j, k in zip(list_of_b_keys, list_of_c_keys, list_of_p_keys):
    # print(len(i), len(j), len(k))

    # for loop to determine number of features (128)
    # for i, j , k in zip(list_of_b_desc, list_of_c_desc, list_of_p_desc):
    # print(i.shape, j.shape, k.shape)

    b_features_count_per_image = [i.shape[0] for i in list_of_b_desc]
    c_features_count_per_image = [i.shape[0] for i in list_of_c_desc]
    p_features_count_per_image = [i.shape[0] for i in list_of_p_desc]

    b_descriptors = list_of_b_desc[0]
    for descriptor in list_of_b_desc[1:]:
        b_descriptors = np.vstack((b_descriptors, descriptor))

    c_descriptors = list_of_c_desc[0]
    for descriptor in list_of_c_desc[1:]:
        c_descriptors = np.vstack((c_descriptors, descriptor))

    p_descriptors = list_of_p_desc[0]
    for descriptor in list_of_p_desc[1:]:
        p_descriptors = np.vstack((p_descriptors, descriptor))

    # print(b_descriptors.shape, c_descriptors.shape, p_descriptors.shape)

    return b_descriptors, c_descriptors, p_descriptors, b_features_count_per_image, c_features_count_per_image, p_features_count_per_image

path = os.getcwd() + os.sep + "Project2_data" + os.sep + "TrainingDataset"
b_descriptors , c_descriptors, p_descriptors, b_features_count_per_image, c_features_count_per_image, p_features_count_per_image  = find_sift_features(path, 1)

def kmeans_clustering(stacked_descriptors):

    model = KMeans(n_clusters=100, random_state=5, max_iter=300)
    model.fit_predict(stacked_descriptors)
    labels = model.labels_
    clust_center = model.cluster_centers_


    return model, labels, clust_center

stack_descriptors = np.vstack((b_descriptors, c_descriptors, p_descriptors))

model, train_labels, cluster_centers = kmeans_clustering(stack_descriptors)


def form_histograms(labels, b_features_count_per_image, c_features_count_per_image, p_features_count_per_image,
                    instance):
    # print(b_l.shape)
    start_position_in_labels = 0
    end_position_in_labels = 0
    list_of_b_labels = []
    list_of_c_labels = []
    list_of_p_labels = []

    #     #all_img_test_histograms = list_of_b_test_labels[0]

    #     for histogram in list_of_b_test_labels[1:]:
    #         all_img_test_histograms = np.vstack((all_img_test_histograms, histogram))

    #     for histogram in list_of_c_test_labels[0:]:
    #         all_img_test_histograms = np.vstack((all_img_test_histograms, histogram))

    #     for histogram in list_of_p_test_labels[0:]:
    #         all_img_test_histograms = np.vstack((all_img_test_histograms, histogram))

    if instance == 1:
        figure, axis = plt.subplots(10, 5, figsize=(16, 9))
    else:
        figure, axis = plt.subplots(2, 5, figsize=(16, 9))

    figure.tight_layout()
    plt.suptitle("Butterfly histogram", y=1)

    # print(len(b_desc))
    # print(b_desc.shape)
    # print(len(b_l))
    all_img_histogram = None

    for i, idx, ax in zip(b_features_count_per_image, range(len(b_features_count_per_image)), np.ravel(axis)):
        # i.shape[0] is the total number of sift features
        length_of_img_descriptors = i
        end_position_in_labels += length_of_img_descriptors
        # Dividing by the number of features (normalization)
        # print(i)
        list_of_b_labels.append(np.divide(labels[start_position_in_labels:end_position_in_labels], i))
        start_position_in_labels = end_position_in_labels
        img_histogram = ax.hist(list_of_b_labels[idx], bins=100)

        stack_img_histogram = img_histogram[0]
        stack_img_histogram = np.hstack((stack_img_histogram, img_histogram[1]))

        # print(img_histogram[0], img_histogram[1])

        if np.all(all_img_histogram != None):
            all_img_histogram = np.vstack((all_img_histogram, stack_img_histogram))
        else:
            all_img_histogram = stack_img_histogram

    # print(list_of_b_labels)
    plt.show()

    if instance == 1:
        figure, axis = plt.subplots(10, 5, figsize=(16, 9))
    else:
        figure, axis = plt.subplots(2, 5, figsize=(16, 9))

    figure.tight_layout()
    plt.suptitle("Hat histogram", y=1)

    # print(len(c_desc))

    for i, idx, ax in zip(c_features_count_per_image, range(len(c_features_count_per_image)), np.ravel(axis)):
        # i.shape[0] is the total number of sift features
        length_of_img_descriptors = i
        end_position_in_labels += length_of_img_descriptors
        # Dividing by the number of features (normalization)
        list_of_c_labels.append(np.divide(labels[start_position_in_labels:end_position_in_labels], i))
        start_position_in_labels = end_position_in_labels
        img_histogram = ax.hist(list_of_c_labels[idx], bins=100)

        stack_img_histogram = img_histogram[0]

        stack_img_histogram = np.hstack((stack_img_histogram, img_histogram[1]))

        # print(img_histogram[0].shape, img_histogram[1])

        if np.all(all_img_histogram != None):
            all_img_histogram = np.vstack((all_img_histogram, stack_img_histogram))
        else:
            all_img_histogram = stack_img_histogram

    # print(list_of_c_labels)
    plt.show()

    if instance == 1:
        figure, axis = plt.subplots(10, 6, figsize=(16, 9))
    else:
        figure, axis = plt.subplots(4, 4, figsize=(16, 9))

    figure.tight_layout()
    plt.suptitle("Plane histogram", y=1)

    # print(len(p_desc))
    for i, idx, ax in zip(p_features_count_per_image, range(len(p_features_count_per_image)), np.ravel(axis)):
        # i.shape[0] is the total number of sift features
        length_of_img_descriptors = i
        end_position_in_labels += length_of_img_descriptors
        # Dividing by the number of features (normalization)
        list_of_p_labels.append(np.divide(labels[start_position_in_labels:end_position_in_labels], i))
        start_position_in_labels = end_position_in_labels
        img_histogram = ax.hist(list_of_p_labels[idx], bins=100)

        stack_img_histogram = img_histogram[0]
        stack_img_histogram = np.hstack((stack_img_histogram, img_histogram[1]))

        # print(img_histogram[0].shape, img_histogram[1].shape)

        if np.all(all_img_histogram != None):
            all_img_histogram = np.vstack((all_img_histogram, stack_img_histogram))
        else:
            all_img_histogram = stack_img_histogram

    # print(all_img_histogram.shape)

    # print(list_of_p_labels)
    plt.show()

    # print(all_img_histogram.shape)
    # print(stack_img_histogram.shape)

    return list_of_b_labels, list_of_c_labels, list_of_p_labels, all_img_histogram


list_of_b_labels, list_of_c_labels, list_of_p_labels, all_img_train_histogram = form_histograms(train_labels,
                                                                                                b_features_count_per_image,
                                                                                                c_features_count_per_image,
                                                                                                p_features_count_per_image,
                                                                                                1)



testing_path = os.getcwd() + os.sep + "Project2_data" + os.sep + "TestingDataset"

b_descriptors, c_descriptors, p_descriptors, b_features_count_per_image, c_features_count_per_image, p_features_count_per_image = find_sift_features(testing_path, 2)


stacked_descriptors_test = np.vstack((b_descriptors, c_descriptors, p_descriptors))

#print([i.shape for i in list_of_b_labels])

#Predict uses the same model and runs it on the testing features to acquire labels
model.fit_predict(stacked_descriptors_test)
stacked_test_labels = model.labels_

#print(type(b_labels_test))



list_of_b_test_labels, list_of_c_test_labels, list_of_p_test_labels, all_img_test_histogram = form_histograms(stacked_test_labels, b_features_count_per_image, c_features_count_per_image, p_features_count_per_image, 2)

butterfly_class_test = [1 for i in list_of_b_test_labels]
hat_class_test = [2 for i in list_of_c_test_labels]
plane_class_test = [3 for i in list_of_p_test_labels]

all_img_test_classes = butterfly_class_test.copy()
all_img_test_classes.extend(hat_class_test)
all_img_test_classes.extend(plane_class_test)


butterfly_class_train = [1 for i in list_of_b_labels]
hat_class_train = [2 for i in list_of_c_labels]
plane_class_train = [3 for i in list_of_p_labels]

all_img_train_classes = butterfly_class_train.copy()
all_img_train_classes.extend(hat_class_train)
all_img_train_classes.extend(plane_class_train)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
#all_img_train_histograms = np.array(all_img_train_histogram)
knn.fit(all_img_train_histogram, all_img_train_classes)
predictions = knn.predict(all_img_test_histogram)

from sklearn.metrics import accuracy_score

ac = accuracy_score(all_img_test_classes, predictions)
print(ac)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_img_test_classes, predictions)
print(cm)

#Linear vectoriztion
from sklearn.svm import LinearSVC

svcmodel = LinearSVC()
svcmodel.fit(all_img_train_histogram, all_img_train_classes)
predictions = svcmodel.predict(all_img_test_histogram)

ac = accuracy_score(all_img_test_classes, predictions)
print(ac)

cm = confusion_matrix(all_img_test_classes, predictions)
print(cm)


#Non linear vectorization
from sklearn.svm import SVC

lsvcmodel = SVC()
lsvcmodel.fit(all_img_train_histogram, all_img_train_classes)
predictions = lsvcmodel.predict(all_img_test_histogram)

ac = accuracy_score(all_img_test_classes, predictions)
print(ac)

cm = confusion_matrix(all_img_test_classes, predictions)
print(cm)