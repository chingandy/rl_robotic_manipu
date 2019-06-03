import cv2
import numpy as np


def rescale_img(arr):
    return (arr - arr.min()) * (1/(arr.max() - arr.min()) * 255).astype('uint8')
# Read image


im = cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE)


scale_percent = 220 # percent of original size
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
# im = np.ones((128,128,3))
# im = rescale_img(im)
print(im.shape)
cv2.imshow("none", im)
cv2.waitKey(0)
print(im.shape)

# im = cv2.imread("blobTest.jpg", cv2.IMREAD_GRAYSCALE)

""" Set up SimpleBlobDetector parameters """
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 200

# Filter by Area
params.filterByArea = True
params.maxArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.001

# Set up the detector with default parameters
detector = cv2.SimpleBlobDetector_create()


# Detect blobs
keypoints = detector.detect(im)

# Draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle
# corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Title", im_with_keypoints)

""" Show keypoints """
# x1, y1 = keypoints[0].pt
# print(np.array(keypoints[0].pt))
# x2, y2 = keypoints[1].pt
# point_1 = tuple([int(i) for i in keypoints[0].pt])
# point_2 = tuple([int(i) for i in keypoints[1].pt])
# print("point_1: ", point_1)
# print("point_2: ", point_2)

cv2.imshow("none", im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
