import cv2
import numpy as np

# Read image
im = cv2.imread("blobTest.jpg", cv2.IMREAD_GRAYSCALE)

""" Set up SimpleBlobDetector parameters """
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area
params.filterByArea = False
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Set up the detector with default parameters
detector = cv2.SimpleBlobDetector_create()


# Detect blobs
keypoints = detector.detect(im)

# Draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle
# corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
# x1, y1 = keypoints[0].pt
# x2, y2 = keypoints[1].pt
point_1 = tuple([int(i) for i in keypoints[0].pt])
point_2 = tuple([int(i) for i in keypoints[1].pt])

im_with_keypoints = cv2.circle(im_with_keypoints, point_1, 10, (86,255,255)) # opencv color: BGR
im_with_keypoints = cv2.circle(im_with_keypoints, point_2, 10, (86,255,255)) # opencv color: BGR
cv2.imshow("Keypoints: ", im_with_keypoints)
# print("Keypoints: ", type(keypoints))
# print("Length of keypoints: ", len(keypoints))
# print(type(keypoints[0]))
# print(keypoints[0].pt)
cv2.waitKey(0)
cv2.destroyAllWindows()
