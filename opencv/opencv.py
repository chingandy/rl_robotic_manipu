import cv2
import numpy as np

#
# def rescale_img(arr):
#     return (arr - arr.min()) * (1/(arr.max() - arr.min()) * 255).astype('uint8')



""" Read image """
# im = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
im = cv2.imread("test.png")


""" Resize image """
# scale_percent = 220 # percent of original size
# width = int(im.shape[1] * scale_percent / 100)
# height = int(im.shape[0] * scale_percent / 100)
# dim = (width, height)
# im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
# im = np.ones((128,128,3))
# im = rescale_img(im)
print(im.shape)
cv2.imshow("Original image", im)
cv2.waitKey(0)
print(im.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

""" Draw detected blobs as red circles """
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle
# corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Blob detected", im_with_keypoints)

""" Show keypoints """
x1, y1 = keypoints[0].pt
print(keypoints[0].pt)
print(keypoints[0].size)
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoints[0].size = 100
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Blob detected with larger size", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
