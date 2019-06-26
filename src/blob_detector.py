import numpy as np
import cv2


def blob_detector(im):
    # Read image
    # im = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread("blobTest.jpg", cv2.IMREAD_GRAYSCALE)
    
    if im.ndim != 3:
        #print("Current img: ", im.shape)
        im = im[0]
        #print("After: ", im.shape)
    """ Set up SimpleBlobDetector parameters """
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 200

    # Filter by Area
    params.filterByArea = True
    params.maxArea = 10

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Set up the detector with default parameters
    detector = cv2.SimpleBlobDetector_create()


    # Detect blobs
    keypoints = detector.detect(im)
    success = True
    try:
        #if keypoints is not None:
        #    print("target pos: ", keypoints[0].pt)
        return np.array(keypoints[0].pt), success
    except IndexError:
        print("Detector failed!!!!")
        success = False
        return np.zeros((1,2)), success
    # Draw detected blobs as red circles
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle
    # corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    # x1, y1 = keypoints[0].pt
    # x2, y2 = keypoints[1].pt
    # point_1 = tuple([int(i) for i in keypoints[0].pt])
    # point_2 = tuple([int(i) for i in keypoints[1].pt])
    # print("point_1: ", point_1)
    # print("point_2: ", point_2)

    # im_with_keypoints = cv2.circle(im_with_keypoints, point_1, 10, (86,255,255)) # opencv color: BGR
    # im_with_keypoints = cv2.circle(im_with_keypoints, point_2, 10, (86,255,255)) # opencv color: BGR
    # cv2.imshow("Keypoints: ", im_with_keypoints)
    # print("Keypoints: ", type(keypoints))
    # print("Length of keypoints: ", len(keypoints))
    # print(type(keypoints[0]))
    # print(keypoints[0].pt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
