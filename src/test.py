# concatenate
# import numpy as np
# x = np.array([1,2,3,4])
# y = np.array([2,3])
# z = np.concatenate((x,y), axis=-1)
# print(z)


import blob_detector
import cv2
# import scipy.misc
import numpy as np
im = np.ndarray((128,128, 3), dtype=np.uint8)
# cv2.imshow("title", im)
# cv2.waitKey(0)

# im = scipy.misc.toimage(im)
# im = cv2.imread('test.png', cv2.IMREAD_COLOR)
# im = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
# print(type(im))
# print(im.shape)
# print(im.shape)
x = blob_detector.blob_detector(im)
print(x)



# from PIL import Image
# import numpy as np
# arr = np.zeros([5, 4, 3], dtype=np.uint8)
# arr[:,:] = [255, 128, 0]
# img = Image.fromarray(arr)
# print(type(img))

# import numpy as np
# x = np.zeros((10,2))
# y = np.zeros((10, 6))
# z = np.concatenate((x,y), axis=1)
# print(z.shape)
