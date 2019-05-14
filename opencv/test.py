x = [21.3, 1.44, 25.9, 4.98]
x_new = [int(i) for i in x]
print(x_new)

y = (21.3, 1.44, 25.9, 4.98)
y_new = [int(j) for j in y]
y_new = tuple(y_new)
print(y_new)




# import cv2
# import numpy as np
#
# test_img = np.array(b[[255,0],[0,255]])
# cv2.imwrite("test_img.jpg", test_img)
# cv2.imshow("test image", test_img)
# cv2.waitKey(0)
