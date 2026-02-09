import cv2
import numpy as np
# Loading source image
src_image = cv2.imread("seele.png")
# Defining the kernel of size 3x3
kernel = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])
 
resulting_image = cv2.filter2D(src_image, -1, kernel)
 
cv2.imshow("original image", src_image)
cv2.imshow("filter2d image", resulting_image)
cv2.imwrite("Filter2d Sharpened Image.jpg", resulting_image)
cv2.waitKey()
cv2.destroyAllWindows()