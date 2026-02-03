import numpy as np
import cv2 as cv

a = np.array((1,2,3))
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])

img = cv.imread(r'seele.png')
print(len(img))
px = img[100,100]
print(px)
kern = np.ones((3,3))
print(kern)
print(img[0,0,0])

def convolution(img, kern):
    print("starting convolution")
    k = (int)(len(kern)/2)
    result = np.zeros((len(img), len(img)))
    for i in range(len(img)):
        for j in range(len(img)):
            for u in range(-k, k):
                for v in range(-k,k):
                    if(i-u < 0 or j-v < 0 or i-u >= len(img) or j-v >= len(img)):
                        result[i, j] = kern[u, v] * 0
                    else:
                        result[i, j] = kern[u, v] * img[i-u, j-v, 1]
    return result

diffimg = convolution(img, kern)
cv.imwrite(r'newimgg.png',diffimg)