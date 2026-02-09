## answers to questions 
#
# 1. 
#
#
#
#
#


## code for convolution and mean filtering

import numpy as np
import cv2 as cv

a = np.array((1,2,3))
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])

img = cv.imread(r'seele.png', cv.IMREAD_GRAYSCALE)
print(len(img))
px = img[100,100]
print(px)
kern = np.ones((3,3))
print(kern)
print(img[0,0])

def convolution(img, kern):
    print("starting convolution")
    k = (int)(len(kern)/2)
    result = np.zeros((len(img), len(img)))
    for i in range(len(img)):
        for j in range(len(img)):
            for u in range(-k, k+1):
                for v in range(-k,k+1):
                    if(i-u < 0 or j-v < 0 or i-u >= len(img) or j-v >= len(img)):
                        result[i, j] = kern[u, v] * 0
                    else:
                        result[i, j] = kern[u, v] * img[i-u, j-v]
    return result

def meanfilter(img, kern):
    print("starting mean filtering")
    k = (int)(len(kern)/2)
    result = np.zeros((len(img), len(img)))
    for i in range(len(img)):
        for j in range(len(img)):
            for u in range(-k, k+1):
                for v in range(-k,k+1):
                    if(i-u < 0 or j-v < 0 or i-u >= len(img) or j-v >= len(img)):
                        result[i, j] += 0
                        # print(result[i,j])
                    else:
                        result[i, j] += img[i-u, j-v]
                        # print(result[i,j])
            # print("pixel at " + str(i) + ", "+ str(j) + ": " +str(result[i,j]))
            result[i, j] = result[i, j]/(len(kern) * len(kern))
    return result


diffimg = convolution(img, kern)
cv.imwrite(r'newimgconv.png',diffimg)

meanimg = meanfilter(img, kern)
cv.imwrite(r'newimgmean.png',meanimg)

white = np.ones((3,3))
for i in range(0,3):
    for j in range(0,3):
        white[i,j] = 255

cv.imwrite(r'white.png', white)
meanimg = meanfilter(white,kern)
cv.imwrite(r'whitemean.png',meanimg)