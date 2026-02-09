## answers to questions 
#
# Q1. 
# 1. the goal of computer vision is to understand the story a picture is telling
# 2. some examples of computer vision are: license plate readers, motion capture,
#    and robotics
# 3. an image is a matrix of 3 element arrays. it contains 3 ints that correspond
#    to the RGB values of the pixel
# Q2.
# 1. replace each pixel by a linear combination
# 2. convolution and correlation are both operations where a kernel is applied
#    to an image, however the kernel in convolution is flipped horizontally
#    and vertically
# Q3 work is in the file called Q3.png

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
kern = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])
print(kern)
print(img[0,0])

bigkern = np.array([
    [0,0,-1,0,0],
    [0,-1,-1,-1,0],
    [-1,-1,15,-1,-1],
    [0,-1,-1,-1,0],
    [0,0,-1,0,0]
])

biggestkern = np.array([
    [0,0,0,-1,0,0,0],
    [0,0,-1,-1,-1,0,0],
    [0,-1,-1,-1,-1,-1,0],
    [-1,-1,-1,25,-1,-1,-1],
    [0,-1,-1,-1,-1,-1,0],
    [0,0,-1,-1,-1,0,0],
    [0,0,0,-1,0,0,0]
])

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

src = cv.imread(r'seele.png')
gaussimg = cv.filter2D(src, -1, kern)
cv.imwrite(r'gaussimg.png',gaussimg)

gaussimg = cv.filter2D(src, -1, bigkern)
cv.imwrite(r'biggaussimg.png',gaussimg)

gaussimg = cv.filter2D(src, -1, biggestkern)
cv.imwrite(r'biggestgaussimg.png',gaussimg)

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