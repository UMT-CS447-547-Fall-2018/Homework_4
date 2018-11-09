from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

# I crashed this afternoon and didn't have any time to 
# implement a sampling version, this uses the potentially
# degenerate [0,0,0], [0,0,255], ... base means

def distance(x,y):
    dist = 0
    for xVal, yVal in zip(x,y):
        dist += (xVal - yVal)**2
    return np.sqrt(dist)
def computeDists(image, means):
    rows = image.shape[0]
    cols = image.shape[1]
    numMeans = len(means)
    distMat = np.zeros((rows, cols, numMeans))
    for row in range(rows):
        for col in range(cols):
            for i in range(numMeans):
                distMat[row][col][i] = distance(image[row][col], means[i])
    return distMat

def assignClasses(image, means, init, dists=None):
    rows = image.shape[0]
    cols = image.shape[1]
    numMeans = means.shape[0]
    if init:
        # Assign classes randomly
        dists = computeDists(image, [[0,0,0],[0,0,255],[0,255,0],[255,0,0],[255,255,0],[255,0,255],[0,255,255], [255,255,255]])
        #Z = np.random.randint(0,numMeans,size = (rows,cols))
        #return Z
    #else:
    Z = np.zeros((rows, cols),dtype=int)
    for row in range(rows):
        for col in range(cols):
            Z[row][col] = int(np.argmin(dists[row][col]))
    return Z

def computeMeans(image, classes, numMeans):
    means = np.zeros((numMeans,3))
    freqs = np.zeros(numMeans)
    rows = image.shape[0]
    cols = image.shape[1]
    for row in range(rows):
        for col in range(cols):
            classVal = classes[row][col]
            means[classVal] += image[row][col]
            freqs[classVal] +=1
    for i in range(numMeans):
        means[i] *= 1.0/freqs[i]
    print means
    print
    return means

def kMeans(image, numMeans, numIterations, init = True, classes = None, means = None):
    rows = image.shape[0]
    cols = image.shape[1]
    if numIterations == 0:
        for row in range(rows):
            for col in range(cols):
                image[row][col] = means[classes[row][col]]
        return image
    elif init:
        means = np.zeros((numMeans, 3))
        classes = assignClasses(image, means, init)
        means = computeMeans(image, classes, numMeans)
        return kMeans(image, numMeans, numIterations-1, False, classes, means)
    else:
        dists = computeDists(image,means)
        classes = assignClasses(image, means, init, dists)
        means = computeMeans(image, classes, numMeans)
        return kMeans(image, numMeans, numIterations-1, False, classes, means)


im = imread('Lola_s_s.jpg')

plt.imshow(im)
plt.show()
im2 = kMeans(im, 8, 5)
plt.imshow(im2)
plt.show()
