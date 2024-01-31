import numpy as np
import cv2 as cv
import imageio

grid = np.load('grid.npy')
result = imageio.imread('result.png')
result = np.asarray(result)
#result = np.swapaxes(result, 0, 2)
temp = np.copy(result)
result[..., 0] = result[..., 2]
result[..., 2] = temp[..., 0]


for circ in grid:
    center = (int(circ[0]),int(circ[1]) )
    radius = int(circ[2])
    
    sX = center[0] - radius
    bX = center[0] + radius
    sY = center[1] - radius
    bY = center[1] + radius
    cutOut = result[sY:bY + 1, sX:bX + 1] 
    print(cutOut.shape)
    
    #cv.imwrite('test.png',cutOut)
    #cv.circle( result, center, radius, (255, 0, 255), 3)






