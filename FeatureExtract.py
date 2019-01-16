# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:23:26 2018

@author: phuonglh
"""

import cv2
import numpy as np
#from skimage.feature import haar_like_feature

# function for rescale original images in different scales
def RescaleImage(imageFile, NoOfScale):
    #read input file
    img = cv2.imread(imageFile,cv2.IMREAD_GRAYSCALE)
    
    #rescale image by NoOfScale 
    listImgs = []
    
    for i in range(NoOfScale):
        img1 = cv2.resize(img, None, fx = 1/(2**i), fy = 1/(2**i))
        listImgs.append(img1)
        
    # return the list of images
    return listImgs

# xpos, ypos: horizontal and vertical cordinates of a point
def RescalePoint(xpos,ypos,NoOfScale):
    listPoints = []
    for i in range(NoOfScale):
        x = round(xpos/(2**i))
        y = round(ypos/(2**i))
        listPoints.append([x,y])
    return listPoints

# compute the RAW features
def computeRAW(listImgs, listPoints, W):
    RawFeature = []
    for i in range(len(listImgs)):
        for j in range(-W,W+1):
            for k in range(-W,W+1):
                x = listPoints[i][0]
                y = listPoints[i][1]
                RawFeature.append(listImgs[i][y+j,x+k])
    return RawFeature

# compute the SUB features
def computeSignedSUB(listImgs, listPoints, W):
    SUBFeature = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1]
        for j in range(-W,W+1):
            for k in range(-W,W+1):                            
                SUBFeature.append(listImgs[i][y+j,x+k].astype(np.int16)-listImgs[i][y,x].astype(np.int16))
    return SUBFeature

# compute the SUB features
def computeUnsignedSUB(listImgs, listPoints, W):
    SUBFeature = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1] 
        for j in range(-W,W+1):
            for k in range(-W,W+1):                           
                SUBFeature.append(np.abs(listImgs[i][y+j,x+k].astype(np.int16)-listImgs[i][y,x].astype(np.int16)))
    return SUBFeature

# compute the SURF features
def computeSURF(listImgs, listPoints, W):
    SURFFeature = []
    surf = cv2.xfeatures2d.SURF_create(extended=1)
    kps, des = surf.detectAndCompute(listImgs[0],None)
    #print(des.shape)
    #t = cv2.xfeatures2d_SURF
    #t.compute(listImgs[0],kps,des)

# compute the GAUSSIAN SUB descriptors
def computeGaussianSUB(listImgs, listPoints, gausSigma, Ng):
    GAUSSSUBFeature = []
    for i in range(len(listImgs)):
        xOff = np.random.normal(0,gausSigma,Ng)
        xOff = np.int16(xOff)
        yOff = np.random.normal(0,gausSigma,Ng)
        yOff = np.int16(yOff)
        x = listPoints[i][0]
        y = listPoints[i][1]
        for p in range(Ng):            
            GAUSSSUBFeature.append(listImgs[i][y+yOff[p],x+xOff[p]].astype(np.int16)-listImgs[i][y,x].astype(np.int16))
    return GAUSSSUBFeature

# compute the GAUSSIAN SUB descriptors
def computeUnsignedGaussianSUB(listImgs, listPoints, gausSigma, Ng):
    GAUSSSUBFeature = []
    for i in range(len(listImgs)):
        xOff = np.random.normal(0,gausSigma,Ng)
        xOff = np.int16(xOff)
        yOff = np.random.normal(0,gausSigma,Ng)
        yOff = np.int16(yOff)
        x = listPoints[i][0]
        y = listPoints[i][1]
        for p in range(Ng):            
            GAUSSSUBFeature.append(np.abs(listImgs[i][y+yOff[p],x+xOff[p]].astype(np.int16)-listImgs[i][y,x].astype(np.int16)))            
    return GAUSSSUBFeature

# compute the HAAR-LIKE descriptors
def computeHAARLIKE(listImgs, listPoints, Ng):
    print(listImgs[0].shape[0],listImgs[0].shape[1])
    print(listImgs[1].shape[0],listImgs[1].shape[1])
    print(listImgs[2].shape[0],listImgs[2].shape[1])
    print(listImgs[3].shape[0],listImgs[3].shape[1])
    print(listImgs[4].shape[0],listImgs[4].shape[1])
    

# Dau vao diem: x: chieu ngang, y: chieu doc
inputF = "C:/PhuongLH/ICTLab/Imorph/Code/001.bmp"
D = 5 # number of scale
W = 8 # window size is 2W+1

listImgs = RescaleImage(inputF,D)
listPoints = RescalePoint(1256,227,D)
w = computeRAW(listImgs, listPoints, W)
print(w)