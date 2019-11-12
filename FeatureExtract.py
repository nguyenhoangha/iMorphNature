# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:23:26 2018

@author: phuonglh
"""

import cv2
import numpy as np
from skimage.transform import integral_image
from sklearn.preprocessing import normalize
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
# Nh HAAR-LIKE features of random size and position are extracted inside each of the D windows, leading to Nh x D features
# the 4-type is used
def computeHAARLIKE(listImgs, listPoints, W, Nh, maxSizeHAAR):
    HAARFeature = []
    for i in range(len(listImgs)):
        # compute the integral image
        ii = integral_image(listImgs[i])
        for j in range(Nh):
            # generate the position of the random
            x = np.random.randint(-W,W) + listPoints[i][0]
            y = np.random.randint(-W,W) + listPoints[i][1]
            size = np.random.randint(0,maxSizeHAAR)
            # calculate the 4-type 
            HAARFeature.append(2*ii[y,x+size]+2*ii[y-size,x] + 2*ii[y+size,x]+2*ii[y,x-size] - ii[y-size,x+size] - 4*ii[y,x] - ii[y+size,x-size] - ii[y+size,x+size]-ii[y-size,x-size])                
    return HAARFeature

def computeFeatureList(imageFile,posFile, negFile, feature, NoScale, W, posFeatureFile, negFeatureFile, Nh = 8, gausSigma = 1.0, Ng = 10, maxSizeHAAR = 10):
    # rescale image
    listImgs = RescaleImage(imageFile,NoScale)
    # read positive point and negative point
    posF = open(posFile,"r")
    negF = open(negFile,"r")
    posFeatureF = open(posFeatureFile,"w")
    negFeatureF = open(negFeatureFile,"w")
    
    # for each positive point, calculate feature
    for line in posF:
        x,y = list(map(int,line.split()))
        listPoints = RescalePoint(x,y,NoScale)
        if (feature == "RAW"):
            des = computeRAW(listImgs,listPoints,W)
        elif (feature == "SSUB"):
            des = computeSignedSUB(listImgs, listPoints, W)
        elif (feature == "USUB"):
            des = computeUnsignedSUB(listImgs, listPoints, W)
        elif (feature == "GSUB"):
            des = computeGaussianSUB(listImgs, listPoints, gausSigma, Ng)
        elif (feature == "UGSUB"):
            des = computeUnsignedGaussianSUB(listImgs, listPoints, gausSigma, Ng)
        elif (feature == "HAAR"):
            des = computeHAARLIKE(listImgs, listPoints, W, Nh,maxSizeHAAR)
        mu = np.mean(des)
        std = np.std(des)
        des = (des - mu)/std
        for i in range(len(des)):
            posFeatureF.write(str(des[i]) + " ")
        posFeatureF.writelines("\n")  
    
    # for each negative point, calculate feature
    for line in negF:
        x,y = list(map(int,line.split()))
        listPoints = RescalePoint(x,y,NoScale)
        if (feature == "RAW"):
            des = computeRAW(listImgs,listPoints,W)
        mu = np.mean(des)
        std = np.std(des)
        des = (des - mu)/std
        for i in range(len(des)):
            negFeatureF.write(str(des[i]) + " ")
        negFeatureF.writelines("\n")  
        

# Dau vao diem: x: chieu ngang, y: chieu doc; shape[0]: chieu cao, shape[1]: chieu rong; img[y,x]
inputF = "C:/PhuongLH/ICTLab/Imorph/Code/001.bmp"
D = 5 # number of scale
W = 8 # window size is 2W+1
Nh = 8

imageFile = "C:/PhuongLH/ICTLab/Imorph/Data/Image_TPS/egfr_F_R_oly_2X_1.tif"
negFile = "C:/PhuongLH/ICTLab/Imorph/Code/Land_Mark_9/egfr_F_R_oly_2X_1_neg.txt"
posFile = "C:/PhuongLH/ICTLab/Imorph/Code/Land_Mark_9/egfr_F_R_oly_2X_1_pos.txt"
posFeatureFile = "C:/PhuongLH/ICTLab/Imorph/Code/Land_Mark_9_RAW/egfr_F_R_oly_2X_1_scale_5_W_8_pos.txt"
negFeatureFile = "C:/PhuongLH/ICTLab/Imorph/Code/Land_Mark_9_RAW/egfr_F_R_oly_2X_1_scale_5_W_8_neg.txt"
#computeFeatureList(imageFile,posFile,negFile,"RAW",D,W,posFeatureFile,negFeatureFile)
#computeFeatureList(imageFile,posFile,negFile,"SSUB",D,W,posFeatureFile,negFeatureFile)
#computeFeatureList(imageFile,posFile,negFile,"USUB",D,W,posFeatureFile,negFeatureFile)
#computeFeatureList(imageFile,posFile,negFile,"GSUB",D,W,posFeatureFile,negFeatureFile)
#computeFeatureList(imageFile,posFile,negFile,"UGSUB",D,W,posFeatureFile,negFeatureFile)
computeFeatureList(imageFile,posFile,negFile,"HAAR",D,W,posFeatureFile,negFeatureFile,Nh)

# HOG, BRISK, GLOH

#listImgs = RescaleImage(inputF,D)
#listPoints = RescalePoint(1256,227,D)
#print(listImgs[0][899,1439])
#w = computeRAW(listImgs, listPoints, W)
#w = computeHAARLIKE(listImgs,listPoints,Nh,5)
#print(w)