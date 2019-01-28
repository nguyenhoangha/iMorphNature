# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:40:40 2018

@author: phuonglh
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('006.bmp')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
F = open("006.txt","r")
i=0
for line in F:
    x,y = list(map(int,line.split()))
    i=i+1
    plt.scatter(x, y, s=100, c='red', marker='o')
    plt.annotate(i,xy=(x+10,y),color='r',weight='bold',size=20)
plt.show()
#plt.savefig('003Out.jpg',dpi=1000)

"""
plt.annotate('25, 50', xy=(25, 50), xycoords='data',
             xytext=(0.5, 0.5), textcoords='figure fraction',
             arrowprops=dict(arrowstyle="->"))
plt.scatter(25, 50, s=50, c='red', marker='o')

cv2.imwrite('test.png',img)
plt.show()

F = open("001.txt","r")
for line in F:
    x,y = list(map(int,line.split()))
    plt.scatter(x, 50, s=50, c='red', marker='o')

"""