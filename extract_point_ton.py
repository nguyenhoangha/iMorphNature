# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:26:47 2019

@author: HuuTon
"""

import glob
import os
import numpy as np
import math
import random
class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

def extractCoors( p, R, Rmax, s, W, P):
	positive = []
	negative = []
	for dx in range(-R,R, 2):
		dy_max = math.floor(math.sqrt(R*R - dx*dx))
		positive.append(Point(p.x + dx, p.y))
		for dy in range(2, dy_max, 2):
			positive.append(Point(p.x + dx, p.y- dy))
			positive.append(Point(p.x + dx, p.y+ dy))

	negCandidates = []
	for dx in range(-Rmax, -R):
		dy_max = math.floor(math.sqrt(Rmax*Rmax - dx*dx))
		for dy in range(-dy_max, dy_max):
			negCandidates.append(Point(p.x + dx, p.y+ dy))
	for dx in range(-R, R):
		dy_max = math.floor(math.sqrt(Rmax*Rmax - dx*dx))
		dy_min = math.ceil(math.sqrt(R*R - dx*dx))
		for dy in range(dy_min, dy_max):
			negCandidates.append(Point(p.x + dx, p.y- dy))
			negCandidates.append(Point(p.x + dx, p.y+ dy))
	for dx in range(R, Rmax):
		dy_max = math.floor(math.sqrt(Rmax*Rmax - dx*dx))
		for dy in range(-dy_max, dy_max):
			negCandidates.append(Point(p.x + dx, p.y+ dy))
	negative = random.sample(negCandidates, P*len(positive))
	#negative = negCandidates
	return positive, negative

for name in glob.glob("F:\Imorph\Ton_done\*.tps"):
    name1=name[0:(len(name)-4)]
    pos_name=name1+'_pos_9.txt'
    neg_name=name1+'_neg_9.txt'
    pos_file=open(pos_name,'w')
    neg_file=open(neg_name,'w')
    with open(name) as f:
        lines = f.readlines()
        x,y = list(map(float,lines[9].split()))
        center = Point(int(x),int(y))
        posPs,negPs = extractCoors(center, 20, 40, 2, 2, 2)
        print('Number of pos points' + str(len(posPs)))
        print('Number of neg points' + str(len(negPs)))
        for p in posPs:
            pos_file.write(str(p.x)+' '+str(p.y)+ '\n')
        for p in negPs:
            neg_file.write(str(p.x)+' '+str(p.y)+ '\n')
    pos_file.close()
    neg_file.close()
        