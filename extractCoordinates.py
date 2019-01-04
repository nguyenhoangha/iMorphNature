import os
import numpy as np
import math
import random
import cv2

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
	


center = Point(300,300)
posPs,negPs = extractCoors(center, 20, 40, 2, 2, 2)
print('Number of pos points' + str(len(posPs)))
print('Number of neg points' + str(len(negPs)))

img = np.zeros((512,512,3), np.uint8)
#cv2.circle(img,(447,63), 63, (0,0,255), -1)

for p in posPs:
	print(str(p.y))
 	#cv2.drawMarker(img, (p.x,p.y), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=2, thickness=1, line_type=cv2.LINE_AA)
for p in negPs:
	cv2.drawMarker(img, (p.x,p.y), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=2, thickness=1, line_type=cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()