from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
import math

path = '/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames'
coords = []
with open('/home/lea/Dokumente/FSU/Anwendungspraktikum/points.txt', 'r') as f:
	for row in f:
		for coord in row.split('; '):
			x,y = coord[1:-1].split(', ')
			coords.append([int(x), int(y)])

f = 18 #first f values represent court
court = np.array(coords)
coords = np.array(coords[18:])

#calculate line through two points
def l(p1,p2):
	return p2-p1

#calulate angular between two lines
def angular(r1, r2): # r1, r2 direction vectors of line
	alphaRad = math.acos(abs(np.dot(r1, r2)) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
	alpha = math.degrees(alphaRad)
	return alpha

cp = np.zeros(len(coords))
for i in range(len(coords)-2):
	print(l(coords[i], coords[i+1]), l(coords[i+1], coords[i+2]))
	if np.any(l(coords[i], coords[i+1])) > 0 and np.any(l(coords[i+1], coords[i+2])) > 0: #make sure not to divide by zero
		alpha = angular(l(coords[i], coords[i+1]), l(coords[i+1], coords[i+2]))
		if alpha < 90:
			cp[i] = 1
	else: # ball did not move in image
		pass

#plt.plot(court[:18,0], court[:18,1], 'r')
col = Colormap('col')
plt.plot(coords[:,0], coords[:,1],'.')
for i in range(len(cp)):
	if cp[int(i)] == 1:
		print(coords[int(i),0], cp[int(i)])
		plt.scatter(coords[int(i),0], coords[int(i),1])
plt.gca().invert_yaxis()
#plt.scatter(coords[:,0], cp)
plt.show()