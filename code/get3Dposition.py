from matplotlib import pyplot as plt
import numpy as np
'''
From 3D position and estimated court edges 3D position of players and ball will be calculated
'''
#path = '/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames'
#coords = []
#with open('/home/lea/Dokumente/FSU/Anwendungspraktikum/points.txt', 'r') as f:
#	for row in f:
#		for coord in row.split('; '):
#			x,y = coord[1:-1].split(', ')
#			coords.append([int(x), int(y)])

#coords = np.array(coords)

#f = 18
#plt.plot(coords[:f,0], coords[:f,1], 'r.')
#plt.scatter(coords[f:,0], coords[f:,1])
#plt.gca().invert_yaxis()
#plt.show()

# Dimension of tennis court in meter
# S: Single, d: Double

SSide = 8.23 # Distance between single sideline
DSide = 10.97
ServNet = 6.40
BaseNet = 11.89

# Radiale Verzerrung GoPro - parameters not working, own calibration necessary
# http://argus.web.unc.edu/camera-calibration-database/
F = 1788
W = 2704
H = 1520
cx = 1351.5
cy = 759.5
k1 = -0.2583
k2 = 0.081 #0.0770

#path = '/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames'
#image = plt.imread(path + '/image_GP_00001.png')
#plt.imshow(image)
#plt.show()
KP = np.array([])
with open('Kalibrierung.txt') as f:
	for row in f:
		img, real = row.split(';')
		img = [int(x) for x in img.split(',')] 
		real = [int(x) for x in real.split(',')]
		KP.append([img, real]) 

# We need to find extrinsic (rotation matrix and translation) and intrinsic (brennweite, pixel size, image center)
# we define alpha = s_y / s_x; f_x = f/s_x, f_y = f/s_y

def radDist(image):
	imgnew = np.zeros(image.shape)
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			x_ = x + x*(k1*(x**2+y**2)+k2*(x**2+y**2)**2)
			y_ = y + y*(k1*(x**2+y**2)+k2*(x**2+y**2)**2)
			#print(x_, y_)
			try:
				imgnew[int(x_),int(y_)] = image[x,y]
			except IndexError:
				pass
	return imgnew

def calibration(m):
	R = np.zeros([3,3])
	#define central point of goProFrame as imagecenter
	o_x, o_y = image.shape[0]/2, image.shape[1]/2
	#shift image points to new origin
	[for x in matches[:,0] x - [o_x,o_y]]
	A = np.zeros([8,8])
	b = np.zeros(8)
	for i in range(8):
		A[i] = [m[i,0,0]*m[i,1,0], m[i,0,0]*m[i,1,1],m[i,0,0]*m[i,1,2], m[i,0,0], -m[i,0,1]*m[i,1,0],-m[i,0,1]*m[i,1,1],-m[i,0,1]*m[i,1,2],-m[i,0,1]] 
	#Solve for r21, r22, r23, t_y, alpha*r11, alpha*r12, alpha*r13, alpha*t_x
	v = np.linalg.solve(A, b)
	t_y = v[3]
	R[1] = v[:3]

	#Calculate scale v_ = gamma*v
	gamma = abs(sqrt(v[0]**2+v[1]**2+v[2]**2))
	alpha = sqrt(v[4]**2+v[5]**2+v[6]**2) / gamma
	#third row of rotation matrix
	t_y = v[7]/alpha
	R[0] = v[4:7]/alpha
	R[2] = np.cross(R[0],R[1])

	#get sign
	if (R[0]*m[0,1]+t_y)*m[0,0,0] > 0:
		R[0] *= -1
		R[1] *= -1
		t_x *= -1
		t_y *= -1

	#Since Rotation matrix not orthogonal because of noise perform SVD
	U,D,V = np.linalg.svd(R)
	R = U @ np.diag(np.ones(len(D)))@V

	return R,t_x,t_y

#NonRad = radDist(image)

#plt.imshow(NonRad)
#plt.show()

def get3D(Pos, calibration):
	return 0

