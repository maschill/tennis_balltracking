from matplotlib import pyplot as plt
import numpy as np
import cv2
from sympy import Point3D, Line3D, Plane


'''
From 3D position and estimated court edges 3D position of players and ball will be calculated
'''

def transform_params(R, T):
	'''
	Transform matrix for further calculations
	:param R: Rotation matrix array
	:param T: Translation matrix array
	:return: R, T, [R^T|-R^T@T]
	'''
	R = cv2.Rodrigues(R[0])[0]
	T = T[0].reshape(-1, 3)[0]
	F = np.zeros((3, 4))
	F[:, :-1] = np.transpose(R)
	F[:, -1] = -np.transpose(R) @ T
	return R,T,F

def get_3D_position(imgpoint, R, M, T, F, objpoint={'X': None, 'Y': None, 'Z': 0}, point=True):
	'''
	To reconstruct a scene from an image the matrix of camera calibration process are used.
	If either one X, Y or Z of 3D position is known we can calculate the real world point.
	If not two points defining the real world ray are calculated. The example shows calculation
	of world point / ray for (280, 825) which is the outer, left, nearby edge of the court.
	This edge was defined origin of world coordinate system (0,0,0)
	point = get_3D_position(imgpoint=[280, 825], R=R, M=M, T=T, F=F,
                        objpoint={'X':None, 'Y':None, 'Z':0})
	ray1 = get_3D_position(imgpoint=[280, 825], R=R, M=M, T=T, F=F,
                       objpoint={'X':None, 'Y':None, 'Z':1},
                       point=False)
	ray2 = get_3D_position(imgpoint=[280, 825], R=R, M=M, T=T, F=F,
                       objpoint={'X':None, 'Y':None, 'Z':None},
                       point=False)
	:param imgpoint: 2d image point [x, y]
	:param R: Rotation matrix
	:param M: Camera matrix
	:param T: Translation matrix
	:param F: [R^T|-R^T@T]
	:param objpoint: World coordinate if one is known
	:param point: if true return point; if false return two points defining a ray
	:return: world point(s)
	'''
	
	# Dict for vector position
	vti = {'X': 0, 'Y': 1, 'Z': 2}

	# If no world coordinate in known
	if objpoint['X'] == None and objpoint['Y'] == None and objpoint['Z'] == None:
		
		camerapt = np.linalg.inv(M) @ np.array([imgpoint[0], imgpoint[1], 1])
		rt = (np.transpose(R) @ T)
		a = 0 + rt[2]
		b = (np.transpose(R) @ camerapt)[2]
		s = a / b
		r1 = F @ np.append(camerapt * s, 1)

		s = 20
		r2 = F @ np.append(camerapt * s, 1)

		return r1, r2

	# If one world coordinate is known
	elif len([x for x in objpoint if objpoint[x] != None]) == 1:
		# Single world point
		if point:
			key = [x for x in objpoint if objpoint[x] != None][0]

			camerapt = np.linalg.inv(M) @ np.array([imgpoint[0], imgpoint[1], 1])
			rt = (np.transpose(R) @ T)
			a = objpoint[key] + rt[vti[key]]
			b = (np.transpose(R) @ camerapt)[vti[key]]
			s = a / b
			r = F @ np.append(camerapt * s, 1)

			return r

		# Two world points defininf ray
		else:
			key = [x for x in objpoint if objpoint[x] != None][0]
			camerapt = np.linalg.inv(M) @ np.array([imgpoint[0], imgpoint[1], 1])
			rt = (np.transpose(R) @ T)
			a = objpoint[key] + rt[vti[key]]
			b = (np.transpose(R) @ camerapt)[vti[key]]
			s = a / b
			r1 = F @ np.append(camerapt * s, 1)

			s = 20
			r2 = F @ np.append(camerapt * s, 1)

			return r1, r2
	else:
		print('Only one value of 3D Point can be set. Other two values must be None')

def get_trajectory(tr2d, R, M, T, F, Z_start=1, Z_end = 0):
	'''
	compute 3D trajectory for 2D image points.
	:param tr2d: 2d trajectory [[x1,y1],[x2,y2],...]
	:param R: Rotation matrix
	:param M: Camera matrix
	:param T: Translation matrix
	:param F: [R^T|-R^T@T]
	:param Z_start: Hight of first point in 2D trajectory
	:param Z_end: Hight of last point in 2D trajectory
	:return: array of 3D world points
	'''
	start_im = tr2d[0][1:3]
	end_im = tr2d[-1][1:3]
	start_3D = get_3D_position(imgpoint=start_im,
							   R=R, M=M, T=T, F=F,
							   objpoint={'X': None, 'Y': None, 'Z': Z_start},
							   point=True)
	end_3D = get_3D_position(imgpoint=end_im,
							 R=R, M=M, T=T, F=F,
							 objpoint={'X': None, 'Y': None, 'Z': Z_end},
							 point=True)
	print(start_3D, end_3D)
	# Get plane through the two points where Z is known
	X_ws, Y_ws, Z_ws = start_3D[0], start_3D[1], start_3D[2]
	X_we, Y_we, Z_we = end_3D[0], end_3D[1], end_3D[2]
	start = np.array([X_ws, Y_ws, Z_ws])
	ende = np.array([X_we, Y_we, Z_we])
	rvec = ende - start
	# normal
	nv = np.cross((rvec), [0, 0, 1])
	plane = Plane(Point3D(X_ws, Y_ws, Z_ws), normal_vector=(nv[0], nv[1], nv[2]))

	# Get point on plane for each point on 2D trajectory
	tr3D = []
	for point in tr2d:
		imgpoint = point[1:3]
		lfa1, lfa2 = np.array(get_3D_position(imgpoint=imgpoint,
											  R=R, M=M, T=T, F=F,
											  objpoint={'X': None, 'Y': None, 'Z': None},
											  point=False))

		line = Line3D(Point3D(lfa1[0], lfa1[1], lfa1[2]), Point3D(lfa2[0], lfa2[1], lfa2[2]))

		# Calculate intersect
		res = plane.intersection(line)
		res = [float(x) for x in res[0]]
		tr3D += [res]

	return tr3D

def get_time(tr2d, FPS=25):
	'''
	Compute duration of ball flight in seconds
	:param tr2d: array of 2d point [[x1, y1], [x2, y2],...]
	:param framerate: frame per second in video
	:return: flight time of ball in seconds
	'''
	start_frame = int(tr2d[0][0])
	end_frame = int(tr2d[-1][0])
	diff_frame = end_frame-start_frame
	time = diff_frame / FPS
	return time