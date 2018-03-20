from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import argparse
from sympy import Point, Point3D, Line, Line3D, Plane

import calibration

'''
From 3D position and estimated court edges 3D position of players and ball will be calculated
'''

def transform_params(R, T):
	R = cv2.Rodrigues(R[0])[0]
	T = T[0].reshape(-1, 3)[0]
	F = np.zeros((3, 4))
	F[:, :-1] = np.transpose(R)
	F[:, -1] = -np.transpose(R) @ T
	return R,T,F

def get_3D_position(imgpoint, R, M, T, F, objpoint={'X': None, 'Y': None, 'Z': 0}, point=True):
	
	# Dict for vector position
	vti = {'X': 0, 'Y': 1, 'Z': 2}

	if objpoint['X'] == None and objpoint['Y'] == None and objpoint['Z'] == None:
		#print('Return two 3D points which define line in 3D, Z=0 for p1')
		
		camerapt = np.linalg.inv(M) @ np.array([imgpoint[0], imgpoint[1], 1])
		rt = (np.transpose(R) @ T)
		a = 0 + rt[2]
		b = (np.transpose(R) @ camerapt)[2]
		s = a / b
		r1 = F @ np.append(camerapt * s, 1)

		s = 20
		r2 = F @ np.append(camerapt * s, 1)

		return r1, r2

	elif len([x for x in objpoint if objpoint[x] != None]) == 1:
		if point:
			key = [x for x in objpoint if objpoint[x] != None][0]
			#print('Return 3D Point for', key[0], 'given 3D value')

			camerapt = np.linalg.inv(M) @ np.array([imgpoint[0], imgpoint[1], 1])
			rt = (np.transpose(R) @ T)
			a = objpoint[key] + rt[vti[key]]
			b = (np.transpose(R) @ camerapt)[vti[key]]
			s = a / b
			r = F @ np.append(camerapt * s, 1)

			return r

		else:
			#print('Return two 3D points which define line in 3D, Z=0 for p1')
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

def get_trajectory(tr2d, R, M, T, F,):
	start_im = tr2d[0][1:3]
	end_im = tr2d[-1][1:3]
	start_3D = get_3D_position(imgpoint=start_im, R=R, M=M, T=T, F=F, objpoint={'X': None, 'Y': None, 'Z': 1}, point=True)
	end_3D = get_3D_position(imgpoint=end_im, R=R, M=M, T=T, F=F, objpoint={'X': None, 'Y': None, 'Z': 0}, point=True)
	print(start_3D, end_3D)
	# Get plane
	X_ws, Y_ws, Z_ws = start_3D[0], start_3D[1], start_3D[2]
	X_we, Y_we, Z_we = end_3D[0], end_3D[1], end_3D[2]
	start = np.array([X_ws, Y_ws, 0])
	ende = np.array([X_we, Y_we, Z_we])
	rvec = ende - start
	nv = np.cross((rvec), [0, 0, 1])
	plane = Plane(Point3D(X_ws, Y_ws, Z_ws), normal_vector=(nv[0], nv[1], nv[2]))

	# Get point on plane for each point on 2D trajectory
	tr3D = []
	for point in tr2d:
		imgpoint = point[1:3]
		lfa1, lfa2 = np.array(get_3D_position(imgpoint=imgpoint, R=R, M=M, T=T, F=F, objpoint={'X': None, 'Y': None, 'Z': 1}, point=False))
		# lfa2 = lfa1
		# np.array(get_3D_position(imgpoint=imgpoint, objpoint={'X':None, 'Y':None, 'Z': 10}))
		print(lfa1, lfa2)
		line = Line3D(Point3D(lfa1[0], lfa1[1], lfa1[2]), Point3D(lfa2[0], lfa2[1], lfa2[2]))

		# Calculate intersect
		res = plane.intersection(line)
		res = [float(x) for x in res[0]]
		tr3D += [res]

	return tr3D

def get_time(tr2d, framerate=25):
    start_frame = int(tr2d[0][0])
    end_frame = int(tr2d[-1][0])
    diff_frame = end_frame-start_frame
    time = diff_frame / framerate
    return time

if __name__  == "__main__":

	parser = argparse.ArgumentParser(description='Calculates camera calibration')
	parser.add_argument('--PathToCaliPoints', metavar='Path to calibration points', nargs=1,
	                    help='txt file with calibration parameters in form (60,899);(1.37,0,0) where first tuple is image point and second tuple is world point')
	parser.add_argument('--imgpath', metavar='Path to image', nargs=1,
	                    help='path to image which was used for defining calibration points, undistorted image will be shown.')
	parser.add_argument('--useexample', metavar='Use example image', help='Use example image and calibration points')
	parser.add_argument('--KerberHalep', metavar='Use Kerber vs. Halep Australian Open video', help='User Kerber Halep example')
	parser.add_argument('--GoPro', metavar='Use GoPro Video', help='Use GoPro Video as example')
	parser.add_argument('--ballpositions', metavar='Position of ball in video',
	                    help='Position of ball in frames which was returned from tensorflow object detetion API')
	args = parser.parse_args()
	print(args)
	if args.PathToCaliPoints is None and args.useexample is None:
		parser.error("if --useexample is None --PathToCaliPoints AND --imgpath is required.")

	if args.useexample is not None:
		# Use example image
		if args.KerberHalep is not None:
			imgpath = os.path.join(os.getcwd(), '../../Videos/GoPro/GoProFrames/3_image_GP_00306.png')
			image = plt.imread(imgpath)
			PathToCaliPoints = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/code/Kalibrierung_KerbHal_planar.txt'
		elif args.GoPro is not None:
			image = cv2.imread('/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames/image_GP_00001.png')
			image = plt.imread(imgpath)
			PathToCaliPoints = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/code/Kalibrierung_planar.txt'
		else:
			imgpath = os.path.join(os.getcwd(), '../../Videos/GoPro/GoProFrames/3_image_GP_00306.png')
			image = plt.imread(imgpath)
			PathToCaliPoints = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/code/Kalibrierung_KerbHal_planar.txt'
	else:
		image = args.imgpath
		img = cv2.imread(image)
		PathToCaliPoints = args.PathToCaliPoints

	#h, w = img.shape[:2]
	#ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints, newcameramtx, roi = calibration.getCalibration(PathToCaliPoints=PathToCaliPoints, image_width = w, image_height = h)
	ret, M, D, R, T, imgpoints, objpoints, newM, roi = getCalibration(PathToCaliPoints=PathToCaliPoints,
	                                                                  image_width=img.shape[1],
	                                                                  image_height=img.shape[0])
	R,T,F = transform_params(R=R,T=T)
	get_3D_position(imgpoint=[280, 825], R=R, M=M, T=T, F=F, objpoint={'X':None, 'Y':None, 'Z':0}, point=True)

	#calibration.getCalibration(PathToCaliPoints=PathToCaliPoints, image_width = 1920, image_height = 1080)

	# Get matrix for 2d to 3d projection

