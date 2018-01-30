import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

#Default Calibration GoPro Hero4 16:9 Wide
PathToCaliPoints = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/code/Kalibrierung_planar.txt'
def getCalibration(PathToCaliPoints=PathToCaliPoints, image_width=1920, image_height=1080):
	# Using planar calibration (court edges)
	imgpoints = np.array([[]]).reshape(0,2)
	objpoints = np.array([[]]).reshape(0,3)
	with open(PathToCaliPoints) as f:
	    for row in f:
	        imgp, objp = row.split(';')
	        imgp = [float(x.strip('))\n(')) for x in imgp.split(',')] 
	        objp = [float(x.strip('))\n(')) for x in objp.split(',')]
	        imgpoints = np.append(imgpoints, [imgp], axis=0)
	        objpoints = np.append(objpoints, [objp], axis=0)
	imgpoints = np.array(imgpoints)
	imgpoints = imgpoints.astype('float32')
	objpoints = np.array(objpoints)
	objpoints = objpoints.astype('float32')

	Parameters = np.zeros(5)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], img.shape[0:2], None, distCoeffs=Parameters)
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(image_width,image_height),1,(image_width,image_height))

	return mtx, dist, newcameramtx, roi

if __name__  == "__main__":
	if len(sys.argv) > 1:
		image = sys.argv[1]
		img = cv2.imread(image)
	else:
		# Use example image
		img = cv2.imread('/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames/image_GP_00001.png')
	h,w = img.shape[:2]
	mtx, dist, newcameramtx, roi = getCalibration()
	print(mtx, dist)

	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# crop the image
	x,y,w,h = roi
	#dst[y:y+h, x:x+w] = [0,0,0]#dst[y:y+h, x:x+w]
	dst[:y, :] = [0,0,0]
	dst[:, :x] = [0,0,0]
	dst[y+h:, :] = [0,0,0]
	dst[:, x+w:] = [0,0,0]
	images = np.hstack((img, dst))
	plt.figure(figsize=(19,8))
	plt.imshow(images)
	plt.show()