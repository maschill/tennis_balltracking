import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse


#Default Calibration GoPro Hero4 16:9 Wide
# PathToCaliPoints = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/code/Kalibrierung_planar.txt'
def getCalibration(PathToCaliPoints, image_width=1920, image_height=1080):
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
            print(imgp, objp)
    imgpoints = np.array(imgpoints)
    imgpoints = imgpoints.astype('float32')
    objpoints = np.array(objpoints)
    objpoints = objpoints.astype('float32')

    Parameters = np.zeros(5)
    # Returns camera matrix, distortion coefficients, rotation and translation vector
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints],
                                                       (image_height,image_width), None, distCoeffs=Parameters)

    # Returns the new camera matrix based on the free scaling parameter
    # Alpha set to one to keep all image points, since there can be valuable information in corners
    # Returns new camera matrix and ROI which can be used to crop image
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(image_width,image_height),1,(image_width,image_height))

    return ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints, newcameramtx, roi

if __name__  == "__main__":

	parser = argparse.ArgumentParser(description='Calculates camera calibration')
	parser.add_argument('--PathToCaliPoints', metavar='Path to calibration points', nargs=1,
	                    help='txt file with calibration parameters in form (60,899);(1.37,0,0) where first tuple is image point and second tuple is world point')
	parser.add_argument('--imgpath', metavar='Path to image', nargs=1,
	                    help='path to image which was used for defining calibration points, undistorted image will be shown.')
	parser.add_argument('--useexample', metavar='Use example image', help='Use example image and calibration points')

	args = parser.parse_args()
	print(args)
	if args.PathToCaliPoints is None and args.useexample is None:
		parser.error("if --useexample is None --PathToCaliPoints AND --imgpath is required.")

	if args.useexample is not None:
		# Use example image
		exampleImagePath = '/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames/image_GP_00001.png'
		img = cv2.imread(exampleImagePath)
		PathToCaliPoints = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/code/Kalibrierung_planar.txt'
		print('Example Image and example calibration Points:', exampleImagePath, PathToCaliPoints)
	else:
		image = args.imgpath
		img = cv2.imread(image)
		PathToCaliPoints = args.PathToCaliPoints

	h,w = img.shape[:2]
	ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints, newcameramtx, roi = getCalibration(PathToCaliPoints, image_width=w, image_height=h)
	# print(mtx, dist)

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
	images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
	plt.figure(figsize=(19,8))
	plt.imshow(images)
	plt.show()

	return