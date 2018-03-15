import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def ReadTFOutput(file, image):
	ballpos = np.array([])
	with open(file, 'r') as f:
		for i, row in enumerate(f):
			# Tensorflow output not consistent
			row = row.replace('    ', ' ')
			row = row.replace('   ', ' ')
			row = row.replace('  ', ' ')

			# Get path to image, bounding box, accuracy, label and maximum of detected boxes
			path, bb, ac, lb, mx = row.split(';')

			# Since only one ball important to us calculate center of bounding box which has highest accuracy
			b1, b2, b3, b4 = [x.strip('[]') for x in bb.split(' ')[0:4]]
			b1, b3 = float(b1) * image.shape[0], float(b3) * image.shape[0]
			b2, b4 = float(b2) * image.shape[1], float(b4) * image.shape[1]
			accuracy = float(ac.split(' ')[0].strip('[]'))
			ballpos = np.append(ballpos, [i, (b2 + b4) / 2, (b1 + b3) / 2, accuracy])

	ballpos = ballpos.reshape(-1, 4)
	return ballpos

def MaxDistBallPerFramePXL(framerate, outleftnet_pxl, outrightnet_pxl, maxspeed_km_h=200):
	# Parameters for smoothing
	maxspeed_km_sec = maxspeed_km_h / 60 / 60
	maxspeed_m_sec = maxspeed_km_sec * 1000
	maxdist_m_frame = maxspeed_m_sec / framerate
	# Get number of pixels
	# Conservative assumption: ball moves orthogonal to optical axis at the opposite court side close to the net
	# use outer serve line points as reference distance

	dx = outleftnet_pxl[1] - outleftnet_pxl[0]
	dy = outrightnet_pxl[0] - outrightnet_pxl[1]
	physical_dist_m = 2 * np.sqrt(np.sqrt(((9.6 - 1.37) / 2) ** 2 + ((18.28 - 5.48) / 2) ** 2) ** 2 + 0.914 ** 2)

	maxdist_frame_x_pxl = maxdist_m_frame / physical_dist_m * dx
	maxdist_frame_y_pxl = maxdist_m_frame / physical_dist_m * dy
	mdist_pxl = int(np.sqrt(maxdist_frame_x_pxl ** 2 + maxdist_frame_y_pxl ** 2))
	print('Maximum distance of ball within one frame (x, y, both)', maxdist_frame_x_pxl, maxdist_frame_y_pxl,
	      np.sqrt(maxdist_frame_x_pxl ** 2 + maxdist_frame_y_pxl ** 2))
	return mdist_pxl

def smoothball(ballpos, ThesBallDetection, mdist_pxl):
	# Array will contain smoothed ball position
	smpos = []

	# We set the threshold for accuracy to 0.2, since we can assuma that on almost ever image a ball is visible
	# So we rather detect a false positiv than miss an important one while bouncing or hitting
	#ThesBallDetection = 0.2
	for pos in range(1, len(ballpos) - 1):
		framecount1 = 1
		framecount2 = 1

		# Check detection threshold and if predecessor box (cond2) and successor box (cond3)
		# or within reachble distance
		cond1 = ballpos[pos, 3] > ThesBallDetection
		cond2 = np.linalg.norm(ballpos[pos, 1:3] - ballpos[pos - 1, 1:3]) < mdist_pxl
		cond3 = np.linalg.norm(ballpos[pos + 1, 1:3] - ballpos[pos, 1:3]) < mdist_pxl
		if cond1 and (cond2 or cond3):
			j1 = pos - 1
			j2 = pos + 1
			dist1 = np.linalg.norm(ballpos[pos, 1:3] - ballpos[j1, 1:3])
			dist2 = np.linalg.norm(ballpos[j2, 1:3] - ballpos[pos, 1:3])

			# Check if predecessor and successor reach threshold and their distance is within range
			# if not next bounding box in according direction is picked
			while ballpos[j1, 3] < ThesBallDetection or dist1 > framecount1 * mdist_pxl:
				j1 -= 1
				framecount1 += 1
				if j1 < 0:
					j1 = None
					dist1 = None
					break
				else:
					dist1 = np.linalg.norm(ballpos[pos, 1:3] - ballpos[j1, 1:3])

			while ballpos[j2, 3] < ThesBallDetection or dist2 > framecount2 * mdist_pxl:
				j2 += 1
				framecount2 += 1
				if j2 >= len(ballpos):
					j2 = None
					dist2 = None
					break
				else:
					dist2 = np.linalg.norm(ballpos[pos, 1:3] - ballpos[j2, 1:3])

			# Calculate mean of three bounding boxes
			if j1 != None and j2 != None and dist1 != None and dist2 != None:
				smpos += [ballpos[pos, 0],
				          np.mean(ballpos[[j1, pos, j2], 1:3], axis=0)[0],
				          np.mean(ballpos[[j1, pos, j2], 1:3], axis=0)[1],
				          ballpos[pos, 3]]

	smpos = np.array(smpos).reshape(-1, 4)

	return smpos

def PlotSmoothVsNotSoSmooth(ballpos, smballpos, backgroundimage, startframe=665, seqlength=21):
	plt.imshow(backgroundimage)
	#k = 665
	oldrow = ballpos[startframe - 1]
	oldsmpos = ballpos[startframe - 1]
	for i, row in enumerate(ballpos[startframe:startframe+seqlength]):
		plt.scatter(row[1], row[2], c='b', alpha=.5)
		plt.arrow(oldrow[1], oldrow[2], row[1] - oldrow[1], row[2] - oldrow[2],
		          width=0.1, head_width=0.5, color='b', alpha=0.5)
		oldrow = row.copy()

		# Check if detected frame is part of smoothed values
		val = np.where(smballpos[:, 0] == row[0])
		if len(val[0]) > 0:
			plt.scatter(smballpos[int(val[0]), 1], smballpos[int(val[0]), 2], c='r', alpha=0.5)
			plt.arrow(oldsmpos[1], oldsmpos[2], smballpos[int(val[0]), 1] - oldsmpos[1],
			          smballpos[int(val[0]), 2] - oldsmpos[2],
			          width=1, head_width=5, color='r', alpha=0.5)
			oldsmpos = smballpos[int(val[0])].copy()
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Smooth Ball positions')
	parser.add_argument('--ballpositions', metavar='position of ball in video',
	                    help='textfile with position of ball in frames which was returned from tensorflow object detetion API')
	parser.add_argument('--imgpath', metavar='Path to image', nargs=1,
	                    help='path to image which was used for defining calibration points, '
	                         'undistorted image will be shown, e.g. ../../Videos/GoPro/GoProFrames/3_image_GP_00002.png')
	parser.add_argument('--startframe', metavar='frames used for example', default=665, nargs=1,
	                    help='Integer for frame number from which on example plot is shown')
	parser.add_argument('--seqlength', metavar='number of frames used for example', default=20, nargs=1,
	                    help='Integer defines number of frames from startframe which will be used for example plot')
	args = parser.parse_args()

	if args.ballpositions is not None:
		file = args.ballpositions
		image = plt.imread(args.imgpath)
	else:
		file = os.path.join(os.getcwd(), '../ballpositions.csv')
		num = format(args.startframe, '05d')
		path = '../../Videos/GoPro/GoProFrames/3_image_GP_' + num + '.png'
		print(num, path)
		image = plt.imread(os.path.join(os.getcwd(), path))

	# Example for Kerber Halep video
	ballpos = ReadTFOutput(file, image)
	mdist_pxl = MaxDistBallPerFramePXL(25, [535,1270], [640, 374], 160)
	smballpos = smoothball(ballpos, ThesBallDetection=0.2, mdist_pxl=mdist_pxl)
	PlotSmoothVsNotSoSmooth(ballpos, smballpos, image, startframe=args.startframe, seqlength=args.seqlength)
