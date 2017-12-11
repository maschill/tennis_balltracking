import os
import cv2
import sys

with open('../cutframes/cutframescoordinates.txt') as f:
	for line in f:
		line = line.split()
		line = [line[0]] + [float(f) for f in line[1:6]]
		print(line)
		image = cv2.imread('../frames/' + line[0])
		imshape = image.shape
		try:
			center_x, center_y = float(line[1]+line[3])*0.5/imshape[1], float(line[2]+line[4])*0.5/imshape[0] 
			newline = ("{} {} {} {} {}\n").format(0,center_x,center_y,float(line[3]-line[1])/imshape[1],float(line[4]-line[2])/imshape[0])
			outf = open('../YOLO/FramesAndTxt/pos-{}.txt'.format(line[0][6:11]), 'w')
			outf.write(newline)
			cv2.imwrite('../YOLO/FramesAndTxt/pos-{}.jpg'.format(line[0][6:11]), image)
		except IndexError:
			print('Error line empty', sys.exc_info()[0])
			pass

outf.close()