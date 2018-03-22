import cv2
import os
import argparse

# This script can be used to check bounding boxes after annotation

parser = argparse.ArgumentParser(description='Calculates camera calibration')
parser.add_argument('--path', 
		    default='/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames',
		    metavar='Path image folder')
parser.add_argument('--annotationfile',
		    default='/home/mueller/code/python/Anwendungspraktikum/cvtennis/annotations/GoProBall_train.txt',
		    metavar='Path to txt file with annotations')

args = parser.parse_args()

path = args.path

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 800)

# Check if next image is 
with open(args.annotationfile, 'r') as f:
	for i, row in enumerate(f):
		name, label, x1, y1, x2, y2 = row.split(' ')
		image = cv2.imread(os.path.join(path, name))
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image, label, (x1,y1), font, 1, (0,0,255), 1, cv2.LINE_AA)
		cv2.imshow('image', image)
		if cv2.waitKey(200) & 0xFF == ord('q'):
			break
