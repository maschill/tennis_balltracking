import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import glob
import os

'''
Script for image annotation.

######Arguments######
annotation
	classes:  Classes can be changed and key can be added
	step: 	  use step to change number of images between images in video

input
	path: 	  path to folder with frames

	optional:
	a: start image
	b: end image

output
	outfile:  txt file where annotations will be saved

name of image that will be saved
	imggroup: name of images that will be saved
	imgtype: .png is default
	digits:  number of digits save
'''

classes = {ord('d'): 'Bounce', ord('s'): 'Serve', ord('h'): 'Hit', ord('f'):'Nothing'}
step = 1
pathFrames = '../../Videos/GoPro/GoProFrames/'
outfile = '../annotations/BounceServeHit.txt'
imggroup = '3_image_GP_'
imgtype = '.png'
digits = '05'

imagelist = sorted(glob.glob(os.path.join(os.getcwd(), pathFrames) + imggroup + '*' + imgtype))
of = open(os.path.join(os.getcwd(), outfile), 'a')
print(imagelist[0:5])
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 800)

i = 3324 #1938 #310
while i < len(imagelist)-1:
    print(i)
    image = cv2.imread(imagelist[i])
    cv2.imshow('image', image)

    key = cv2.waitKey(0) & 0xFF

    if key == 83:
        i += 1
    elif key == 81:
        i -= 1
    elif key == ord('p'):
        i += 100
    elif key == ord('o'):
        i -= 100
    elif key == ord('h'):
        of.write('{} {} \n'.format(imagelist[i].split('/')[-1], classes[key]))
        i += 1
    elif key == ord('f'):
        of.write('{} {} \n'.format(imagelist[i].split('/')[-1], classes[key]))
        i += 1
    elif key == ord('d'):
        of.write('{} {} \n'.format(imagelist[i].split('/')[-1], classes[key]))
        i += 1
    elif key == ord('s'):
        of.write('{} {} \n'.format(imagelist[i].split('/')[-1], classes[key]))
        i += 1
    elif key == ord('q'):
        print('Stopped cropping at image:', imagelist[i])
        break

of.close()