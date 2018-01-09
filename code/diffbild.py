import numpy as np
import cv2
import os
import glob

# Create difference images from video frames
# Will be used for ball detection
# path: folder containing frames

path =  '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames'
imgtype = 'png'
imgname = 'image_GP_'
digits = '05'
outdir = '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFramesDiff/'

files = sorted(glob.glob(path + '/*.' + imgtype))
for i in range(1, len(files)):
	diff = cv2.absdiff(cv2.imread(files[i]), cv2.imread(files[i-1]))
	imgname = imgname + format(i, digits) + imgtype
	cv2.imwrite(outdir + imgname, diff)

'''
path = os.getcwd() + '/../tennis_top_cat_mouse.mp4'
print(path)

cap = cv2.VideoCapture(path)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    print('RET: ', ret)
    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 'q':
        break
    

cap.release()
cv2.destroyAllWindows()
''' 