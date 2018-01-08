import numpy as np
import cv2
import os
import glob
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
path =  '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames'
files = sorted(glob.glob(path + '/*.png'))
print(files)
#fgbg = cv2.createBackgroundSubtractorMOG2()
#path = os.getcwd() + '/../tennis_top_cat_mouse.mp4'
#cap = cv2.VideoCapture(path)
for i in range(1, len(files)):
#while(1):
	#ret, frame = cap.read()
	#ret_, frame_ = cap.read()
	#diff = cv2.acsdiff(frame_, frame)
	diff = cv2.absdiff(cv2.imread(files[i]), cv2.imread(files[i-1]))
	imgname = 'image_GP_' + format(i, '05') + '.png'
	print(imgname)
	cv2.imwrite('/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFramesDiff/' + imgname, diff)
	#image = cv2.imread(files[i])
	#fgmask = fgbg.apply(image)
	#cv2.imshow('diff', diff)
	#plt.show()

	#k = cv2.waitKey(20) & 0xff
	#if k == 'q':
	#	break

#cv2.destroyAllWindows()

