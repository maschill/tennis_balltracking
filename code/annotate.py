import cv2
import matplotlib.pyplot as plt
import sys

'''
Script for image annotation.

######Arguments######
annotation
	classes:  Classes can be changed and key can be added
	step: 	  use step to change number of images between images in video

input
	path: 	  path to folder with frames

output
	outfile:  txt file where annotations will be saved

name of image that will be saved
	imggroup: name of images that will be saved
	imgtype: .png is default
	digits:  number of digits save
'''

classes = {ord('f'): 'Ball', ord('d'): 'Spieler', ord('t'): 'Aufschlag', ord('v'): 'Tennisschlaeger'}
step = 15
path = '../../Videos/GoPro/GoProFrames/'
outfile = '../annotations/GoProAnnotationSpieler_test.txt'
imggroup = 'image_GP_'
imgtype = '.png'
digits = '05'

# from: https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
def draw_rectangle(event,x,y,flags,param):
	global mouseXs, mouseYs, mouseXe, mouseYe, cropped
	if event == cv2.EVENT_LBUTTONDOWN:
		cropped = False
		mouseXs,mouseYs = x,y

		cv2.imshow('image', image)

	elif event == cv2.EVENT_LBUTTONUP:
		cropped = True
		mouseXe, mouseYe = x,y
		# Show image if image was cropped

		copy = image.copy()
		cv2.rectangle(copy, (mouseXs, mouseYs), (mouseXe, mouseYe), (0,0,255), 1)
		cv2.imshow('image', copy)
		#cv2.namedWindow("imagecut")
		#cv2.imshow('imagecut', image[mouseYs:mouseYe, mouseXs:mouseXe])

def draw():
	cv2.rectangle(image, (mouseXs, mouseYs), (mouseXe, mouseYe))


endprogram = False
inprogress = True
f = open(outfile, 'a')

for i in range(int(sys.argv[1]), int(sys.argv[2]), step):

	inprogress = True

	if endprogram: break

	imgname = imggroup + format(i, digits) + imgtype
	print(path + imgname)
	image = cv2.imread(path + imgname)
	copy = image.copy()
	cv2.namedWindow('image')
	cv2.imshow('image', image)

	annotations = []

	while inprogress:
		cv2.setMouseCallback("image", draw_rectangle)

		# Check next step
		key = cv2.waitKey(0) & 0xFF

		# If area was selected, save image and coordiantes
		if key in classes and cropped == True:

			#bounding box set, make image new image
			image = copy.copy()
			test = copy.copy()
			#now select class
			font = cv2.FONT_HERSHEY_SIMPLEX
			label = classes[key]

			#second copy to correct label
			cv2.putText(copy, label, (mouseXs,mouseYs), font, 1, (0,0,255), 1, cv2.LINE_AA)
			cv2.rectangle(copy, (mouseXs, mouseYs), (mouseXe, mouseYe), (0,0,255), 1)
			cv2.imshow('image', copy)
			
			annotations.append("{} {} {} {} {} {}\n".format(imgname, classes[key], mouseXs, mouseYs, mouseXe, mouseYe))

			#login frame and class or change class to new one
			label_wrong = True
			while label_wrong:
				key_ano = cv2.waitKey(0) & 0xFF
				if key_ano == ord('g'):
					label_wrong = False
					image = copy.copy()
					copy = copy.copy()
					pass
				elif key_ano in classes:
					copy = test.copy()
					label = classes[key_ano]
					cv2.putText(copy, label, (mouseXs,mouseYs), font, 1, (0,0,255), 1, cv2.LINE_AA)
					cv2.rectangle(copy, (mouseXs, mouseYs), (mouseXe, mouseYe), (0,0,255), 1)
					cv2.imshow('image', copy)
					annotations[-1] = "{} {} {} {} {} {}\n".format(imgname, classes[key_ano], mouseXs, mouseYs, mouseXe, mouseYe)

		# if no ball on image click s next will come up
		elif key == ord('s'):
			for ano in annotations:
				print(ano)
				f.write(str(ano))
				#print("{} {} {} {} {} {}\n".format(imgname, classes[key], mouseXs, mouseYs, mouseXe, mouseYe))			
			inprogress = False

		# stop cropping and save cropped eges to file
		elif key == ord('q'):
			print('Stopped cropping at image:', i)
			inprogress = False
			endprogram = True
			break

		elif key == ord('p'):
			i += 100

		elif key == ord('o'):
			i -= 100

f.close()