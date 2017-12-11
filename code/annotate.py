import cv2
import matplotlib.pyplot as plt
import sys

#~/Dokumente/FSU/Anwendungspraktikum/cvtennis/tensorflow/models/ssd_inception_v2_coco_2017_11_17
#~/Dokumente/FSU/Anwendungspraktikum/cvtennis/tensorflow/models/ssd_inception_v2_coco_2017_11_17
path = '../GoProFrames/GoProFramesDiff/'
outpath = '../GoProFrames/GoProFramesDiff/'
writefile = 'GoProDiffAnnotation.txt'

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

#edges = []
endprogram = False
inprogress = True
f = open(outpath+writefile, 'a')

classes = {ord('f'): 'Ball', ord('d'): 'Spieler', ord('t'): 'Aufschlag', ord('v'): 'Tennisschlaeger'}
step = 1

for i in range(int(sys.argv[1]), int(sys.argv[2]), step):

	inprogress = True

	if endprogram == True:
		break

	imgname = 'image_GP_' + format(i, '05') + '.png'
	image = cv2.imread(path + imgname)
	copy = image.copy()
	cv2.namedWindow('image')
	cv2.imshow('image', image)

	annotations = []

	while inprogress:
		cv2.setMouseCallback("image", draw_rectangle)
		#print('Mouse captured')
		# Check next step
		key = cv2.waitKey(0) & 0xFF
		#print('Key captured', key)
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
			#print(annotations)

			#login frame and class or change class to new one
			label_wrong = True
			while label_wrong:
				#print('in loop', label)
				key_ano = cv2.waitKey(0) & 0xFF
				if key_ano == ord('g'):
					#print('in loop if', label)
					label_wrong = False
					image = copy.copy()
					copy = copy.copy()
					pass
				elif key_ano in classes:
					#print('in loop elif', label)
					copy = test.copy()
					label = classes[key_ano]
					cv2.putText(copy, label, (mouseXs,mouseYs), font, 1, (0,0,255), 1, cv2.LINE_AA)
					cv2.rectangle(copy, (mouseXs, mouseYs), (mouseXe, mouseYe), (0,0,255), 1)
					cv2.imshow('image', copy)
					annotations[-1] = "{} {} {} {} {} {}\n".format(imgname, classes[key_ano], mouseXs, mouseYs, mouseXe, mouseYe)
					#print(annotations)

		# if no ball on image click s next will come up
		elif key == ord('s'):
			for ano in annotations:
				print(ano)
				f.write(str(ano))

				#print("{} {} {} {} {} {}\n".format(imgname, classes[key], mouseXs, mouseYs, mouseXe, mouseYe))			
				#f.write("{} {} {} {} {} {}\n".format(imgname, classes[key], mouseXs, mouseYs, mouseXe, mouseYe))
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

#if interrupted write edges to file and end program
#pkl.dump(edges, open(outpath + writefile, 'wb'))
