import cv2
import matplotlib.pyplot as plt
import sys

path = '../frames/'
outpath = '../cutframes/test'
writefile = 'cutframescoordinates_test.txt'

# from: https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
def draw_rectangle(event,x,y,flags,param):
	global mouseXs,mouseYs,mouseXe,mouseYe, cropped
	if event == cv2.EVENT_LBUTTONDOWN:
		cropped = False
		mouseXs,mouseYs = x,y
	elif event == cv2.EVENT_LBUTTONUP:
		cropped = True
		mouseXe, mouseYe = x,y
		# Show image if image was cropped

		cv2.namedWindow("imagecut")
		cv2.imshow('imagecut', image[mouseYs:mouseYe, mouseXs:mouseXe])

#edges = []
endprogram = False
inprogress = True
f = open(outpath+writefile, 'a')


for i in range(int(sys.argv[1]), int(sys.argv[2]), int((int(sys.argv[2]) - int(sys.argv[1])) / 100)):
	inprogress = True

	if endprogram == True:
		break

	imgname = 'image_' + format(i, '05') + '.png'
	image = cv2.imread(path + imgname)
	cv2.namedWindow("image")
	cv2.imshow('image', image)

	while inprogress:
		cv2.setMouseCallback("image", draw_rectangle)

		# Check next step
		key = cv2.waitKey(0) & 0xFF

		# If area was selected, save image and coordiantes
		if key == ord('f') and cropped == True:
			#edges.append([imgname, mouseXs, mouseYs, mouseXe, mouseYe])
			#print(edges)
			print("{} {} {} {} {}\n".format(imgname, mouseXs, mouseYs, mouseXe, mouseYe))			
			f.write("{} {} {} {} {}\n".format(imgname, mouseXs, mouseYs, mouseXe, mouseYe))
			cv2.imwrite(outpath+'cut_'+imgname, image[mouseYs:mouseYe, mouseXs:mouseXe])
			inprogress = False

		# if no ball on image click d next will come up
		elif key == ord('d'):
			#edges.append([imgname])
			print("{}\n".format(imgname))
			f.write("{}\n".format(imgname))
			inprogress = False

		# stop cropping and save cropped eges to file
		elif key == ord('q'):
			print('Stopped cropping at image:', i)
			inprogress = False
			endprogram = True
			break
f.close()

#if interrupted write edges to file and end program
#pkl.dump(edges, open(outpath + writefile, 'wb'))
