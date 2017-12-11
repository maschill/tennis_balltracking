import cv2

path = '/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/diff_frames/'

cv2.namedWindow('image')

actual_row = None

file = open('/home/lea/Dokumente/FSU/Anwendungspraktikum/cvtennis/cutframes/diffframes_annotation.txt', 'r')
for i in range(1000):
	row = file.readline()
	name, label, x1, y1, x2, y2 = row.split(' ')
	actual_row = name

	image = cv2.imread(path + name)
	cv2.imshow('image', image)

	if actual_row == name:
		name, label, x1, y1, x2, y2 = row.split(' ')
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image, label, (x1,y1), font, 1, (0,0,255), 1, cv2.LINE_AA)
		image = image.copy()
	else:
		actual_row = name

	cv2.imshow('image', image)

	key = cv2.waitKey(0) & 0xFF
	if key == ord('f'):
		pass


			#~/Dokumente/FSU/Anwendungspraktikum/cvtennis/diff_frames/image_00500.png
