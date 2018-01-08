import cv2

# Differenzenbilder unterschiedlich auf PC zu Hause und PC Uni

path = '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames/'

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 800)

file = open('/home/mueller/code/python/Anwendungspraktikum/cvtennis/annotations/GoProDiffAnnotation_train.txt', 'r')
current_row = file.readline()
current_name, label, x1, y1, x2, y2 = current_row.split(' ')
img_number = int(current_name[-9:-4])
current_name = 'image_GP_{:05}.png'.format(img_number)
image = cv2.imread(path + current_name)
cv2.imshow('image', image)


with open('/home/mueller/code/python/Anwendungspraktikum/cvtennis/annotations/GoProDiffAnnotation_train.txt', 'w') as f:
	for row in file:
	#for i in range(1, 1000):
	#	row = file.readline()
		name, label, x1, y1, x2, y2 = row.split(' ')
		img_number = int(name[-9:-4])
		name = 'image_GP_{:05}.png'.format(img_number)
		f.write('{} {} {} {} {} {}'.format(name, label, x1, y1, x2, y2))

		if current_row == name:
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image, label, (x1,y1), font, 1, (0,0,255), 1, cv2.LINE_AA)
			image = image.copy()
			current_row = name

		if name != current_row:
			cv2.imshow('image', image)
			image = cv2.imread(path + name)
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image, label, (x1,y1), font, 1, (0,0,255), 1, cv2.LINE_AA)
			image = image.copy()
			current_row = name
			if cv2.waitKey(200) & 0xFF == ord('q'):
				break