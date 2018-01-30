import cv2

# Difference images extracted with ffmpeg were not similar
# Dependence on ffmpeg version?

path = '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames/'

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 800)

# Check if next image is 
with open('/home/mueller/code/python/Anwendungspraktikum/cvtennis/annotations/GoProBall_train.txt', 'r') as f:
	for i, row in enumerate(f):
		name, label, x1, y1, x2, y2 = row.split(' ')
		image = cv2.imread(path + name)
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image, label, (x1,y1), font, 1, (0,0,255), 1, cv2.LINE_AA)
		cv2.imshow('image', image)
		if cv2.waitKey(200) & 0xFF == ord('q'):
			break
