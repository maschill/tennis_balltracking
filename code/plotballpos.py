import os
import cv2

file = os.path.join(os.getcwd(), '../ballpositions.csv')

# Load data into array
print('load into array...')
data = []
with open(file) as f:
	for i, row in enumerate(f):
		data = data + [row]

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 800)

for i in range(9200, 12000):
	print(i, row)
	row = data[i]
	row = row.replace('    ', ' ')
	row = row.replace('   ', ' ')
	row = row.replace('  ', ' ')
	path, bb, ac, lb, mx = row.split(';')
	image = cv2.imread(path)
	b1, b2, b3, b4 = [x.strip('[]') for x in bb.split(' ')[0:4]]
	b1, b3 = float(b1) * image.shape[0], float(b3) * image.shape[0]
	b2, b4 = float(b2) * image.shape[1], float(b4) * image.shape[1]
	if float(ac.split(' ')[0].strip('[]')) > 0.3:
		cv2.rectangle(image, (int(b2), int(b1)), (int(b4), int(b3)), (0, 0, 255), 1)
	cv2.imshow('image', image)
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break