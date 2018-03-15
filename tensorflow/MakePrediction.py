from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import glob
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, '/home/mueller/progs/models/research/object_detection')
sys.path.append('..')
from utils import label_map_util

from utils import visualization_utils as vis_util

#if tf.__version__ != '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# This is needed to display the images.
#%matplotlib inline

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Ball Detection Prediction')
	parser.add_argument('--PATH_TO_TEST_IMAGES_DIR', metavar='path to folder with images',
	                    help='folder containing images on which prediction will be made')
	parser.add_argument('--TEST_IMAGE_FILE', metavar = 'txt file with ground truth relativ from current directory',
						help='if set all images in file will be used for detection task')
	parser.add_argument('--Player', metavar = 'detect player in image', nargs=1,
	                    help='if set model for player detection is used. Otherwise ball will be detected.')
	args = parser.parse_args()
	print(args)
# What model to download.
MODEL_DIR = '/home/mueller/code/python/Anwendungspraktikum/cvtennis/tensorflow/models/'
if args.Player:
	print('Player detection')
	MODEL_NAME = 'GoProSpielerGraph'
	LABEL_MAP = 'spieler_label_map.pbtxt'
	NUM_CLASSES = 2
	writefile = open('results/PositionsPlayer_.csv', 'a')
else:
	print('Ball detection')
	MODEL_NAME = 'GoProBall1742Graph'
	LABEL_MAP = 'ball_label_map.pbtxt'
	NUM_CLASSES = 1
	writefile = open('results/PositionsBall_.csv', 'a')

# Enter path to folder with frames of Video
print(args.TEST_IMAGE_FILE, args.PATH_TO_TEST_IMAGES_DIR)
if args.TEST_IMAGE_FILE:
	file = args.TEST_IMAGE_FILE
	print(file)
	TEST_IMAGE_FILE = os.path.join(os.getcwd(), file)
	TEST_IMAGE_PATHS = []
	with open(TEST_IMAGE_FILE) as f:
		for row in f:
			imgname = row.split(' ')[0]
			TEST_IMAGE_PATHS += [os.path.join(os.path.join(os.getcwd(), 
				'../../Videos/GoPro/GoProFrames'), imgname)]
elif args.PATH_TO_TEST_IMAGES_DIR:
	PATH_TO_TEST_IMAGES_DIR = args.PATH_TO_TEST_IMAGES_DIR
	TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.png')))
else:
	sys.exit('Missing Input! Either Testfile or folder with images necessary (use python MakePrediction.py -h)')


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_DIR + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', LABEL_MAP)
print(PATH_TO_CKPT, PATH_TO_LABELS)

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, 
	use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (8, 6)

#cv2.startWindowThread()
#cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('frame', 800, 800)
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		# Definite input and output Tensors for detection_graph
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		#plt.figure(figsize=IMAGE_SIZE)
		for i in range(0,len(TEST_IMAGE_PATHS),1):
			image_path = TEST_IMAGE_PATHS[i] #1-2
			print(i, image_path)
			image = Image.open(image_path)
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
			# Visualization of the results of a detection.
			#image_np_color = cv2.imread('/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames/image_GP_{}'.format(TEST_IMAGE_PATHS[i][-9:]))
			vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=8)

			#plt.imshow(image_np)
			#plt.pause(.001)
			#plt.draw()

			if args.PATH_TO_TEST_IMAGES_DIR:
				writestr = '{};[{} {} {} {} {}];{};{};{}\n'.format(image_path, boxes[0][0],boxes[0][1], boxes[0][2], 
					boxes[0][3], boxes[0][4], scores[0][0:5], classes[0][0:5], num)
				writefile.write(writestr)
			elif args.TEST_IMAGE_FILE:
				writestr = '{},{},{},{},{}\n'.format(image_path, boxes[0][0], scores[0][0], classes[0][0], num)
				writefile.write(writestr)				
			#if cv2.waitKey(24) & 0xFF==ord('q'):
		#		break

writefile.close()
			#plt.imshow(image_np)
			#plt.show()