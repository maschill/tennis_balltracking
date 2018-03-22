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
# You can either provide a folder with images for detection
# For testsets using either a folder containing test images (TEST_IMAGE_DIR) only or
# a combination of folder (TEST_IMAGE_DIR) with images and a file (TEST_IMAGE_FILE)
# with names of testimages can be passed

# Example usage:
# Test image direcroy:
# python MakePrediction.py --TEST_IMAGE_DIR='/YOUR/PATH/TO/cvtennis/images/ObjectDetectionExamples/'

# Test image file + directory containing all frames:
# python MakePrediction.py --TEST_IMAGE_DIR='/YOUR/PATH/TO/Videos/GoPro/GoProFrames' 
# --TEST_IMAGE_FILE='/YOUR/PATH/TO/cvtennis/annotations/GoProBall_train.txt' 

# Any directory:
# python MakePrediction.py --PATH_TO_IMAGES_DIR='/YOUR/PATH/TO/cvtennis/images/ObjectDetectionExamples/' 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Ball Detection Prediction')
	parser.add_argument('--PATH_TO_IMAGES_DIR', metavar='path to folder with images',
		     help='folder containing images on which prediction will be made, e.g. YOUR/PATH/../../Videos/GoPro/GoProFrames')
	parser.add_argument('--TEST_IMAGE_FILE', metavar = 'txt file with ground truth relativ from current directory',
		     help='if set all images in file will be used for detection task')
	parser.add_argument('--TEST_IMAGE_DIR', metavar = 'Folder with test images from current directory', 
		     help='e.g. ../../Videos/GoPro/GoProFrames can be used if GoPro Testfile is used as well')
	parser.add_argument('--output_file', metavar='directory for output', 
		     help='csv file in which output will be saved. If not provided output will not be saved')
	parser.add_argument('--Player', metavar = 'detect player in image', nargs=1,
		     help='if set model for player detection is used. Otherwise ball will be detected.')
	args = parser.parse_args()

# What model to download.
MODEL_DIR = os.path.join(os.getcwd(), 'models/')

# The player model was trained with GoPro images only.
if args.Player:
	print('Player detection')
	MODEL_NAME = 'GoProSpielerGraph'
	LABEL_MAP = 'spieler_label_map.pbtxt'
	NUM_CLASSES = 2
	maxboxnum = 2
	if args.output_file:
	  writefile = open(args.output_file, 'a')
else:
	print('Ball detection')
	MODEL_NAME = 'GoProBall1742Graph'
	LABEL_MAP = 'ball_label_map.pbtxt'
	NUM_CLASSES = 1
	maxboxnum = 1
	if args.output_file:
	  writefile = open(args.output_file, 'a')

# Enter path to folder with frames of Video
if args.TEST_IMAGE_FILE and args.TEST_IMAGE_DIR:
	TEST_IMAGE_FILE = os.path.join(os.getcwd(), args.TEST_IMAGE_FILE)
	IMAGE_PATHS = []
	with open(TEST_IMAGE_FILE) as f:
		for row in f:
			imgname = row.split(' ')[0]
			IMAGE_PATHS += [os.path.join(args.TEST_IMAGE_DIR, imgname)]
elif args.PATH_TO_IMAGES_DIR:
	IMAGE_PATHS = sorted(glob.glob(os.path.join(args.PATH_TO_IMAGES_DIR, '*.png')))
elif args.TEST_IMAGE_DIR:
	IMAGE_PATHS = sorted(glob.glob(os.path.join(args.TEST_IMAGE_DIR, '*.png')))
else:
	sys.exit('Missing Input! Either PATH_TO_IMAGES_DIR or (TEST_IMAGE_DIR and TEST_IMAGE_FILE) with images necessary (use python MakePrediction.py -h for help)')


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

if not args.output_file:
  print('Show video')
  plt.figure(figsize=(12,8))
  
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
		for i in range(0,len(IMAGE_PATHS),1):
			image_path = IMAGE_PATHS[i]
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
			# Visualization of the results of a detection if no output file is passed.
			if not args.output_file:
				vis_util.visualize_boxes_and_labels_on_image_array(
					  image_np,
					  np.squeeze(boxes),
					  np.squeeze(classes).astype(np.int32),
					  np.squeeze(scores),
					  category_index,
					  use_normalized_coordinates=True,
					  line_thickness=8,
					  min_score_thresh=.1,
					  max_boxes_to_draw=maxboxnum)
				plt.imshow(image_np)
				plt.pause(.001)
				plt.draw()
			print('{}; {}; {}; {}; {}\n'.format(i, image_path, boxes[0][0], scores[0][0], classes[0][0]))

			if args.PATH_TO_IMAGES_DIR and args.output_file:
				writestr = '{};[{} {} {} {} {}];{};{};{}\n'.format(image_path, boxes[0][0],boxes[0][1], boxes[0][2], 
					boxes[0][3], boxes[0][4], scores[0][0:5], classes[0][0:5], num)
				writefile.write(writestr)
			# If test images are used we only save the most likely bounding box for a ball detection
			elif args.TEST_IMAGE_FILE and args.output_file:
				writestr = '{},{},{},{},{}\n'.format(image_path, boxes[0][0], scores[0][0], classes[0][0], num)
				writefile.write(writestr)
				
			if cv2.waitKey(24) & 0xFF==ord('q'):
				break
if args.output_file:
  writefile.close()
