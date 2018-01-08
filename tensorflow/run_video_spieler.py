import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import glob

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

#'/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames'
# Enter path to folder with frames of Video
PATH_TO_TEST_IMAGES_DIR = sys.argv[1]

# What model to download.
MODEL_DIR = '/home/mueller/code/python/Anwendungspraktikum/cvtennis/tensorflow/models/'
MODEL_NAME = 'GoProSpielerGraph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_DIR + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'spieler_label_map.pbtxt')
print(PATH_TO_CKPT, PATH_TO_LABELS)

NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Get Video
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image_GP_{:05}.png'.format(i)) for i in range(0, 3400) ]
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.png'))

# Size, in inches, of the output images.
IMAGE_SIZE = (8, 6)

#cv2.startWindowThread()
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 800)
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
		plt.figure(figsize=IMAGE_SIZE)
		for i in range(1100, 1200):
			print(i)
			image_path1 = TEST_IMAGE_PATHS[i] #1-2
			image_path2 = TEST_IMAGE_PATHS[i-1]
			image1 = cv2.imread(image_path1)
			image2 = cv2.imread(image_path2)
			image = Image.open(image_path1)
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			#image_np1 = load_image_into_numpy_array(image1)
			#image_np2 = load_image_into_numpy_array(image2)
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
			
			cv2.imshow('frame', image_np)
			if cv2.waitKey(24) & 0xFF==ord('q'):
				break


			#plt.imshow(image_np)
			#plt.show()