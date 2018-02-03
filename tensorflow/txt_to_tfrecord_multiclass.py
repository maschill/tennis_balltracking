import tensorflow as tf
import sys
import os
import io
import csv
from PIL import Image

from object_detection.utils import dataset_util

# Get help:
# https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
# 

#???
# Better use: https://docs.python.org/3.5/library/argparse.html
flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to input csv file from current directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord from current directory')
flags.DEFINE_string('image_folder', '', 'Folder with images')
FLAGS = flags.FLAGS
#???

def create_tf_example(path, imgname, label, xmin, ymin, xmax, ymax):
	# TODO(user): Populate the following variables from your example.
	with tf.gfile.GFile(os.path.join(path, '{}'.format(imgname)), 'rb') as fid:
		encoded_png = fid.read()
	print(os.path.join(path, '{}'.format(imgname)))

	encoded_png_io = io.BytesIO(encoded_png)
	image = Image.open(encoded_png_io)

	width, height = image.width, image.height # Image height
	filename =  imgname.encode('utf8') # Filename of the image. Empty if image is not from file
	image_format = b'png'

	xmins = [ x / width for x in xmin]#[ xmin / width] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [ x / width for x in xmax] # List of normalized right x coordinates in bounding box
	# (1 per box)
	ymins = [ y / height for y in ymin] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [ y / height for y in ymax] # List of normalized bottom y coordinates in bounding box
	# (1 per box)
	classes_text = [str(lab).encode('utf8') for lab in label] # List of string class name of bounding box (1 per box)

	#items = {'Spieler1': 1, 'Spieler2': 2}
	items = {'Ball': 1}
	classes = [int(items[str(x)]) for x in label] # List of integer class id of bounding box (1 per box)

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename),
		'image/source_id': dataset_util.bytes_feature(filename),
		'image/encoded': dataset_util.bytes_feature(encoded_png),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))

	return tf_example



def main(_):
	print(os.path.join(os.getcwd(), FLAGS.output_path))

	writer = tf.python_io.TFRecordWriter(os.path.join(os.getcwd(), FLAGS.output_path))
	# TODO(user): Write code to read in your dataset to examples variable
	path = os.path.join(os.getcwd(), FLAGS.input_path)

	label_, xmin_, ymin_, xmax_, ymax_ = [], [], [], [], []

	# Get first image
	with open(path) as imagelabels:
		readeronce = csv.reader(imagelabels, delimiter=' ')
		row = readeronce.__next__()
		current_image = row[0]
		val = len(list(readeronce))
		print(row, val)
	imagelabels.close()


	with open(path) as imagelabels:
		reader = csv.reader(imagelabels, delimiter=' ')
		#for row in imagelabels:
		for i in range(0, val):
			try:
				print(i, val)
				next_row = reader.__next__()

				if current_image == next_row[0]:
					label_.append(next_row[1])
					xmin_.append(float(next_row[2]))
					ymin_.append(float(next_row[3]))
					xmax_.append(float(next_row[4]))
					ymax_.append(float(next_row[5]))

				else:
					print('Start writing {} to file ..'.format(current_image))
					tf_example = create_tf_example(FLAGS.image_folder, current_image, label_,
												   xmin_, ymin_, xmax_, ymax_)
					writer.write(tf_example.SerializeToString())
					current_image = next_row[0]
					label_, xmin_, ymin_, xmax_, ymax_ = [next_row[1]], [float(next_row[2])], [float(next_row[3])], [float(next_row[4])], [float(next_row[5])]
				if i == val-1:
					tf_example = create_tf_example(FLAGS.image_folder, current_image, label_, xmin_, ymin_, xmax_, ymax_)
					writer.write(tf_example.SerializeToString())
			except ValueError:
				print('No ball in image {} found'.format(row.split()[0]), sys.exc_info()[0])
				pass
	print('Done writing to file')
	writer.close()


if __name__ == '__main__':
	tf.app.run()
