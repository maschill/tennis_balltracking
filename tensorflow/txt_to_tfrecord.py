import tensorflow as tf
import sys
import os
import io
from PIL import Image

from object_detection.utils import dataset_util

# Get help:
# https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
# 

#???
# Better use: https://docs.python.org/3.5/library/argparse.html
flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to input csv file from current directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
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

	xmins = [ xmin / width] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [ xmax / width] # List of normalized right x coordinates in bounding box
	# (1 per box)
	ymins = [ ymin / height] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [ ymax / height] # List of normalized bottom y coordinates in bounding box
	# (1 per box)
	classes_text = [str(label).encode('utf8')] # List of string class name of bounding box (1 per box)
	
	items = {'Ball': 1, 'Spieler': 2}
	classes = [int(items[str(label)])] # List of integer class id of bounding box (1 per box)

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

	with open(path) as imagelabels:
		for row in imagelabels:
			try:
				imgname, label, xmin, ymin, xmax, ymax = row.split(' ')
				print('Start writing {} to file ..'.format(imgname))
				tf_example = create_tf_example(FLAGS.image_folder, imgname, label, float(xmin), float(ymin), float(xmax), float(ymax))
				writer.write(tf_example.SerializeToString())
			except ValueError:
				print('No ball in image {} found'.format(row.split()[0]), sys.exc_info()[0])
				pass
	print('Done writing to file')
	writer.close()


if __name__ == '__main__':
	tf.app.run()
