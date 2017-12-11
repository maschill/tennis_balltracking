# README #

This README would normally document whatever steps are necessary to get your application up and running.

#ToDo before training
*	Annotated Frames located at ../Videos/VideonameFrames/
	frames Extract: ffmpeg -i Tennis_Best_Points_2017.mkv frames/image_TBP17_%05d.png
*	Create tf.record files in tensorflow/data; check out tensorflow/README_Lea_tf.txt
*	faster_rcnn_resnet101_coco_2017_11_08 model used for training tensorflow/models/ can be downloaded from 
	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
	Rename model.ckpt.data-00000-of-00001 to model.ckpt
*	Change paths in tensorflow/model.config file
*	go to models/research/object_detection and run train.py

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
