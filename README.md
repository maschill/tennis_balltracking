# README #

3D reconstruction of tennis ball trajectory and tennis court detection to generate game statistics from tennis videos. To see the results of the detection checkout our video in **results/BounceAndHitDetection.mp4** (green Bounce, red Hit)

* The workflow for reconstruction is described in **code/DataFlow.ipynb**. Steps are as follows
	* Detect tennis balls using tensorflow object detection api
	* Detect court lines to estimate calibration parameters
	* Calibrate camera based on tennis court edges
	* A random forest detects hit and bounce points in 2D video
	* Reconstruct 3D trajectory between hit (assume hit takes place at 1 meter height z=1) an bounce (z=0) point
	* Calculate for example speed of serve based on trajectory and time

### How do I get set up? ###

* To run jupyter notebook Install python > 3.5 with jupyter (we recommend using anaconda) and packages (tested version) 
	* cv2 (3.2.0)
	* sklearn (0.19.1)
	* pandas (0.22.0)
	* numpy (1.13.3)
	* matplotlib (2.2.2)
* To see 3D animation in notebook install
	* holoviews (1.8.3)
	* plotly (2.2.3)

### Who do I talk to? ###

* For questions about reconstruction and ball detection contact mueller.lea@uni-jena.de and for court detection matthias.schilling@uni-jena.de
