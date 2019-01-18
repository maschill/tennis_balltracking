# Import python packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
import pandas as pd

# Import own modules
from src.modules.reconstruct.calibration import getCalibration
from src.modules.reconstruct.get3Dposition import transform_params, get_trajectory, get_time
from src.data.preprocessing.smoothballposition import ReadTFOutput
from src.modules.hitandbounce.models.criticalPoints import genFeatures
from src.visualization.visu3dtrajectory import plot3dtrajectory


######################################################################################
# Load data and config file
######################################################################################
# Set parameters
KerberHalep = True

type_height = {'hit': 1, 'bounce':0, 'serve':2}

# Set start time in video for trajectory reconstruction
MINUTE_START = 0
SECOND_START = 665/25
START_TYPE = 'hit'

# Set end time in video for trajectory reconstruction
MINUTE_END = 0
SECOND_END = 686/25
END_TYPE = 'bounce'

# Load example image and calibration manually defined points
if KerberHalep:
    image = plt.imread('data/raw/images/3_image_GP_00306.png')
    PathToCaliPoints = 'data/annotations/calibrate/Kalibrierung_KerbHal_planar.txt'
    ballpostf = 'data/processed/ballpositions.csv'
    FPS = 25
    maxballspeed = 200
else:
    print('Please define config file ... ')
    #image = cv2.imread('images/image_GP_00001.png')
    #PathToCaliPoints = 'data/annotations/calibrate/Kalibrierung_planar.txt'


######################################################################################
# Camera calibration
######################################################################################
ret, M, D, R, T, imgpoints, objpoints, newM, roi = getCalibration(PathToCaliPoints=PathToCaliPoints,
                                                                  image_width=image.shape[1],
                                                                  image_height=image.shape[0])
print('RMS re-projection error:', ret)

R,T,F = transform_params(R=R,T=T)
print('Camera matrix: \n', M, '\n')
print('Rotation matrix: \n', R, '\n')
print('Translation matrix: \n', T, '\n')
print('Distortion coefficients: \n' ,D,'\n')

######################################################################################
# Court detection
######################################################################################
# This part was done by my project partner and is not included here



######################################################################################
# Detect ball and player
######################################################################################
# Load 2D ball positions (detected using tensorflow object detection api)
ballpos = ReadTFOutput(ballpostf, image)
# Load 2D smoothed ball position (generated using smoothballposition.py)
with open('data/processed/pickle_smoothed_ball_position.pkl', 'rb') as file:
    smballpos = pickle.load(file)



######################################################################################
# Find bounce and hit points
######################################################################################
#load joblib model (random forest model)
clf = joblib.load('src/modules/hitandbounce/models/hitandbouncedetectionmodel.joblib')
# Number of points used for regression
nump = 3
# Because of smoothing edges disappear. Jump can be used to leave points out
jump = 1

bouncehitfeatures = genFeatures(smpos=smballpos, nump=nump, jump=jump)
bouncehitfeatures = pd.DataFrame(bouncehitfeatures)
bouncehitfeatures[[1,2,3,4,5,6,7]] = bouncehitfeatures[[1,2,3,4,5,6,7]].apply(pd.to_numeric)
bouncehitfeatures = bouncehitfeatures.set_index(0)
bouncehitfeatures.columns = ['angle', 'sc1', 'sc2', 'r1_x', 'r1_y', 'r2_x', 'r2_y']
bouncehitfeatures['sum_sc'] = bouncehitfeatures['sc1']+bouncehitfeatures['sc2']
features = ['angle','sc1','sc2','r1_x','r1_y','r2_x','r2_y','sum_sc']
cpreg = clf.predict(bouncehitfeatures[features])
bouncehitfeatures['prediction'] = cpreg



######################################################################################
# 3D reconstruction of 2D image points
######################################################################################
# start and endpoint in original images
START_FRAME_NUMBER = int(FPS*60*MINUTE_START+FPS*SECOND_START)
END_FRAME_NUMBER = int(FPS*60*MINUTE_END+FPS*SECOND_END)
start_3dreconstruction = START_FRAME_NUMBER #665
end_3dreconstruction = END_FRAME_NUMBER #686
a = np.argmin(abs(smballpos[:, 0] - ballpos[start_3dreconstruction][0]))
b = np.argmin(abs(smballpos[:, 0] - ballpos[end_3dreconstruction][0]))
trajectory2D = smballpos[a:b]

# plot trajectory in 2d image
plt.imshow(image)
plt.scatter(trajectory2D[:,1], trajectory2D[:,2], c='y')
fig = plt.gcf()
fig.savefig('temp-img.png')
plt.show()
print('2D image points: \n ', trajectory2D[:,1:3])

# compute 3d trajectory
trajectory3D = np.around(get_trajectory(trajectory2D,
                                        R=R, M=M, T=T, F=F,
                                        Z_start = type_height[START_TYPE],
                                        Z_end = type_height[END_TYPE]), 2)
print('3D coordinates: \n ', trajectory3D)

# Compute flight time of video snippet
time3D = get_time(trajectory2D)

# Compute average ball speed in video snippet
INCH_FACTOR = 39.370
traj_data = trajectory3D*INCH_FACTOR
print(traj_data)

# compute distance in 3D
distance = 0
for i in range(1,len(traj_data)):
    distance += np.linalg.norm(traj_data[i-1]-traj_data[i])

speed = ((distance/INCH_FACTOR)/time3D)*3.6
print('Ball speed ~ %.2f km/h' % speed)



######################################################################################
# 3D visualization
######################################################################################
# check your browser for visualization
plot3dtrajectory(traj_data=traj_data)