from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from sklearn import linear_model
import numpy as np
import math


def regression(points):
    reg = linear_model.LinearRegression()
    reg.fit(points[:,0].reshape(-1,1), points[:,1].reshape(-1,1))
    return reg.coef_, reg.intercept_, reg.score(points[:,0].reshape(-1,1), points[:,1].reshape(-1,1))

#calulate angular between two lines
# r1, r2 direction vectors of line
def angular(r1, r2):
    try:
        alphaRad = math.acos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
        alpha = math.degrees(alphaRad)
        return alpha
    #In case lines are (almost) parallel
    except ValueError:
        return 0

# Number of points which are used for regression
# Because of smoothing edges disappear. Jump can be used to leave points out
def genFeatures(smpos, nump=3, jump=1):

	# Output array
	bouncehitfeatures = []

	# Generate features for each point in smoothed points
	for i in range(jump+nump-1, len(smpos)-nump-jump-1,1):

	    # Calculate regression through nump number of points
	    slo1, int1, sc1 = regression(smpos[i-jump-nump+1:i-jump+1, 1:3])
	    slo2, int2, sc2 = regression(smpos[i+jump:i+jump+nump, 1:3])

	    # Check orientation, since slope must be negativ if x and y < 0
	    x1, x2 = 1, 1
	    val1 = smpos[i-jump,1:3] - smpos[i-jump-nump+1,1:3]
	    if val1[0] < 0 and val1[1] < 0:
	        slo1[0, 0] *= -1
	        x1 = -1
	    if val1[0] < 0 and val1[1] > 0:
	        slo1[0, 0] *= -1
	        x1= -1
	    #print(i, i+jump+nump-1)
	    val2 = smpos[i+jump+nump-1,1:3] - smpos[i+jump,1:3]
	    if val2[0] < 0 and val2[1] < 0:
	        slo2[0, 0] *= -1
	        x2 = -1
	    if val2[0] < 0 and val2[1] > 0:
	        slo2[0, 0] *= -1
	        x2 = -1

	    #calculate angle if except lines are parallel
	    alpha = angular([x1,slo1[0,0]], [x2,slo2[0,0]])

	    # Calculate sum of direction vectors between each two points
	    r1 = sum(smpos[i-jump-nump+1:i-jump+1, 1:3] - smpos[i+jump, 1:3])
	    r2 = sum(smpos[i+jump:i+jump+nump, 1:3] - smpos[i+jump, 1:3])

	    # Create full imagename from tensorflow digits
	    imgname = '3_image_GP_'+str(int(smpos[i,0])).zfill(5)+ '.png'
	    bouncehitfeatures += [imgname, alpha, sc1, sc2, r1[0], r1[1], r2[0], r2[1]]

	bouncehitfeatures = np.array(bouncehitfeatures).reshape(-1, 8)
	
	return bouncehitfeatures