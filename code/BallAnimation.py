import matplotlib.pyplot as plt
import numpy as np
import time
import os
import math
from sklearn import linear_model
from matplotlib import animation

file = os.path.join(os.getcwd(), '../ballpositions.csv')

# Load data into array
print('load into array...')
data = []
with open(file, 'r') as f:
    for i, row in enumerate(f):
        data = data + [row]
        
path = '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames/3_image_GP_00002.png'
image = plt.imread(path)
print(image.shape)

ballpos = np.array([])
for i in range(4757, len(data)):
    row = data[i]
    row = row.replace('    ', ' ')
    row = row.replace('   ', ' ')
    row = row.replace('  ', ' ')
    path, bb, ac, lb, mx = row.split(';')
    b1, b2, b3, b4 = [x.strip('[]') for x in bb.split(' ')[0:4]]
    b1, b3 = float(b1) * image.shape[0], float(b3) * image.shape[0]
    b2, b4 = float(b2) * image.shape[1], float(b4) * image.shape[1]
    accuracy = float(ac.split(' ')[0].strip('[]'))
    ballpos = np.append(ballpos, [i, (b2+b4)/2, (b1+b3)/2, accuracy])
    
ballpos = ballpos.reshape(-1,4)
ballpos[0:10]

#calculate line through two points
def l(p1,p2):
    return p2-p1

def reg(points):
    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(points[:,0].reshape(-1,1), points[:,1].reshape(-1,1))
    return reg.coef_, reg.intercept_ 

#calulate angular between two lines
def angular(r1, r2): # r1, r2 direction vectors of line
    try:
        alphaRad = math.acos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
        alpha = math.degrees(alphaRad)
        return alpha
    #In case lines are (almost) parallel
    except ValueError:
        return 0

backup = ballpos
ballpos[np.where(ballpos[:,1] > 0.03)]


cpreg = np.zeros(len(ballpos))
ang = np.zeros(len(ballpos))
#plt.figure()
nump = 2
for i in range(0,len(ballpos-nump)-nump,1):
    #plt.plot(ballpos[i:i+nump, 1], ballpos[i:i+nump, 2], 'bo', alpha=0.5)
    #plt.plot(ballpos[i+1:i+nump+1, 1], ballpos[i+1:i+nump+1, 2], 'ro', alpha=0.5)
    slo1, int1 = reg(ballpos[i:i+nump, 1:3])
    slo2, int2 = reg(ballpos[i+1:i+nump+1, 1:3])
    x1, x2 = 1, 1
    #Check orientation, since slope must be negativ if x and y < 0
    val1 = ballpos[i+nump-1,1:3] - ballpos[i,1:3]
    if val1[0] < 0 and val1[1] < 0:
        slo1[0, 0] *= -1
        x1 = -1
    if val[0] < 0 and val[1] > 0:
        slo1[0, 0] *= -1
        x1= -1

    val2 = ballpos[i+nump,1:3] - ballpos[i+1,1:3]
    if val2[0] < 0 and val2[1] < 0:
        slo2[0, 0] *= -1
        x2 = -1
    if val2[0] < 0 and val2[1] > 0:
        slo2[0, 0] *= -1
        x2 = -1

    '''
    x1 = np.linspace(np.min(ballpos[i:i+nump, 1]), np.max(ballpos[i:i+nump, 1]), 500)
    y1 = [val*slo1[0,0]+int1 for val in x1]
    plt.plot(x1, y1, 'b--')

    x = np.linspace(np.min(ballpos[i+1:i+nump+1, 1]), np.max(ballpos[i+1:i+nump+1, 1]), 500)
    y = [val*slo2[0,0]+int2 for val in x]
    plt.plot(x, y, 'r--')

    plt.show()
    '''
    #calculate angle if except lines are parallel
    # 2-bounce, 1-hit
    alpha = angular([x1,slo1[0,0]], [x2,slo2[0,0]])
    ang[i+1] = alpha
    #Distinguish between bounce and hit
    if alpha > 50:
        r1 = sum(ballpos[i+1-3:i+1, 2] - ballpos[i+1, 2])
        r2 = sum(ballpos[i+1:i+1+3, 2] - ballpos[i+1, 2])
        if np.sign(r1) != np.sign(r2):
            cpreg[i+1] = 2
        else:
            cpreg[i+1] = 1


fig = plt.figure()
imgpath = '/home/mueller/code/python/Anwendungspraktikum/Videos/GoPro/GoProFrames/3_image_GP_00306.png'
img = plt.imread(imgpath)
plt.imshow(img)
plt.show()

%matplotlib notebook

fig,ax = plt.subplots(1,1)
plt.imshow(img)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1920)
ax.set_ylim(0,1080)
scatter, = ax.plot([], [], 'bo')
cmap = ['b', 'r', 'g']
scatter = []
for c in cmap:
    scatter.append(ax.plot([], [], 'o', color=c))

def init():
    for scat in scatter:
        scat[0].set_data([],[])
    return scatter

def pltpos(i):
    print(i)
    i += 306
    np = 20
    vals = [k+i for k,x in enumerate(ballpos[i:i+np, 3]) if x > 0.03]
    blue = [k for k in vals if cpreg[k] == 0]
    red = [k for k in vals if cpreg[k] == 1]
    green = [k for k in vals if cpreg[k] == 2]
    x = ballpos[blue,1]
    y = ballpos[blue,2]
    x1 = ballpos[red, 1]
    y1 = ballpos[red, 2]
    x2 = ballpos[green, 1]
    y2 = ballpos[green, 2]
    xlist = [x, x1, x2]
    ylist = [y, y1, y2]
    for lnum, scat in enumerate(scatter):
        scat[0].set_data(xlist[lnum], ylist[lnum])
    return scatter

ani = animation.FuncAnimation(fig, pltpos, init_func=init, frames=20000, interval=200, blit=True)
plt.gca().invert_yaxis()
plt.show()