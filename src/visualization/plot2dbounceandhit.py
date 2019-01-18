# runs only if integrated in jupyter notebook
def plot2dtrajectory(smballpos, ballpos, cpreg, jump, nump, starting_frame_number):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from matplotlib import animation


    imgpath = os.path.join(os.getcwd(), 'images/3_image_GP_00306.png')
    backgroundimage = plt.imread(imgpath)

    fig, ax = plt.subplots(1, 1)
    plt.imshow(backgroundimage)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    scatter, = ax.plot([], [], 'b.')
    cmap = ['y', 'r', 'g', 'b']

    scatter = [ax.imshow(backgroundimage)]
    for c in cmap:
        scatter.append(ax.plot([], [], '.', color=c))


    def init():
        for scat in scatter[1:]:
            print(scat[0])
            scat[0].set_data([], [])
        scatter[0].set_data(backgroundimage)
        return scatter


    def pltpos(i):
        i += starting_frame_number
        npoi = 20
        # If frames were extracted they can be plotted, otherwise example image is used as background image
        try:
            imgpath = os.path.join(os.getcwd(), '../../Videos/GoPro/GoProFrames/3_image_GP_' +
                                   str(int(ballpos[i, 0])).zfill(5) + '.png')
            img = plt.imread(imgpath)
        except FileNotFoundError:
            img = backgroundimage

        # Check if detection is in smoothed positions
        j = np.where(smballpos[:, 0] == int(ballpos[i, 0]))
        if len(j[0]) > 0:
            vals = [k + int(j[0]) for k, x in enumerate(smballpos[int(j[0]):int(j[0]) + npoi, 3])]
            ball = [k for k in vals if cpreg[k - jump - nump] == 'Nothing']
            hit = [k for k in vals if cpreg[k - jump - nump] == 'Hit']
            bounce = [k for k in vals if cpreg[k - jump - nump] == 'Bounce']
            pink = [k for k in vals if cpreg[k - jump - nump] == 3]
            x = smballpos[ball, 1]
            y = smballpos[ball, 2]
            x1 = smballpos[hit, 1]
            y1 = smballpos[hit, 2]
            x2 = smballpos[bounce, 1]
            y2 = smballpos[bounce, 2]
            x3 = smballpos[pink, 1]
            y3 = smballpos[pink, 2]
            xlist = [x, x1, x2, x3]
            ylist = [y, y1, y2, y3]

            for lnum, scat in enumerate(scatter[1:]):
                scat[0].set_data(xlist[lnum], ylist[lnum])

        # If not, plot previous points
        #else:
        #    for lnum, scat in enumerate(scatter[1:]):
        #        scat[0].set_data(xlist[lnum], ylist[lnum])

        scatter[0].set_data(img)

        return scatter


    ani = animation.FuncAnimation(fig, pltpos, init_func=init, frames=20000, interval=200, blit=True)
    plt.gca().invert_yaxis()
    plt.show()