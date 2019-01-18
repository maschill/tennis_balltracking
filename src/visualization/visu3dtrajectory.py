def plot3dtrajectory(traj_data):
    import numpy as np
    import cv2
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, plot

    init_notebook_mode(connected=True)

    #set field dimensions and constants
    INCH_FACTOR = 39.370

    x0 = 25
    y0 = 25

    thicknes = 3

    total_len = 936
    total_width = 432

    far_left = (x0,y0)
    far_left_single = (x0+54,y0) # 4 foot 6 inches
    far_right_single = (x0+378,y0)
    far_right = (x0+total_width,y0) # 36 feet

    near_left = (x0, y0+total_len)
    near_left_single = (x0+54,y0+total_len) # 4 foot 6 inches
    near_right_single = (x0+378,y0+total_len)
    near_right = (x0+total_width,y0+total_len) # 36 feet

    far_left_service = (x0+54, y0+216)
    far_left_outer_service = (x0, y0+216)
    far_right_service = (x0+378, y0+216)
    far_center_service = (x0+216, y0+216)
    far_right_outer_service = (x0+total_width, y0+216)

    near_left_service = (x0+54, y0+720)
    near_left_outer_service =  (x0, y0+720 )
    near_right_service = (x0+378, y0+720)
    near_center_service = (x0+216, y0+720)
    near_right_outer_service = (x0+total_width, y0+720)
    near_center = (x0+216, y0+total_len)


    center_left = (x0-36,y0+936//2)
    center_right = (x0+total_width+36, y0+936//2)

    left_net_post = (x0-36,936//2+y0,42)
    right_net_post = (x0+total_width+36,y0+936//2,42)

    points_new = np.array([[x[0],x[1],1] for x in [far_left,far_left_single,far_right_single,far_right,
    near_left,near_left_single,near_right_single,near_right,
    far_left_service,far_right_service,far_center_service,
    near_left_service,near_right_service,near_center_service, center_left, center_right]])

    background = np.zeros((total_len+x0*2,total_width+72,3), dtype='uint8')+255
    warp_lines = [
        [points_new[0], points_new[4]], #left sideline
        [points_new[3], points_new[7]], # right sideline
        [points_new[0], points_new[3]], #far baseline
        [points_new[4], points_new[7]], #near baseline
        [points_new[1], points_new[5]], #right single sideline
        [points_new[2], points_new[6]], #left single sideline
        [points_new[8], points_new[9]], #far service line
        [points_new[11], points_new[12]], #near service line
        [points_new[10], points_new[13]],  #center service line
        [points_new[14], points_new[15]]
        ]

    for line in warp_lines:
        cv2.line(background, tuple(line[0][:2]), tuple(line[1][:2]), (0,0,0), thicknes)
    court=background.swapaxes(0,1)

    court_rgba = np.dstack([court/255, (np.isin(court[:,:,1],0)+1)*.5])

    #traj_data = np.genfromtxt("../../results/trajectory3Dexample.txt")*INCH_FACTOR
    #duration = 0.8
    #distance = 0
    #for i in range(1,len(traj_data)):
    #    distance += np.linalg.norm(traj_data[i-1]-traj_data[i])

    #speed = ((distance/INCH_FACTOR)/duration)*3.6
    #print('%.2f km/h' % speed)

    fitX = traj_data[:,0]
    fitY = traj_data[:,1]
    fitZ = traj_data[:,2]

    # plotly visu
    pts = [go.Scatter3d(x=[1], y=[1], z=[1])]
    for line in warp_lines:
        pts.append(go.Scatter3d(x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], z=[0, 0], marker=dict(
            size=3,
            color='red',
            colorscale='#1f77b4',
        ),
                                line=dict(
                                    color='#1f77b4',
                                    width=10
                                ),
                                showlegend=False,
                                hoverinfo='none'
                                )
                   )

    for post in (left_net_post, right_net_post):
        pts.append(go.Scatter3d(x=[post[0], post[0]], y=[post[1], post[1]], z=[0, post[2]],
                                line=dict(
                                    color='purple',
                                    width=10
                                ),
                                marker=dict(
                                    size=3,
                                    color='purple',
                                    colorscale='Viridis',
                                ),
                                showlegend=False
                                )
                   )

    pts.append(go.Scatter3d(x=[left_net_post[0], right_net_post[0]],
                            y=[left_net_post[1], right_net_post[1]],
                            z=[left_net_post[2], right_net_post[2]],
                            line=dict(
                                color='purple',
                                width=10
                            ),
                            marker=dict(
                                size=3,
                                color='purple',
                                colorscale='Viridis',
                            ),
                            showlegend=False
                            )
               )

    pts.append(go.Mesh3d(
        x=[x0, x0 + total_width, x0, x0 + total_width],
        y=[y0, y0, y0 + total_len, y0 + total_len],
        z=[0, 0, 0, 0],
        showlegend=False,
        showscale=False,
        opacity=0.5,
        color='lightblue'  # [x0, x0+total_width, x0, x0+total_width]
    ))

    pts.append(go.Mesh3d(
        x=[left_net_post[0], right_net_post[0], left_net_post[0], right_net_post[0]],
        y=[left_net_post[1], right_net_post[1], left_net_post[1] + 1, right_net_post[1] + 1],
        z=[0, 0, left_net_post[2], right_net_post[2]],
        #     showlegend=False,
        #     showscale=False,
        opacity=0.5,
        color='lightblue'
    ))

    pts.append(go.Mesh3d(
        x=[fitX[0], fitX[0], fitX[-1], fitX[-1]],
        y=[fitY[0], fitY[0] + 1, fitY[-1], fitY[-1] + 1],
        z=[0, 100, 0, 100],
        color='teal',
        opacity=.5
    )
    )

    visible = [True for x in pts]
    not_visible = [True for x in pts]
    not_visible[-1] = False

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                range=[-50, total_width + 100],
                showaxeslabels=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                range=[0, total_len + 50],
                showgrid=False,
                #         zeroline=False,
                showline=False,
                showaxeslabels=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                range=[-.1, 500],
                showgrid=False,
                #         zeroline=False,
                showline=False,
                showaxeslabels=False,
                showticklabels=False,
                title=''
            )),
        updatemenus=[{'type': 'buttons',
                      'buttons': [
                          {
                              'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                              'fromcurrent': True,
                                              'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                              'label': 'Play',
                              'method': 'animate'
                          },
                          {
                              'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                                'transition': {'duration': 0}}],
                              'label': 'Pause',
                              'method': 'animate'
                          },
                          {
                              'args': [{'visible': visible}],
                              'label': 'Show Plane',
                              'method': 'update'
                          },
                          {
                              'args': [{'visible': not_visible}],
                              'label': 'Hide Plane',
                              'method': 'update'
                          }

                      ],
                      }]
    )

    frames = [dict(data=[dict(x=fitX[:k],
                              y=fitY[:k],
                              z=fitZ[:k],
                              type='scatter3d',
                              marker=dict(
                                  color='gold',
                                  size=5
                              ),
                              line={'width': 3, 'color': 'gold'})
                         ]) for k in range(len(fitX))]

    fig = go.Figure(data=pts, layout=layout, frames=frames)
    plot(fig)

if __name__ == "__main__":
    import numpy as np
    INCH_FACTOR = 39.370
    traj_data = np.genfromtxt("../../results/trajectory3Dexample.txt") * INCH_FACTOR

    plot3dtrajectory(traj_data=traj_data)