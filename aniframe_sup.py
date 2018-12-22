# coding: utf-8


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg'

def ani_frame_traintest(tensor,name='testvideo',sactime_true = None, sactime_test = None):
    # make animation of saccade, mark the time of an actual and predicted saccade
    
    # saccade is predicted if sactime_test>0.2
    
    # can provide a logical vector of saccade times: the time of saccade will be marked with a black square in the top-left corner
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # first frame
    im = ax.imshow(np.squeeze(tensor[:,:,0]),cmap='gray',interpolation='none')
    im.set_clim([-2,3])
    fig.set_size_inches([5,5])

    plt.tight_layout()

    # update a frame
    def update_img(n):
        tmp = np.squeeze(tensor[:,:,n])
        if sactime_true[n]:
            tmp[0,0],tmp[1,0],tmp[0,1],tmp[1,1] = -5,-5,-5,-5
        if sactime_test[n]>0.2:
            tmp[0,2],tmp[0,3],tmp[1,2],tmp[1,3] = 5,5,5,5
        im.set_data(tmp)
        return im

    nFrames = tensor.shape[2]
    ani = animation.FuncAnimation(fig,update_img,nFrames)
    writer = matplotlib.animation.FFMpegWriter(fps=8)
    
    filename = name+'.mp4'
    ani.save(filename,writer=writer)
    plt.show()
    return ani