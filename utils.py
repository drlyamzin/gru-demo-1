
# coding: utf-8


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg'


def ani_frame_traintest(tensor,name='testvideo',sactime_true = None, sactime_test = None):
    # make animation of saccade, mark the time of an actual and predicted saccade
    
    # saccade is predicted if sactime_test>0.2
    
    # input tensor should be squeezed to a matrix in time (i.e. one example) e.g. np.squeeze(tensor[m,:,:,:])
    # dimensions of input tensor are [nH,nW,nT]
    
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
    writer = matplotlib.animation.FFMpegWriter(fps=20)
    
    filename = name+'.mp4'
    ani.save(filename,writer=writer)
    plt.show()
    return ani

def datajitter_wgcamp(tensor,saccades,gcamp):
    
    # jitters a trial in time and space by +-3 in all three coordinates,
    # expands the time of saccade to -1 frame before and +5 frames after - may help fitting and visualizing the saccade
    # in animation
    
    [m,H,W,T] = tensor.shape
    [m,nTiles,T] = gcamp.shape
    tensor_out = np.zeros((m,32,32,T-6))
    saccade_out = np.zeros((m,T-6))
    gcamp_out = np.zeros((m,nTiles,T-6))
    print(tensor_out.shape)
    print(saccade_out.shape)
    print(gcamp_out.shape)
    
    for im in range(m):
        # jitter in time (+crop)
        this_t_jt = np.random.randint(-3,3)
        
        this_tensor = tensor[im,:,:,:]
        this_tensor = np.roll(this_tensor,this_t_jt,axis=2)
        this_tensor = this_tensor[:,:,3:-3]
    
        this_saccade = saccades[im,:]
        # expand the moments of saccades >
        these_saccade_times = np.where(this_saccade)[0]
        for t in these_saccade_times:
            t0 = max(0,t-1)
            tend = min(T,t+5)
            this_saccade[t0:tend]=1
        # <
        
        this_saccade = np.roll(this_saccade,this_t_jt)
        new_saccade = this_saccade[3:-3]
        
        this_gcamp = gcamp[im,:,:]
        this_gcamp = np.roll(this_gcamp,this_t_jt,axis=1)
        this_gcamp = this_gcamp[:,3:-3]
    
        # jitter in space (+crop)
        this_x_jt = np.random.randint(-3,3)
        this_y_jt = np.random.randint(-3,3)
    
        tensor_out[im,:,:,:] = this_tensor[3+this_x_jt:this_x_jt+35,3+this_y_jt:this_y_jt+35,:]
        saccade_out[im,:] = new_saccade
        gcamp_out[im,:,:] = this_gcamp
    
    return tensor_out, saccade_out, gcamp_out