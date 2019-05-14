import pandas as pd
import glob
import numpy as np
from itertools import chain
import os
import cv2
from moviepy.editor import VideoFileClip
import itertools
import skvideo.io

def get_fps(videoname):
    clip = VideoFileClip(videoname)
    return clip.fps

def read_video(video):
    videogen = skvideo.io.vreader(video)
    new_videogen = itertools.islice(videogen, 0, 1, 1)
    for image in new_videogen:
        a = 1
    return image

def get_video_information_yt(file_path):
    videofiles = np.array(glob.glob(os.path.join(file_path,'video*')))
    videofiles = videofiles[np.array([len(os.path.basename(i)) if i[-3:]!='pkl' else 0 for i in videofiles])==len('video_000000.mp4')]
    # get fps and screen dim
    df_fps = pd.DataFrame()
    fpsl = []
    rowlist = []; collist = []
    for ivideo in videofiles:
        print(ivideo)
        if os.path.basename(ivideo)[-3:]=='avi':
            fps = 30
        else:
            fps = get_fps(ivideo)
        fpsl.append(fps)
        img = read_video(ivideo)
        nrows = len(img)
        ncols = len(img[0])
        rowlist.append(nrows); collist.append(ncols);
    df_fps['fps'] = pd.Series(fpsl)
    df_fps['pixel_x'] = pd.Series(collist)
    df_fps['pixel_y'] = pd.Series(rowlist)
    df_fps['video'] = [os.path.basename(i)[:-4] for i in videofiles]
    return df_fps

def get_video_information_clinical(file_path):
    videofiles = np.array(glob.glob(os.path.join(file_path,'822487*')))
    videofiles = [i for i in videofiles if i[-3:]!='pkl']
    videofiles = [i for i in videofiles if i[-len('openposeLabeled.mp4'):]!='openposeLabeled.mp4']
    # get fps and screen dim
    df_fps = pd.DataFrame()
    fpsl = []
    rowlist = []; collist = []
    for ivideo in videofiles:
        if os.path.basename(ivideo)[:6]=='822487':
            fps = 30
        else:
            fps = get_fps(ivideo)
        fpsl.append(fps)
        img = read_video(ivideo)
        nrows = len(img)
        ncols = len(img[0])
        rowlist.append(nrows); collist.append(ncols);
    df_fps['fps'] = pd.Series(fpsl)
    df_fps['pixel_x'] = pd.Series(collist)
    df_fps['pixel_y'] = pd.Series(rowlist)
    df_fps['video'] = [os.path.basename(i)[:-4] for i in videofiles]
    return df_fps

def load_raw_pkl_files(path):
    pklfiles = np.array(glob.glob(os.path.join(path,'*.pkl')))
    df_pkl = pd.DataFrame()
    for file in pklfiles:
        one_file = pd.read_pickle(file).reset_index().drop('index',axis = 1)
        df_pkl = df_pkl.append(one_file)
    df_pkl = df_pkl.reset_index().drop('index', axis = 1)
    return df_pkl

def get_skel(df):
    if len(list(itertools.chain(*df.limbs_subset)))>0:
        peaks = df.peaks.iloc[0]
        parts_in_skel = df.limbs_subset.iloc[0]
        person_to_peak_mapping = [list(i[:-2]) for i in parts_in_skel] 
        skel_idx = [[i]*(len(iskel)-2) for i, iskel in enumerate(parts_in_skel)]
        idx_df = pd.DataFrame.from_dict({'peak_idx':list(itertools.chain(*person_to_peak_mapping)),\
         'person_idx':list(itertools.chain(*skel_idx))})
        peaks_list = list(chain.from_iterable(peaks))
        x = [ipeak[0] for ipeak in peaks_list]
        y = [ipeak[1] for ipeak in peaks_list]
        c = [ipeak[2] for ipeak in peaks_list]
        peak_idx = [ipeak[3] for ipeak in peaks_list]
        kp_idx = list(chain.from_iterable([len(ipeak)*[i] for i,ipeak in enumerate(peaks)]))
        peak_df = pd.DataFrame.from_dict({'x':x,'y':y,'c':c,'peak_idx':peak_idx,'part_idx':kp_idx})
        kp_df = pd.merge(idx_df, peak_df, on='peak_idx', how='left').drop('peak_idx',axis=1)
        kp_df = kp_df.loc[~kp_df.c.isnull(),:]
    else:
        kp_df = pd.DataFrame()
    return kp_df

def edit_df(df, df_fps):
    # keep person index with max number of keypoints per frame
    counts = df.groupby(['video','frame', 'person_idx'])['c'].count().reset_index()
    max_rows = counts.groupby(['video','frame'])['c'].idxmax().tolist()
    max_rows_df = counts.loc[max_rows,['video','frame', 'person_idx']]
    max_rows_df['dum'] = 1
    df = pd.merge(df.reset_index(), max_rows_df, on=['video','frame', 'person_idx'], how='inner')

    # add keypoint labels
    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    df['bp'] = [bps[int(i)] for i in df.part_idx]
    df = df[['video','frame', 'x', 'y', 'bp', 'part_idx']] 

    # include row for each keypoint and frame
    max_frame = df.groupby('video').frame.max().reset_index()
    max_frame['frame_vec'] = max_frame.frame.apply(lambda x: np.arange(0,x+1))
    max_frame['bp'] = pd.Series([bps]*len(max_frame))
    y =[]
    _ = max_frame.apply(lambda x: [y.append([x['video'], x['bp'], i]) for i in x.frame_vec], axis=1)
    all_frames = pd.DataFrame(y, columns = ['video','bp','frame'])
    z =[]
    _ = all_frames.apply(lambda x: [z.append([x['video'], x['frame'], i]) for i in x.bp], axis=1)
    all_frames = pd.DataFrame(z, columns = ['video','frame', 'bp'])
    df = pd.merge(df, all_frames, on = ['video','frame','bp'], how='outer')
    df = pd.merge(df,df_fps, on = 'video', how='outer')
    df['time'] = df['frame']/df['fps']
    
    part_idx_df = df[['bp', 'part_idx']].drop_duplicates().dropna().sort_values('part_idx')
    df = pd.merge(df.drop('part_idx', axis=1), part_idx_df, on= 'bp', how='inner')
    
    return df

def interpolate_df(df):
    df = df.sort_values('frame')
    df['x']=df.x.interpolate()
    df['y']=df.y.interpolate()
    return df

def smooth(d, var, winmed, winmean):
    winmed1 = winmed
    winmed = int(winmed*d.fps.unique()[0])
    winmean = int(winmean*d.fps.unique()[0])
    d = d.reset_index(drop=True)
    if winmed>0:
        x = d.sort_values('frame')[var].rolling(center=True,window=winmed).median()
        d[var] = x.rolling(center=True,window=winmean).mean()
    else:
        d[var] = d[var]
    d['smooth'] = winmed1
    return d

def comp_joint_angle(df, joint_str):
    df = df.loc[(df.bp=='L'+joint_str)|(df.bp=='R'+joint_str)]
    df = pd.pivot_table(df, columns = ['bp'], values=['x', 'y'], index=['frame'])
    # zangle =np.arctan2(Rj.y.iloc[0]-Lj.y.iloc[0],Rj.x.iloc[0]-Lj.x.iloc[0])
    df[joint_str+'_angle']= np.arctan2((df['y', 'R'+joint_str]-df['y', 'L'+joint_str]),(df['x', 'R'+joint_str]-df['x', 'L'+joint_str]))
    df = df.drop(['x', 'y'], axis=1)
    return df

def comp_center_joints(df, joint_str, jstr):
    df = df.loc[(df.bp=='L'+joint_str)|(df.bp=='R'+joint_str)]
    df = pd.pivot_table(df, columns = ['bp'], values=['x', 'y'], index=['frame'])
    # zangle =np.arctan2(Rj.y.iloc[0]-Lj.y.iloc[0],Rj.x.iloc[0]-Lj.x.iloc[0])
    df[jstr+'y']= (df['y', 'R'+joint_str]+df['y', 'L'+joint_str])/2
    df[jstr+'x']= (df['x', 'R'+joint_str]+df['x', 'L'+joint_str])/2
    df = df.drop(['x', 'y'], axis=1)
    return df

def normalise_skeletons(df):
    
    ''' Rotate keypoints around reference points (center of shoulders, center of hips)\
    Normalise points by reference distance (trunk length)'''

    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    ubps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist", "REye","LEye","REar","LEar"]
    lbps = ["RKnee","RAnkle","LHip","LKnee","LAnkle","RHip"]
    u_idx = np.where(np.isin(bps, ubps)==1)[0]
    l_idx = np.where(np.isin(bps, lbps)==1)[0]
    df['upper'] = np.isin(df.bp, ubps)*1
    # compute shoulder and hip angles for rotating, upper and lower body 
    # reference parts, now center of shoulders and hips
    
    s_angle = df.groupby(['video']).apply(lambda x: comp_joint_angle(x,'Shoulder')).reset_index()
    h_angle = df.groupby(['video']).apply(lambda x: comp_joint_angle(x,'Hip')).reset_index()
    uref = df.groupby(['video']).apply(lambda x: comp_center_joints(x, 'Shoulder', 'uref')).reset_index()
    lref = df.groupby(['video']).apply(lambda x: comp_center_joints(x, 'Hip', 'lref')).reset_index()
    s_angle['Hip_angle'] = h_angle['Hip_angle']
    s_angle = pd.merge(s_angle, uref, on=['video', 'frame'], how='inner')
    s_angle = pd.merge(s_angle, lref, on=['video', 'frame'], how='inner')
    s_angle.columns = s_angle.columns.get_level_values(0)

    df = pd.merge(df,s_angle, on=['video', 'frame'], how = 'outer')
    # set up columns, reference parts and reference angles

    df['refx'] = df['urefx']*df['upper'] + df['lrefx']*(1-df['upper'])
    df['refy'] = df['urefy']*df['upper'] + df['lrefy']*(1-df['upper'])
    df['ref_dist'] = np.sqrt((df['urefy']-df['lrefy'])**2+(df['urefx']-df['lrefx'])**2)
    df['ref_angle'] = df['Shoulder_angle']*df['upper'] + df['Hip_angle']*(1-df['upper'])

    df.loc[df.ref_angle<0,'ref_angle'] = 2*np.pi + df.loc[df.ref_angle<0,'ref_angle'] 
    df.loc[df.ref_angle<np.pi,'ref_angle'] = np.pi - df.loc[df.ref_angle<np.pi,'ref_angle'] 
    df.loc[(df.ref_angle>np.pi)&(df.ref_angle<2*np.pi),'ref_angle'] = 3*np.pi - df.loc[(df.ref_angle>np.pi)&(df.ref_angle<2*np.pi),'ref_angle']
    df['x_rotate'] = df['refx'] + np.cos(df['ref_angle'])*(df['x']-df['refx']) - np.sin(df['ref_angle'])*(df['y'] - df['refy'])
    df['y_rotate'] = df['refy'] + np.sin(df['ref_angle'])*(df['x']-df['refx']) + np.cos(df['ref_angle'])*(df['y'] - df['refy'])
    df['x_rotate'] = (df['x_rotate']-df['refx'])/df['ref_dist']
    df['y_rotate'] = (df['y_rotate']-df['refy'])/df['ref_dist']
    df['x'] = df['x_rotate']
    df['y'] = df['y_rotate']
    # add to lower body to make trunk length 1
    df.loc[df.upper==0,'y'] = df.loc[df.upper==0,'y']+1
    df['delta_t'] = 1/df['fps']
    
    return df

def get_joint_angles(df):
    
    ''' Compute joint angles from x,y coordinates '''

    df = df[~df.x.isnull()]
    df['delta_t'] = 1/df['fps']

    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    joints = [['part0', 'part1', 'part2'],\
    ['RShoulder','Neck', 'RElbow'],\
    ['RElbow', 'RShoulder','RWrist'],\
    ['RHip','LHip', 'RKnee'],\
    ['RKnee', 'RHip','RAnkle'],\
    ['LShoulder','Neck', 'LElbow'],\
    ['LElbow', 'LShoulder','LWrist'],\
    ['LHip','RHip', 'LKnee'],\
    ['LKnee', 'LHip','LAnkle']]
    headers = joints.pop(0)
    df_joints = pd.DataFrame(joints, columns=headers).reset_index()
    df_joints['bpindex'] = df_joints['index']+1

    df_1 = df
    for i in [0,1,2]:
        df_joints['bp'] = df_joints['part'+str(i)]
        df_1 = pd.merge(df_1,df_joints[['bp', 'bpindex']], on='bp', how='left')
        df_1['idx'+str(i)] = df_1['bpindex'] 
        df_1 = df_1.drop('bpindex', axis=1)
        df_1['x'+str(i)] = df_1['x']*df_1['idx'+str(i)]/df_1['idx'+str(i)]
        df_1['y'+str(i)] = df_1['y']*df_1['idx'+str(i)]/df_1['idx'+str(i)]
    df0 = df_1[['video', 'frame', 'idx0', 'x0', 'y0', 'bp']]; df0 = df0.rename(index=str, columns={"bp": "bp0", "idx0": "idx"}); df0 = df0[~df0.idx.isnull()]
    df1 = df_1[['video', 'frame', 'idx1', 'x1', 'y1', 'bp']]; df1 = df1.rename(index=str, columns={"bp": "bp1", "idx1": "idx"}); df1 = df1[~df1.idx.isnull()]
    df2 = df_1[['video', 'frame', 'idx2', 'x2', 'y2', 'bp']]; df2 = df2.rename(index=str, columns={"bp": "bp2", "idx2": "idx"}); df2 = df2[~df2.idx.isnull()]
    df_2 = pd.merge(df0,df1, on=['video', 'frame', 'idx'], how='inner')
    df_2 = pd.merge(df_2,df2, on=['video', 'frame', 'idx'], how='inner')

    # compute angle here
    df_2['dot'] = (df_2['x1'] - df_2['x0'])*(df_2['x2'] - df_2['x0']) + (df_2['y1'] - df_2['y0'])*(df_2['y2'] - df_2['y0'])
    df_2['det'] = (df_2['x1'] - df_2['x0'])*(df_2['y2'] - df_2['y0']) - (df_2['y1'] - df_2['y0'])*(df_2['x2'] - df_2['x0'])
    df_2['angle_degs'] = np.arctan2(df_2['det'],df_2['dot'])*180/np.pi
    # hip and shoulder should be same regardless of side
    # elbow/knee give flexion/extension information only
    df_2['side'] = df_2.bp0.str[:1]
    df_2['part'] = df_2.bp0.str[1:]
    df_2['angle'] = df_2.angle_degs # same on left/right
    df_2.loc[(df_2['bp0']=='LShoulder')|(df_2['bp0']=='LHip'),'angle'] = \
    df_2.loc[(df_2['bp0']=='LShoulder')|(df_2['bp0']=='LHip'),'angle']*(-1)
    df_2.loc[(df_2['part']=='Elbow')|(df_2['part']=='Knee'),'angle'] = \
    np.abs(df_2.loc[(df_2['part']=='Elbow')|(df_2['part']=='Knee'),'angle'])
    # shoulders/hips: change to -180-+180 to 0-360 if neg: 360+angle
    df_2.loc[((df_2['part']=='Shoulder')|(df_2['part']=='Hip'))& (df_2.angle<0),'angle'] = \
    df_2.loc[((df_2['part']=='Shoulder')|(df_2['part']=='Hip'))& (df_2.angle<0),'angle']+360
    # can include shoulder rotation
    df_2['bp'] = df_2['bp0']

    df_info = df.groupby(['video', 'frame', 'fps','time', 'delta_t']).mean().reset_index()[['video', 'frame', 'fps','time', 'delta_t']]
    df_angle = pd.merge(df_2[['video', 'frame', 'bp', 'side', 'part', 'angle']],\
    df_info, on=['video', 'frame'], how='inner').drop_duplicates()
    return df_angle


def angular_disp(x,y): 
    possible_angles = np.asarray([y-x, y-x+360, y-x-360])
    idxMinAbsAngle = np.abs([y-x, y-x+360, y-x-360]).argmin(axis=0)
    smallest_angle = np.asarray([possible_angles[idxMinAbsAngle[i],i] for i in range(len(possible_angles[0]))])
    return smallest_angle

def get_angle_displacement(df, inp, outp): # different from other deltas - need shortest path
    df = df.sort_values('frame')
    angle = np.array(df[inp])
    a = angular_disp(angle[0:len(angle)-1], angle[1:len(angle)])
    df[outp] = np.concatenate((np.asarray([0]),a))
    return df

def smooth_dyn(df, inp, outp, win):
    fps = df['fps'].unique()[0]
    win = int(win*fps)
    x = df[inp].interpolate()
    df[outp] = x.rolling(window=win,center=False).mean()
    return df

def get_delta(df, inp, outp):
    x = df[inp]
    df[outp]  = np.concatenate((np.asarray([0]),np.diff(x)))*(np.asarray(x*0)+1)
    return df

def get_dynamics_xy(xdf, delta_window):
    # get velocity, acceleration
    xdf = xdf[['video','frame','x','y','bp','fps','pixel_x', 'pixel_y','time','delta_t', 'part_idx']]
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'x','d_x')).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'y','d_y')).reset_index(drop=True)
    xdf['displacement'] = np.sqrt(xdf['d_x']**2 + xdf['d_y']**2)
    xdf['velocity_x_raw'] = xdf['d_x']/xdf['delta_t']
    xdf['velocity_y_raw'] = xdf['d_y']/xdf['delta_t']
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'velocity_x_raw','velocity_x', delta_window))
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'velocity_y_raw','velocity_y', delta_window))
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'velocity_x','delta_velocity_x')).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'velocity_y','delta_velocity_y')).reset_index(drop=True)
    xdf['acceleration_x_raw'] = xdf['delta_velocity_x']/xdf['delta_t']
    xdf['acceleration_y_raw'] = xdf['delta_velocity_y']/xdf['delta_t']
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'acceleration_x_raw','acceleration_x', delta_window))
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'acceleration_y_raw','acceleration_y', delta_window))
    xdf['acceleration_x2'] = xdf['acceleration_x']**2
    xdf['acceleration_y2'] = xdf['acceleration_y']**2
    xdf['speed_raw'] = xdf['displacement']/xdf['delta_t']
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'speed_raw','speed', delta_window))
    xdf['part'] = xdf.bp.str[1:]
    xdf['side'] = xdf.bp.str[:1]
    return xdf

def get_dynamics_angle(adf, delta_window):
    adf = adf.groupby(['bp','video']).apply(lambda x: get_angle_displacement(x,'angle','displacement')).reset_index(drop=True)
    adf['velocity_raw'] = adf['displacement']/adf['delta_t']
    adf = adf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'velocity_raw','velocity', delta_window))
    adf = adf.groupby(['bp','video']).apply(lambda x: get_delta(x,'velocity','delta_velocity')).reset_index(drop=True)
    adf['acceleration_raw'] = adf['delta_velocity']/adf['delta_t']
    adf = adf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'acceleration_raw','acceleration', delta_window))
    adf['acceleration2'] = adf['acceleration']**2
    adf['part'] = adf.bp.str[1:]
    adf['side'] = adf.bp.str[:1]
    return adf
