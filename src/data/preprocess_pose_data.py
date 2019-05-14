import sys
sys.path.insert(0,'../modules')
import pandas as pd
import os
from util_data import get_fps, read_video,load_raw_pkl_files, get_skel, edit_df, interpolate_df, smooth, comp_joint_angle, comp_center_joints, normalise_skeletons, get_joint_angles, angular_disp, get_angle_displacement, smooth_dyn, get_delta, get_dynamics_xy, get_dynamics_angle

def main(data_set):
    pose_estimates_path = '../data/pose_estimates/'+data_set+'/py'
    median_window = 1
    mean_window = 1
    delta_window = .25 # smoothing applied to delta_x, velocity, acceleration
    df = pd.read_pickle(os.path.join(pose_estimates_path, 'pose_estimates.pkl'))
    # normalise x and y by image length (conserve aspect ratio)
    df['x'] = pd.to_numeric(df['x'])
    df['y'] = pd.to_numeric(df['y'])
    df['x'] = (df['x'] - df['pixel_x']/2)/df['pixel_y']
    df['y'] = (df['y'] - df['pixel_y']/2)/df['pixel_y']
    # interpolate
    df = df.groupby(['video', 'bp']).apply(interpolate_df).reset_index(drop=True)
    # median and mean filter
    median_window = .5
    mean_window = .5
    df = df.groupby(['video', 'bp']).apply(lambda x: smooth(x, 'y', median_window, mean_window)).reset_index(drop=True)
    df = df.groupby(['video', 'bp']).apply(lambda x: smooth(x, 'x', median_window, mean_window)).reset_index(drop=True)
    # rotate and normalise by reference
    xdf = normalise_skeletons(df)
    # extract angles
    adf = get_joint_angles(df)
    # get dynamics
    xdf = get_dynamics_xy(xdf, delta_window)
    adf = get_dynamics_angle(adf,delta_window)
    # save
    xdf.to_pickle(os.path.join(pose_estimates_path, 'processed_pose_estimates_coords.pkl'))
    adf.to_pickle(os.path.join(pose_estimates_path, 'processed_pose_estimates_angles.pkl'))
    
if __name__== '__main__':
    main()