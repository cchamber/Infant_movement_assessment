import sys
sys.path.insert(0,'../modules')
import pandas as pd
import os

from util_data import get_fps, read_video, get_video_information_yt, get_video_information_clinical,load_raw_pkl_files, get_skel, edit_df, interpolate_df, smooth, comp_joint_angle, comp_center_joints, normalise_skeletons, get_joint_angles, angular_disp, get_angle_displacement, smooth_dyn, get_delta, get_dynamics_xy, get_dynamics_angle


def main(data_set, raw_pose_estimates_video_path):

    pose_estimates_path = '../data/pose_estimates/'+data_set+'/py'
    pose_estimate_animation_path = '../data/pose_estimates/'+data_set+'/video'

    # get video information
    if os.path.exists(os.path.join(pose_estimates_path, 'video_info.pkl'))==0:
        if data_set == 'youtube':
            df_fps = get_video_information_yt(raw_pose_estimates_video_path)
        elif data_set == 'clinical':
            df_fps = get_video_information_clinical(raw_pose_estimates_video_path)
        else:
            df_fps = get_video_information_yt(raw_pose_estimates_video_path)
        df_fps.to_pickle(os.path.join(pose_estimates_path, 'video_info.pkl'))
    else:
        df_fps = pd.read_pickle(os.path.join(pose_estimates_path, 'video_info.pkl')) 

    # load raw pkl files
    if os.path.exists(os.path.join(pose_estimates_path, 'raw_pose_estimates.pkl'))==0:
        df_pkl = load_raw_pkl_files(raw_pose_estimates_video_path)
        if data_set == 'youtube':
            df_pkl['video'] = df_pkl.video.str[:12]
            
        df_pkl.to_pickle(os.path.join(pose_estimates_path, 'raw_pose_estimates.pkl'))
    else:
        df_pkl = pd.read_pickle(os.path.join(pose_estimates_path, 'raw_pose_estimates.pkl'))

    # extract skeleton from raw pose estimates
    if os.path.exists(os.path.join(pose_estimates_path, 'pose_estimates.pkl'))==0:
        df = df_pkl.groupby(['video', 'frame']).apply(get_skel)
        df = edit_df(df, df_fps)
        df.to_pickle(os.path.join(pose_estimates_path, 'pose_estimates.pkl'))
        
if __name__== '__main__':
    main()