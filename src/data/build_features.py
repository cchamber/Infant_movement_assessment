import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../src/modules')
import circstats as CS
import scipy as sc

def angle_features(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    # - absolute angle
    a_mean = np.degrees(CS.nanmean(np.array(np.radians(df['angle']))))
    # - variability of angle
    a_stdev = np.sqrt(np.degrees(CS.nanvar(np.array(np.radians(df['angle'])))))
    # - measure of complexity (entropy)
    a_ent = ent(df['angle'].round())
    # - median absolute velocity
    median_vel = (np.abs(df['velocity'])).median()
    # - variability of velocity
    IQR_vel = (df['velocity']).quantile(.75) - (df['velocity']).quantile(.25)
    # - variability of acceleration
    IQR_acc = df['acceleration'].quantile(.75) - df['acceleration'].quantile(.25)

    return pd.DataFrame.from_dict({'video':np.unique(df.video),'bp':np.unique(df.bp),\
    'mean_angle':a_mean, 'stdev_angle':a_stdev, 'entropy_angle':a_ent,
    'median_vel_angle':median_vel,'IQR_vel_angle':IQR_vel,\
    'IQR_acc_angle': IQR_acc})

def xy_features(df):
    # - absolute position/angle
    median_dist = np.sqrt(df['x']**2+df['y']**2).median()
    # - variability of position/angle
    IQR_dist = np.sqrt(df['x']**2+df['y']**2).quantile(.75)-np.sqrt(df['x']**2+df['y']**2).quantile(.25)
    # - median speed
    median_speed = df['speed'].median()
    # - variability of speed
    IQR_speed = df['speed'].quantile(.75)-df['speed'].quantile(.25) 
    # - median absolute velocity
    median_vel = (np.abs(df['velocity_x']).median() +np.abs(df['velocity_y']).median())/2
    # - variability of velocity
    IQR_vel = ((df['velocity_x'].quantile(.75)-df['velocity_x'].quantile(.25)) +\
    (df['velocity_y'].quantile(.75)-df['velocity_y'].quantile(.25)))/2
    # - variability of acceleration
    IQR_acc = (df['acceleration_x'].quantile(.75) - df['acceleration_x'].quantile(.25))\
    +(df['acceleration_y'].quantile(.75) - df['acceleration_y'].quantile(.25))/2
    # - measure of complexity (entropy)
    ent_x = ent(df['x'].round(2))
    ent_y = ent(df['y'].round(2))
    mean_ent = (ent_x+ent_y)/2
    # define part and side here
    return pd.DataFrame.from_dict({'video':np.unique(df.video),'bp':np.unique(df.bp),\
    'mediandist': median_dist, 'IQRdist': IQR_dist,\
    'medianspeed':median_speed, 'IQRspeed':IQR_speed,\
    'medianvel':median_vel, 'IQRvel':IQR_vel,'IQRacc':IQR_acc,'meanent':mean_ent})

def ent(data):
    p_data= data.value_counts()/len(data) #  probabilities
    entropy=sc.stats.entropy(p_data)
    return entropy

def corr_lr(df, var):
    idf = pd.DataFrame()
    idf['R'] = df[df.side=='R'].reset_index()[var]
    idf['L'] = df[df.side=='L'].reset_index()[var]
    return idf.corr().loc['L','R']


def main(data_set):
    pose_estimates_path = '../data/pose_estimates/'+data_set+'/py'
    feature_path = '../data/interim'

    xdf = pd.read_pickle(os.path.join(pose_estimates_path, 'processed_pose_estimates_coords.pkl'))
    adf = pd.read_pickle(os.path.join(pose_estimates_path, 'processed_pose_estimates_angles.pkl'))
    # angle features
    feature_angle = adf.groupby(['bp','video']).apply(angle_features).reset_index(drop=True)
    feature_angle = pd.pivot_table(feature_angle, index='video', columns=['bp'])
    l0 = feature_angle.columns.get_level_values(1)
    l1 = feature_angle.columns.get_level_values(0)
    cols = [l1[i]+'_'+l0[i] for i in range(len(l1))]
    feature_angle.columns = cols
    feature_angle =feature_angle.reset_index()
    # - measure of symmetry (left-right cross correlation)
    corr_joint = adf.groupby(['video', 'part']).apply(lambda x:corr_lr(x,'angle')).reset_index()
    corr_joint['part'] = 'lrCorr_angle_'+corr_joint['part']
    corr_joint.columns = ['video', 'feature', 'Value']
    corr_joint = pd.pivot_table(corr_joint, index='video', columns=['feature'])
    l1 = corr_joint.columns.get_level_values(1)
    corr_joint.columns = l1
    corr_joint = corr_joint.reset_index()
    feature_angle = pd.merge(feature_angle,corr_joint, on='video', how='outer')
    # xy features
    bps = ['LAnkle', 'RAnkle', 'LWrist', 'RWrist']
    feature_xy = xdf[np.isin(xdf.bp, bps)].groupby(['bp','video']).apply(xy_features)
    feature_xy = pd.pivot_table(feature_xy, index='video', columns=['bp'])
    l0 = feature_xy.columns.get_level_values(1)
    l1 = feature_xy.columns.get_level_values(0)
    cols = [l1[i]+'_'+l0[i] for i in range(len(l1))]
    feature_xy.columns = cols
    feature_xy = feature_xy.reset_index()
    # - measure of symmetry (left-right cross correlation)
    xdf['dist'] = np.sqrt(xdf['x']**2+xdf['y']**2)
    corr_joint = xdf.groupby(['video', 'part']).apply(lambda x:corr_lr(x,'dist')).reset_index()
    corr_joint['part'] = 'lrCorr_x_'+corr_joint['part']
    corr_joint.columns = ['video', 'feature', 'Value']
    corr_joint = pd.pivot_table(corr_joint, index='video', columns=['feature'])
    l1 = corr_joint.columns.get_level_values(1)
    corr_joint.columns = l1
    corr_joint = corr_joint.reset_index()
    feature_xy = pd.merge(feature_xy, corr_joint, on='video', how='outer')

    features = pd.merge(feature_xy, feature_angle, on='video', how='outer')
    features.to_pickle(os.path.join(feature_path, 'features_'+data_set+'.pkl'))
    
if __name__== '__main__':
    main()