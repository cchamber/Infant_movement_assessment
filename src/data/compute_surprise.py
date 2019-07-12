import pandas as pd
import os
from scipy.stats import norm
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def main(path):

    features = pd.read_pickle(os.path.join(path,'features_merged.pkl'))

    ref_stats = pd.DataFrame()
    ref_stats = features[features.category ==0].groupby(['feature_name','age_bracket', 'part'])['Value']\
    .apply(norm.fit).reset_index()
    ref_stats[['mean_ref', 'sd_ref']] = ref_stats['Value'].apply(pd.Series)
    ref_stats['var_ref'] = ref_stats['sd_ref']**2
    ref_stats = ref_stats.reset_index().drop('Value', axis=1)
    features = pd.merge(features,ref_stats, on=['feature_name','age_bracket', 'part'], how='inner')
    features['minus_log_pfeature'] = -1*(.5*np.log(2*np.pi*features['var_ref']) + ((features['Value']-features['mean_ref'])**2)/(2*features['var_ref']))
    features['feature'] = features.part +'_'+ features.feature_name

    # We will include the following features computed from the positions of the extremities (wrists/ankles) and joint angles (elbows/knees):
    # - absolute position/angle
    # - variability of position/angle
    # - median speed
    # - variability of speed
    # - median absolute velocity
    # - variability of velocity
    # - variability of acceleration
    # - measure of complexity (entropy)
    # - measure of symmetry (left-right cross correlation)
    

    feature_list = ['Ankle_medianx','Wrist_medianx','Ankle_mediany','Wrist_mediany',\
                'Knee_mean_angle','Elbow_mean_angle',\
                'Ankle_IQRx', 'Wrist_IQRx','Ankle_IQRy', 'Wrist_IQRy',\
                'Knee_stdev_angle', 'Elbow_stdev_angle',\
                'Ankle_medianspeed','Wrist_medianspeed',\
                'Ankle_IQRspeed', 'Wrist_IQRspeed',\
                'Ankle_medianvelx','Wrist_medianvelx','Ankle_medianvely','Wrist_medianvely',\
                'Knee_median_vel_angle','Elbow_median_vel_angle',\
                'Ankle_IQRvelx','Wrist_IQRvelx','Ankle_IQRvely','Wrist_IQRvely',\
                'Knee_IQR_vel_angle','Elbow_IQR_vel_angle',\
                'Ankle_IQRaccx','Wrist_IQRaccx','Ankle_IQRaccy','Wrist_IQRaccy',\
                'Knee_IQR_acc_angle','Elbow_IQR_acc_angle',\
                'Ankle_meanent', 'Wrist_meanent','Knee_entropy_angle', 'Elbow_entropy_angle',\
                'Ankle_lrCorr_x', 'Wrist_lrCorr_x','Knee_lrCorr_angle', 'Elbow_lrCorr_angle']
    
    features = features.loc[np.isin(features.feature, feature_list)]

    surprise = features.groupby(['infant', 'age_in_weeks','risk', 'age_bracket', 'category'])['minus_log_pfeature'].sum().reset_index()
    surprise['z'] = (surprise['minus_log_pfeature'] - surprise.loc[surprise.risk==0, 'minus_log_pfeature'].mean())/surprise.loc[surprise.risk==0,'minus_log_pfeature'].std()
    surprise['p'] =(sc.stats.norm.sf(np.abs(surprise['z']))*2).round(3)
    surprise.to_pickle(os.path.join(path, 'bayes_surprise.pkl'))
    features.to_pickle(os.path.join(path, 'final_feature_set.pkl'))

if __name__== '__main__':
    path = '../data/processed/'
    main(path)
    
