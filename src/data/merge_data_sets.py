import pandas as pd
import numpy as np
import os

def main():
    feature_path = '../data/interim'
    meta_data_path = '../data/video_meta_data'
    save_path = '../data/processed'

    age_threshold = 10

    # features
    yt = pd.read_pickle(os.path.join(feature_path, 'features_youtube.pkl'))
    clin = pd.read_pickle(os.path.join(feature_path, 'features_clinical.pkl'))

    # meta data youtube
    # id: unique id per infant
    # rated age in weeks
    meta_data_yt = pd.read_pickle(os.path.join(meta_data_path, 'meta_data_yt.pkl'))
    yt = pd.merge(yt,meta_data_yt, on='video', how='inner')

    # meta data clinical
    # id: unique id per infant
    # age (corrected and chronological), risk
    meta_data_clin = pd.read_pickle(os.path.join(meta_data_path,'meta_data_clin.pkl'))
    
    info_series = pd.Series([i.replace('-', '_').split('_') for i in clin.video])
    info_df = info_series.apply(pd.Series)
    info_df.columns = ['dum', 'infant', 'session', 'trial', 'GP', 'edited']
    clin[['infant', 'session', 'trial']] = info_df[['infant', 'session', 'trial']]

    # add meta-data columns:
    # category: label 0 -yt, 1 -clin
    # risk: 0 -yt, >0 -clin
    # exclude infants with missing BINS score: 28,29,32,33

    # merge meta-data and features dataframe
    clin['category'] = 1
    yt['category'] = 0
    yt['risk'] = 0
    yt['infant'] = 'yt_'+yt['infant_id'].astype(int).astype(str)
    meta_data_clin['risk'] = meta_data_clin['Risk_low0_mod1_high2_corr'] # corrected risk for preterm infants
    meta_data_clin.loc[meta_data_clin.risk.isnull(),'risk'] = meta_data_clin.loc[meta_data_clin.risk.isnull(), 'Risk_low0_mod1_high2_chron'] # chronological risk for term infants
    meta_data_clin['risk'] = meta_data_clin['risk']+1
    meta_data_clin['chron_age'] = meta_data_clin['Months_chron']*4 + meta_data_clin['Days_chron']/7 # chronological age for term infants
    meta_data_clin['age_in_weeks'] = meta_data_clin['Months_corr']*4 + meta_data_clin['Days_corr']/7 # corrected age for preterm infants
    meta_data_clin.loc[meta_data_clin.age_in_weeks.isnull(),'age_in_weeks'] = meta_data_clin.loc[meta_data_clin.age_in_weeks.isnull(), 'chron_age']
    meta_data_clin = meta_data_clin[['infant','session' ,'risk', 'age_in_weeks']]
    clin['infant'] = clin.infant.astype(int)
    clin['session'] = clin.session.astype(int)
    clin = pd.merge(clin, meta_data_clin, on=['infant', 'session'], how='inner')
    clin['infant'] = 'clin_'+clin['infant'].astype(str)+'_'+clin['age_in_weeks'].astype(int).astype(str)

    clin = clin.drop(['session', 'trial'], axis=1)
    yt = yt.drop('infant_id', axis=1)

    features = clin.append(yt)
    features = features.set_index(['video','category', 'infant', 'age_in_weeks', 'risk']).reset_index()
    # average across rows for same infant
    features = features.groupby('infant').mean().reset_index()

    id_vars = ['infant', 'category','age_in_weeks', 'risk']
    # pivot dataframe
    features = pd.melt(features, id_vars=id_vars, var_name="feature", value_name="Value")
    # add age bracket
    features['age_bracket'] = (features.age_in_weeks>age_threshold)+0
    # average across left and right sides
    feature_str = features.feature.str.split('_')
    side =pd.Series([ i[-1:][0][0] if i[0]!='lrCorr' else '' for i in feature_str])
    part =pd.Series([ i[-1:][0][1:] if i[0]!='lrCorr' else i[-1:][0] for i in feature_str])
    feature = pd.Series(['_'.join(i[:-1]) for i in feature_str])
    feature_attributes = pd.DataFrame.from_dict({'side': side, 'part': part, 'feature_name': feature})
    features[['feature_name', 'part', 'side']] = feature_attributes
    features = features.groupby(['infant', 'part','feature_name']).mean().reset_index()
    features.to_pickle(os.path.join(save_path, 'features_merged.pkl'))
    
if __name__== '__main__':
    main()