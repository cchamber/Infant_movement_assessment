import glob
import pandas as pd
import os
import json
from util import get_prediction, get_image_meta_data_from_filenames, get_ground_truth, get_keypoints
import itertools
import numpy as np
from itertools import chain
import cv2
import matplotlib.pyplot as plt


def get_groundtruth_and_predictions(save_images, image_path, gt_path, keras_weights_files, model_names):
    
    image_df = get_image_meta_data_from_filenames(image_path)
    # get ground truth: keypoints and bounding box 
    gt_df = get_ground_truth(gt_path)

    kp_gt_df = gt_df.groupby('id').apply(get_keypoints).reset_index()[['id', 'part_label', 'x_gt', 'y_gt']]
    kp_gt_df['person_idx'] = kp_gt_df['id'].astype(str).str[-2:].astype(int)
    kp_gt_df['id'] = kp_gt_df['id'].astype(str).str[:-2].astype(int)
    kp_gt_df.loc[kp_gt_df.x_gt==0, 'x_gt'] = np.nan
    kp_gt_df.loc[kp_gt_df.y_gt==0, 'y_gt'] = np.nan

    bbox_df = gt_df.bbox.apply(pd.Series); bbox_df.columns = ['left', 'bottom', 'width', 'height']
    bbox_df['id'] = gt_df['id'].astype(str).str[:-2].astype(int)
    bbox_df['bbox_diag'] = np.sqrt(bbox_df.width**2+bbox_df.height**2)
    kp_gt_df = pd.merge(kp_gt_df, bbox_df[['id', 'bbox_diag']], on='id', how='left')
    kp_gt_df.loc[kp_gt_df.x_gt==0, 'x_gt'] = np.nan
    kp_gt_df.loc[kp_gt_df.y_gt==0, 'y_gt'] = np.nan

    # get prediction
    pred_df = pd.DataFrame()
    for i, (iweight, imodel) in enumerate(zip(keras_weights_files, model_names)):
        zdf = get_prediction(iweight, imodel, save_images, image_path)
        if i ==0:
            pred_df = zdf.reset_index()
        elif i>0:
            pred_df = pd.merge(pred_df, zdf.reset_index(), on=['id', 'part_label'], how='outer')

    columns = pd.MultiIndex.from_product([['ground_truth'],['x','y']], names=['var_type','dim'])
    index = pd.MultiIndex.from_arrays([np.array(kp_gt_df.id), np.array(kp_gt_df.part_label)], names=['id','part_label'])
    gt_df_merge = pd.DataFrame(kp_gt_df[['x_gt', 'y_gt']].as_matrix(),index=index, columns=columns).reset_index()

    gt_pred_df = pd.merge(gt_df_merge, pred_df, on=['id','part_label'], how = 'outer')

    bbox = bbox_df[['id','bbox_diag']]
    columns = pd.MultiIndex.from_arrays([['id','ground_truth'],['','bbox_diag']], names=['var_type','dim'])
    bbox.columns = columns
    gt_pred_df = pd.merge(gt_pred_df, bbox, on='id', how='outer')
    gt_pred_df['frame', ''] = gt_pred_df['id', ''].astype(str).str[7:]
    gt_pred_df['video', ''] = gt_pred_df['id', ''].astype(str).str[1:7]

    # compute distance
    for i in model_names:
        gt_pred_df[i, 'distance_pix'] = np.sqrt((gt_pred_df[i].x - gt_pred_df.ground_truth.x)**2 + (gt_pred_df[i].y-gt_pred_df.ground_truth.y)**2) 
        gt_pred_df[i, 'distance_norm'] = gt_pred_df[i, 'distance_pix']/gt_pred_df['ground_truth','bbox_diag']
    return gt_pred_df


def main():
    save_images = 1
    image_path = '/data2/clairec/infant_NN_training_dataset/val_all_100818_1inf_step1'
    gt_path = '/data2/clairec/infant_NN_training_dataset/person_keypoints_val_all_100818_1inf_step1.json'
    keras_weights_files = ['../models/cmu_model.h5', '../models/trained_model_oct.h5']
    model_names = ['original_model', 'trained_model']
    image_prediction_path = '../data/pose_model/images'
    for i in model_names:
        if os.path.isdir(os.path.join(image_prediction_path, i))==False:
            os.mkdir(os.path.join(image_prediction_path, i))
            
    gt_pred_df = get_groundtruth_and_predictions(save_images, image_path, gt_path, keras_weights_files, model_names)

    gt_pred_df.to_pickle('../data/pose_model/model_predictions_and_groundtruth.pkl')

if __name__ == '__main__':
    main()
