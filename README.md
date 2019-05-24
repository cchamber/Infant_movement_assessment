# Infant pose estimation and infant movement-based assessment


<img src="path/to/image“ width="500" align="middle">


## Project Contributors
Claire Chambers clairenc@seas.upenn.edu   
Nidhi Seethapathi snidhi@seas.upenn.edu  
Rachit Saluja rsaluja@seas.upenn.edu  
Michelle Johnson johnmic@pennmedicine.upenn.edu  
Konrad Kording kording@seas.upenn.edu  


## Link to data
https://figshare.com/s/10034c230ad9b2b2a6a4 . 

Directory structure
------------

    ├── README.md  
    ├── data (Download [here] (https://figshare.com/s/10034c230ad9b2b2a6a4))  
    │   ├── example_video  
    │   ├── interim  
    │   ├── pose_estimates  
    │   ├── pose_model  
    │   ├── processed  
    │   ├── video_meta_data  
    │   └── visualization  
    ├──models (Download [here] (https://figshare.com/s/10034c230ad9b2b2a6a4))  
    ├── notebooks  
    ├──src  
    │   ├──  data  
    │   ├──  modules  
    │   ├──  pose_model  
    │   └──  visualization 


--------

## Motivation
Infant movement-based assessments identify infants at risk for neuromotor diseases. Existing tests are based on visual inspection of infant movement by clinicians and are not widely available. Population screening would identify a greater number of at-risk infants. To do population screening, we need automated tests.

## Project
Here we provide methods to automate movement-based assessments for infants using video analysis of infant movement.
We provide:
- A pose estimation model trained to extract infant pose from videos
- A normative database of infant pose and movement
- A pipeline to retrain a neural network using ground-truth data
- A pipeline to extract pose from videos, and statistically compares at-risk infants to a normative database of infant movement

## Data
To create a normative database of healthy infant movement, we collected video data of infants on YouTube (video URLs and  infant pose data [here](https://figshare.com/s/10034c230ad9b2b2a6a4)). To validate our approach, we recorded infants at different levels of neuromotor risk collected in the laboratory (infant pose data [here](https://figshare.com/s/10034c230ad9b2b2a6a4)).

## Pipeline

#### Infant movement based assessment
- Extract pose from videos ([pose_extraction.zip](https://figshare.com/s/10034c230ad9b2b2a6a4))
- Compare movement of at-risk infants with normative database (notebooks/master.ipynb, notebooks/visualize_results.ipynb) using pre-registered set of kinematic features, [here](https://osf.io/hv7tm/)

#### Pose estimation model
Option to examine success in extracting pose from labelled data
- Transfer learning applied to OpenPose pose estimation algorithm from human adults to infants, [code here](https://github.com/cchamber/openpose_keras)
- Measurement of pose model error with respect to ground truth data (notebooks/master.ipynb,
notebooks/visualize_pose_model_error.ipynb)

# Set up
## Requirements
Cuda 8, cudnn 6, numpy, pandas, keras 2.2.4, tensorflow-gpu 1.4.0, glob, os, json, itertools, cv2, matplotlib, math, io, PIL, IPython, scipy


## Clone repo and download figshare data
`git clone https://github.com/cchamber/Infant_movement_assessment`

Download [repo.zip](https://figshare.com/s/10034c230ad9b2b2a6a4) from Figshare. Unzip. Add `data` and `models` folders to main directory  

# Measurement of pose model error
Compare performance of pose estimation models, e.g. before and after transfer learning.  
Compute and visualize error of trained pose estimation model using ground truth data (images and key point labels in [COCO format](www.cocodataset.org/).  
Requires ground-truth data (images and joint position labels). Ground truth labelled data is not provided here.  

## Get model predictions and load ground truth
- `src/pose_model/get_model_predictions_and_groundtruth.py`, line 61-80. Set the file paths to label ground-truth data, image data, and model files.

- Load ground-truth label data, generate model predictions and save images with predicted pose:
In `notebooks/master.ipynb`, run cells 1 and 2.

## Compute and visualize pose model error
- Compute model error and visualize model error:
Run `notebooks/visualize_pose_model_error.ipynb`  

Performance is quantified by the rmse, precision, and recall.  
We compute rmse normalized by bounding box size. RMSE is computed only for keypoints which are both in the ground truth data set and model predictions.  


# Extract pose from videos 
Download [pose_extraction.zip](https://figshare.com/s/10034c230ad9b2b2a6a4) and add `colab_openpose` to Google Drive  
Open `Get_pose_estimates_from_video.ipynb` with Google Colab    
Add videos for pose estimation to `colab_openpose/videos`  
The notebook will use the model `model.h5` in `colab_openpose/videos` to generate pose estimates. The `model.h5` file was trained with infant data. `trained_model.h5` is a copy of the infant pose model. The original cmu model (`original_cmu_model.h5`) can be used by renaming it to `model.h5`.  

Follow instructions in the notebook.  
Run the first cell of the notebook. Then go to the URL as instructed, connect to google account, and enter the generated authorization code in the notebook.  
Run the second cell. Pose estimates and videos with overlaid pose are output to the `output_files` folder.  
