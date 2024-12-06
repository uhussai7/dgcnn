import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
import torch
from dataHandling.dataGrab import PreprocData
from gconv.modules import *
from configs.config import get_cfg_defaults
import numpy as np
import h5py
import nibabel as nib
from pathlib import Path
from utils.models import predictor,create_dir_name
import argparse
import subprocess
import nibabel as nib

def angle(A,B):
    """
    Returns the angle difference between two vectors niis as another nii
    """
    A = A.reshape([-1, 3])
    B = B.reshape([-1, 3])
    dot = A*B
    dot = np.sum(dot,-1)
    dot = np.rad2deg(np.arccos(dot))
    for i in range(0,dot.shape[0]):
        if dot[i] > 90:
            dot[i] = 180 - dot[i]
    return dot

def diff(A,B):
    """
    Returns the scalar difference between two scalar niis as a nii
    """
    diff = np.abs(A-B)
    return diff

def get_unique_labels():
    cfg=get_cfg_defaults()
    subjs=[]
    for subj in os.listdir(cfg.PATHS.TESTING_PATH):
        if os.path.isfile(os.path.join(cfg.PATHS.TESTING_PATH,subj,'masks','wmparc.nii.gz')):
            subjs.append(subj)
    roi_labels=np.unique(nib.load(os.path.join(cfg.PATHS.TESTING_PATH,subjs[0],'masks','wmparc.nii.gz')).get_fdata())
    return [int(label) for label in roi_labels if 1<= label <=255],subjs

def read_roi_names():

    roi_labels, subjs=get_unique_labels()

    roi_names_file='/home/u2hussai/dgcnn/dataHandling/freesurfer/FreeSurferColorLUT.txt'
    with open(roi_names_file, 'r') as file:
        lines = file.readlines()
    results = []
    for line in lines:
        # Split the line into parts
        parts = line.split()
        # Extract the first column (number) and the second column (name

        col1 = parts[0]
        col2 = parts[1]
        
        if int(col1) in roi_labels:
            results.append((col1, col2))
    return results, subjs

def merge_cfg(config_path):
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.CONFIG_PATH+config_path)
    return create_dir_name(cfg)

# models_config = ['/2d/Fat2d-Nsubs-15.yaml', '/3d/Fat3d-Nsubs-40.yaml', 
#                  '/Mixed/FatMixed-Nsubs-15.yaml', '/MixedR/VeryFatMixedR-Nsubs-15.yaml']

models_config = ['/3d/Fat3d-Nsubs-40.yaml', '/MixedR/VeryFatMixedR-Nsubs-15.yaml']
version=[0,3]
roi_vals_names, subjs=read_roi_names()
roi_names=[roi[1] for roi in roi_vals_names]

print(subjs)

print(len(roi_names))

Nmodels=3 
Nrois=len(roi_names)
Ncon=2 #fa versus theta
Nsubs=len(subjs)
acc=np.empty([Nmodels,Ncon,Nrois,Nsubs]) # first index is mean and std
transform='regular'

# cfg=get_cfg_defaults()
# roi_count=0
# for roi_val,roi_name in roi_vals_names:
#     V1_acc=np.zeros([3,Nsubs])
#     FA_acc=np.zeros([3,Nsubs])
#     for s,subj in enumerate(subjs):
#         print(roi_name,subj)
#         roi_mask=nib.load(os.path.join(cfg.PATHS.TESTING_PATH,subj,'masks','wmparc.nii.gz'))

#         FA_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_FA.nii.gz')
#         FA_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_FA.nii.gz')
#         FA_pred1_path=os.path.join(os.path.join(merge_cfg(models_config[0]),'lightning_logs','version_%d'%version[0],'predictions',transform),subj,'diffusion','dtifit_network_FA.nii.gz')
#         FA_pred2_path=os.path.join(os.path.join(merge_cfg(models_config[1]),'lightning_logs','version_%d'%version[1],'predictions',transform),subj,'diffusion','dtifit_network_FA.nii.gz')

#         V1_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_V1.nii.gz')
#         V1_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_V1.nii.gz')
#         V1_pred1_path=os.path.join(os.path.join(merge_cfg(models_config[0]),'lightning_logs','version_%d'%version[0],'predictions',transform),subj,'diffusion','dtifit_network_V1.nii.gz')
#         V1_pred2_path=os.path.join(os.path.join(merge_cfg(models_config[1]),'lightning_logs','version_%d'%version[1],'predictions',transform),subj,'diffusion','dtifit_network_V1.nii.gz')

#         FA_gt=nib.load(FA_gt_path).get_fdata()[roi_mask==roi_val]
#         FA_6=nib.load(FA_6_path).get_fdata()[roi_mask==roi_val]
#         FA_model1=nib.load(FA_pred1_path).get_fdata()[roi_mask==roi_val]
#         FA_model2=nib.load(FA_pred2_path).get_fdata()[roi_mask==roi_val]

#         V1_gt=nib.load(V1_gt_path).get_fdata()[roi_mask==roi_val,:]
#         V1_6=nib.load(V1_6_path).get_fdata()[roi_mask==roi_val,:]
#         V1_model1=nib.load(V1_pred1_path).get_fdata()[roi_mask==roi_val,:]
#         V1_model2=nib.load(V1_pred2_path).get_fdata()[roi_mask==roi_val,:]

#         FA_6_metric=np.nanmean(diff(FA_gt,FA_6))
#         FA_pred1_metric=np.nanmean(diff(FA_gt,FA_model1))
#         FA_pred2_metric=np.nanmean(diff(FA_gt,FA_model2))

#         V1_6_metric=np.nanmean(angle(V1_gt,V1_6))
#         V1_pred1_metric=np.nanmean(angle(V1_gt,V1_model1))
#         V1_pred2_metric=np.nanmean(angle(V1_gt,V1_model2))

#         FA_acc[0,s]=FA_6_metric
#         FA_acc[1,s]=FA_pred1_metric
#         FA_acc[2,s]=FA_pred2_metric

#         V1_acc[0,s]=V1_6_metric
#         V1_acc[1,s]=V1_pred1_metric
#         V1_acc[2,s]=V1_pred2_metric
    
#     acc[0,:,0,roi_count]=FA_acc.mean(-1)
#     acc[1,:,0,roi_count]=FA_acc.std(-1)
#     acc[0,:,1,roi_count]=V1_acc.mean(-1)
#     acc[1,:,1,roi_count]=V1_acc.std(-1)
    
#     roi_count+=

cfg=get_cfg_defaults()
roi_count=0
for s,subj in enumerate(subjs):
    roi_mask=nib.load(os.path.join(cfg.PATHS.TESTING_PATH,subj,'masks','wmparc.nii.gz')).get_fdata().astype(int)

    #print(np.unique(roi_mask))

    FA_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_FA.nii.gz')
    FA_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_FA.nii.gz')
    FA_pred1_path=os.path.join(os.path.join(merge_cfg(models_config[0]),'lightning_logs','version_%d'%version[0],'predictions',transform),subj,'diffusion','dtifit_network_FA.nii.gz')
    FA_pred2_path=os.path.join(os.path.join(merge_cfg(models_config[1]),'lightning_logs','version_%d'%version[1],'predictions',transform),subj,'diffusion','dtifit_network_FA.nii.gz')

    V1_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_V1.nii.gz')
    V1_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_V1.nii.gz')
    V1_pred1_path=os.path.join(os.path.join(merge_cfg(models_config[0]),'lightning_logs','version_%d'%version[0],'predictions',transform),subj,'diffusion','dtifit_network_V1.nii.gz')
    V1_pred2_path=os.path.join(os.path.join(merge_cfg(models_config[1]),'lightning_logs','version_%d'%version[1],'predictions',transform),subj,'diffusion','dtifit_network_V1.nii.gz')

    FA_gt_=nib.load(FA_gt_path).get_fdata()
    FA_6_=nib.load(FA_6_path).get_fdata()
    FA_model1_=nib.load(FA_pred1_path).get_fdata()
    FA_model2_=nib.load(FA_pred2_path).get_fdata()

    V1_gt_=nib.load(V1_gt_path).get_fdata()
    V1_6_=nib.load(V1_6_path).get_fdata()
    V1_model1_=nib.load(V1_pred1_path).get_fdata()
    V1_model2_=nib.load(V1_pred2_path).get_fdata()

    roi_count=0
    for roi_val,roi_name in roi_vals_names:
        #print(np.unique(roi_mask))
        roi_val=int(roi_val)
        print(roi_val,roi_name,subj,np.sum((roi_mask==int(roi_val))))

        FA_gt=FA_gt_[roi_mask==roi_val]
        FA_6=FA_6_[roi_mask==roi_val]
        FA_model1=FA_model1_[roi_mask==roi_val]
        FA_model2=FA_model2_[roi_mask==roi_val]

        V1_gt=V1_gt_[roi_mask==roi_val,:]
        V1_6=V1_6_[roi_mask==roi_val,:]
        V1_model1=V1_model1_[roi_mask==roi_val,:]
        V1_model2=V1_model2_[roi_mask==roi_val,:]

        FA_6_metric=np.nanmean(diff(FA_gt,FA_6))
        FA_pred1_metric=np.nanmean(diff(FA_gt,FA_model1))
        FA_pred2_metric=np.nanmean(diff(FA_gt,FA_model2))

        V1_6_metric=np.nanmean(angle(V1_gt,V1_6))
        V1_pred1_metric=np.nanmean(angle(V1_gt,V1_model1))
        V1_pred2_metric=np.nanmean(angle(V1_gt,V1_model2))

        #FA_acc[0,s]+=FA_6_metric
        #FA_acc[1,s]+=FA_pred1_metricw
        #FA_acc[2,s]+=FA_pred2_metric

        #V1_acc[0,s]+=V1_6_metric
        #V1_acc[1,s]+=V1_pred1_metric
        #V1_acc[2,s]+=V1_pred2_metric

        #print(diff(FA_gt,FA_6))

        acc[:,0,roi_count,s]=np.asarray([FA_6_metric,FA_pred1_metric,FA_pred2_metric])
        acc[:,1,roi_count,s]=np.asarray([V1_6_metric,V1_pred1_metric,V1_pred2_metric])
        roi_count+=1


table= {
        'roi_names':[],
        'fa_6':[],
        'fa_base':[],
        'fa_model':[],
        'theta_6':[],
        'theta_base':[],
        'theta_model':[]
        }

roi_count=0
for roi_val,roi_name in roi_vals_names:
    table['roi_names'].append(roi_name)
    table['fa_6'].append(          '%.2f, %.2f'%(np.nanmean(acc[0,0,roi_count]),np.nanstd(acc[0,0,roi_count]) ))
    table['fa_base'].append(       '%.2f, %.2f'%(np.nanmean(acc[1,0,roi_count]),np.nanstd(acc[1,0,roi_count]) ))
    table['fa_model'].append(      '%.2f, %.2f'%(np.nanmean(acc[2,0,roi_count]),np.nanstd(acc[2,0,roi_count]) ))
    table['theta_6'].append(       '%.2f, %.2f'%(np.nanmean(acc[0,1,roi_count]),np.nanstd(acc[0,1,roi_count]) ))
    table['theta_base'].append(    '%.2f, %.2f'%(np.nanmean(acc[1,1,roi_count]),np.nanstd(acc[1,1,roi_count]) ))
    table['theta_model'].append(   '%.2f, %.2f'%(np.nanmean(acc[2,1,roi_count]),np.nanstd(acc[2,1,roi_count]) ))
    roi_count+=1

import pandas as pd
df=pd.DataFrame(data=table)



print(df.to_latex())
# Nrois=len(roi_names)
# Nsubs=25
# Nmodels=3 #no model, base, gauge
# Ntype=2 #fa theta
# acc=np.zeros([Nmodels,Ntype,Nrois,Nsubs]) #

# #@'/Mixed/FatMixed-Nsubs-15.yaml') #'/Mixed/SlimMixed.yaml')#'/Slim2d_Nsubs-15.yaml')

# transform='regular'



# for c,config in enumerate(models_config):    
#     cfg=get_cfg_defaults()
#     cfg.merge_from_file(cfg.PATHS.CONFIG_PATH +config)
#     predictions_path=os.path.join(create_dir_name(cfg),'lightning_logs','version_%d'%version[c],'predictions',transform)
#     subjs=os.listdir(predictions_path)

#     for subj in subjs:
        
#         mask_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'masks','mask.nii.gz')
#         rois_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'freesurfer','wmparc.nii.gz')
        
#         FA_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_FA.nii.gz')
#         FA_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_FA.nii.gz')
#         FA_pred_path=os.path.join(predictions_path,subj,'diffusion','dtifit_network_FA.nii.gz')

#         V1_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_V1.nii.gz')
#         V1_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_V1.nii.gz')
#         V1_pred_path=os.path.join(predictions_path,subj,'diffusion','dtifit_network_V1.nii.gz')

#         mask=nib.load(mask_path).get_fdata()

#         FA_gt=nib.load(FA_gt_path).get_fdata()
#         FA_6=nib.load(FA_6_path).get_fdata()
#         FA_pred=nib.load(FA_pred_path).get_fdata()

#         V1_gt=nib.load(V1_gt_path).get_fdata()
#         V1_6=nib.load(V1_6_path).get_fdata()
#         V1_pred=nib.load(V1_pred_path).get_fdata()

#         for roi_val,roi_name in roi_vals_names:
        

#         # print(V1_gt.shape)

#         FA_6_metric=np.nanmean(diff(FA_gt,FA_6))
#         FA_pred_metric=np.nanmean(diff(FA_gt,FA_pred))

#         V1_6_metric=np.nanmean(angle(V1_gt,V1_6))
#         V1_pred_metric=np.nanmean(angle(V1_gt,V1_pred))

#         print('------------------------------------------------------------------------------')
#         print('subj-%s'%str(subj))
#         print('FA_6_metric:%f'%FA_6_metric)
#         print('FA_pred_metric:%f'%FA_pred_metric)
#         print('V1_6_metric:%f'%V1_6_metric)
#         print('V1_pred_metric:%f'%V1_pred_metric)
#         print('------------------------------------------------------------------------------')
#         print('\n')
#         print('\n')


#     # fa_6=np.zeros([len(roi_names),len(subjs)])
#     # fa_base=np.zeros([len(roi_names),len(subjs)])
#     # fa_model=np.zeros([len(roi_names),len(subjs)])

#     # theta_6=np.zeros([len(roi_names),len(subjs)])
#     # theta_base=np.zeros([len(roi_names),len(subjs)])
#     # theta_model=np.zeros([len(roi_names),len(subjs)])
