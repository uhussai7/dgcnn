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
from pathlib import Path
import pickle

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


nsubs=[5,10,15,20,30,40]
configs=['/3d/Fat3d-Nsubs-%d.yaml'%nsub for nsub in nsubs]
version=0

transform='regular'
base='_base_network'

N_pred=25

FA_table=np.zeros([len(nsubs),N_pred,2])
V1_table=np.zeros([len(nsubs),N_pred,2])

for c,config in enumerate(configs):

    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.CONFIG_PATH +config)
    
    predictions_path=os.path.join(create_dir_name(cfg),'lightning_logs','version_%d'%version,'predictions',transform)

    subjs=os.listdir(predictions_path)

    for s,subj in enumerate(subjs):
        print(subj)
        mask_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'masks','mask.nii.gz')
        
        FA_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_FA.nii.gz')
        FA_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_FA.nii.gz')
        FA_pred_path=os.path.join(predictions_path,subj,'diffusion','dtifit_network_FA.nii.gz')

        V1_gt_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','90','dtifit','dtifit_V1.nii.gz')
        V1_6_path=os.path.join(cfg.PATHS.TESTING_PATH,subj,'diffusion','6','dtifit','dtifit_V1.nii.gz')
        V1_pred_path=os.path.join(predictions_path,subj,'diffusion','dtifit_network_V1.nii.gz')

        mask=nib.load(mask_path).get_fdata()

        FA_gt=nib.load(FA_gt_path).get_fdata()[mask==1]
        FA_6=nib.load(FA_6_path).get_fdata()[mask==1]
        FA_pred=nib.load(FA_pred_path).get_fdata()[mask==1]

        V1_gt=nib.load(V1_gt_path).get_fdata()[mask==1,:]
        V1_6=nib.load(V1_6_path).get_fdata()[mask==1,:]
        V1_pred=nib.load(V1_pred_path).get_fdata()[mask==1,:]

        # print(V1_gt.shape)

        FA_6_metric=np.nanmean(diff(FA_gt,FA_6))
        FA_pred_metric=np.nanmean(diff(FA_gt,FA_pred))

        V1_6_metric=np.nanmean(angle(V1_gt,V1_6))
        V1_pred_metric=np.nanmean(angle(V1_gt,V1_pred))

        # print('------------------------------------------------------------------------------')
        # print('subj-%s'%str(subj))
        # print('FA_6_metric:%f'%FA_6_metric)
        # print('FA_pred_metric:%f'%FA_pred_metric)
        # print('V1_6_metric:%f'%V1_6_metric)
        # print('V1_pred_metric:%f'%V1_pred_metric)
        # print('------------------------------------------------------------------------------')
        # print('\n')
        # print('\n')

        FA_table[c,s,0]=FA_6_metric
        FA_table[c,s,1]=FA_pred_metric
        V1_table[c,s,0]=V1_6_metric
        V1_table[c,s,1]=V1_pred_metric

print('saving as: ',cfg.PATHS.SCRATCH + 'FA%s_table.npy'%base)
np.save(cfg.PATHS.SCRATCH + 'FA%s_table.npy'%base,FA_table)
print('saving as: ',cfg.PATHS.SCRATCH + 'V1%s_table.npy'%base)
np.save(cfg.PATHS.SCRATCH + 'V1%s_table.npy'%base,V1_table)

        

