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

#add parser

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

parser = argparse.ArgumentParser(
        description="Script for predicting"
    )

parser.add_argument(
    '-v', '--version',
    type=int,
    default=0,
    help='If continuing from last checkpoint provide the version'
)

parser.add_argument(
    '-c', '--config',
    type=str,
    required=True,
    help='Model config file'
)

parser.add_argument(
    '-t', '--transform',
    type=str,
    default='None',
    help='Rotate or randomize bvecs'
)

args=parser.parse_args()
print(args)


path_to_bash=os.path.join(os.path.dirname(__file__).split('/jobHandling')[0],'dataHandling','bash_scripts','dtifit_on_subjects.sh')

cfg=get_cfg_defaults()
cfg.merge_from_file(cfg.PATHS.CONFIG_PATH +args.config)#@'/Mixed/FatMixed-Nsubs-15.yaml') #'/Mixed/SlimMixed.yaml')#'/Slim2d_Nsubs-15.yaml')

transform='regular'
if args.transform == 'rotate':
    transform='rotate'
    print('Using rotated bvecs')
    opts=["PATHS.TESTING_PREPROC_PATH",cfg.PATHS.TESTING_ROTATED_PREPROC_PATH]
    opts=["PATHS.TESTING_PATH",cfg.PATHS.TESTING_ROTATED_PATH]
    cfg.merge_from_list(opts)
if args.transform == 'random':
    transform='random'
    print('Using random bvecs')
    opts=["PATHS.TESTING_PREPROC_PATH",cfg.PATHS.TESTING_RANDOM_PREPROC_PATH]
    opts=["PATHS.TESTING_PATH",cfg.PATHS.TESTING_RANDOM_PATH]
    cfg.merge_from_list(opts)

version = args.version

predictions_path=os.path.join(create_dir_name(cfg),'lightning_logs','version_%d'%version,'predictions',transform)

subjs=os.listdir(predictions_path)

accuracy_summary={'dFA_6':[],
                  'dtheta_6':[],
                  'dFA_model':[],
                  'dtheta_model':[]}

for subj in subjs:
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

    accuracy_summary['dFA_6'].append(FA_6_metric)
    accuracy_summary['dtheta_6'].append(V1_6_metric)
    accuracy_summary['dFA_model'].append(FA_pred_metric)
    accuracy_summary['dtheta_model'].append(V1_pred_metric)

accuracy_summary['dFA_6'] = np.asarray(accuracy_summary['dFA_6'])
accuracy_summary['dtheta_6'] = np.asarray(accuracy_summary['dtheta_6'])
accuracy_summary['dFA_model'] = np.asarray(accuracy_summary['dFA_model'])
accuracy_summary['dtheta_model'] = np.asarray(accuracy_summary['dtheta_model'])

print('---------------------------------SUMMARY--------------------------------------')
print('FA_6_metric:%.4f,%.4f'%(accuracy_summary['dFA_6'].mean(),accuracy_summary['dFA_6'].std()))
print('FA_pred_metric:%.4f,%.4f'%(accuracy_summary['dFA_model'].mean(),accuracy_summary['dFA_model'].std()))
print('V1_6_metric:%.4f,%.4f'%(accuracy_summary['dtheta_6'].mean(),accuracy_summary['dtheta_6'].std()))
print('V1_pred_metric:%.4f,%.4f'%(accuracy_summary['dtheta_model'].mean(),accuracy_summary['dtheta_model'].std()))
print('------------------------------------------------------------------------------')
print('\n')
print('\n')


accuracy_path=Path(os.path.join(create_dir_name(cfg),'lightning_logs','version_%d'%version,'accuracy',transform))

accuracy_path.mkdir(parents=True, exist_ok=True)

with open(str(accuracy_path) +'/accuracy_summary.pkl', 'wb') as f:
    pickle.dump(accuracy_summary, f)

    


#regular and random gt path: /home/u2hussai/project/u2hussai/niceData/testing/102816/diffusion/90/dtifit
#rotate gt path: /
# okay, so after checking seems like we can just go to testing pathing and do <subj>/diffusion/90/dtifit