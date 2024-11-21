
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
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

#add parser

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
if args.transform == 'random':
    transform='random'

version = args.version

predictions_path=os.path.join(create_dir_name(cfg),'lightning_logs','version_%d'%version,'predictions',transform)

subjs=os.listdir(predictions_path)

for subj in subjs:
    print(subj)
    path_to_data=os.path.join(predictions_path,subj,'diffusion')
    print(path_to_data)
    subprocess.run([path_to_bash,path_to_data,path_to_data,'_network'])
