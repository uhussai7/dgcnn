
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
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




cfg=get_cfg_defaults()
cfg.merge_from_file(cfg.PATHS.CONFIG_PATH +args.config)#@'/Mixed/FatMixed-Nsubs-15.yaml') #'/Mixed/SlimMixed.yaml')#'/Slim2d_Nsubs-15.yaml')

transform='regular'
if args.transform == 'rotate':
    transform='rotate'
    print('Using rotated bvecs')
    opts=["PATHS.TESTING_PREPROC_PATH",cfg.PATHS.TESTING_ROTATED_PREPROC_PATH]
    cfg.merge_from_list(opts)
if args.transform == 'random':
    transform='random'
    print('Using random bvecs')
    opts=["PATHS.TESTING_PREPROC_PATH",cfg.PATHS.TESTING_RANDOM_PREPROC_PATH]
    cfg.merge_from_list(opts)

version = args.version

subjs= os.listdir(cfg.PATHS.TESTING_PREPROC_PATH) #'211417'


for subj in subjs:
    checkpoint_path=os.path.join(create_dir_name(cfg),'lightning_logs','version_%d'%version,'checkpoints')
    checkpoints=os.listdir(checkpoint_path)
    print('These are the checkpoints',checkpoints)
    epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
    max_epoch_ind=np.argsort(epochs)[-1]
    max_epoch=epochs[max_epoch_ind]
    resume_checkpoint=os.path.join(checkpoint_path,checkpoints[max_epoch_ind])
    print('Loading checkpoint:',resume_checkpoint)
    model=DNet.load_from_checkpoint(resume_checkpoint,cfg=cfg)

    path_to_save=os.path.join(cfg.PATHS.MODELS_PATH,
                              cfg.MODEL.NAME,
                              'Nsubs-%d'%int(cfg.INPUT.NSUBJECTS),
                              'lightning_logs',
                              'version_%d'%version,
                              'predictions',
                              transform,
                              str(subj),
                              'diffusion'
                                )

    pred=predictor(cfg,model,subj,path_to_save)
    pred.predict()


# import icosahedron
# import predictingScalar
# import torch 
# import os
# import sys
# import numpy as np



# #remove "net" at end 
# netpath_5='/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/bvec-dirs-6_type-ip-on_Ntrain-1498_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'
# netpath_10='/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/bvec-dirs-6_type-ip-on_Ntrain-3022_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'
# netpath_15='/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/bvec-dirs-6_type-ip-on_Ntrain-4487_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'


# subs = np.array([5,10,15])
# nsubs=int(sys.argv[1])
# subs_ind = np.where(subs==nsubs)[0][0]
# nets=[netpath_5,netpath_10,netpath_15]
# netpath=nets[subs_ind]

# print('Using network at path: ',netpath)

# subjects=os.listdir('/home/u2hussai/project/u2hussai/niceData/testing/')

# for i in range(0,25):
#     sub =subjects[i]
#     out_dir='/home/u2hussai/projects/ctb-akhanf/u2hussai/predictions_'+str(nsubs) + '/'+sub+'/'
    
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#         print('making prediction on '+out_dir)

#     datapath= '/home/u2hussai/project/u2hussai/niceData/testing/'+sub+'/diffusion/6/'
#     tpath = '/home/u2hussai/project/u2hussai/niceData/testing/'+sub+'/structural/'
#     maskfile = '/home/u2hussai/project/u2hussai/niceData/testing/'+sub+'/masks/mask.nii.gz'
#     print(datapath)
#     ico = icosahedron.icomesh(m=4)

#     predictor = predictingScalar.residual5dPredictorScalar(datapath + 'diffusion/',
#                                                 datapath + 'dtifit/dtifit',
#                                                 datapath + 'dtifit/dtifit',
#                                                 tpath,
#                                                 maskfile,
#                                                 netpath,
#                                                 B=1,
#                                                 H=5,
#                                                 Nc=16,
#                                                 Ncore=100,
#                                                 core=ico.core_basis,
#                                                 core_inv=ico.core_basis_inv,
#                                                 zeros=ico.zeros,
#                                                 I=ico.I_internal,
#                                                 J=ico.J_internal)

#     if torch.cuda.is_available():
#         predictor.net=predictor.net.cuda().eval()


#     predictor.predict(out_dir)




