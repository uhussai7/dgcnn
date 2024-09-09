from utils import diffusion
from utils.preprocessing import training_data
import os
from configs.config import get_cfg_defaults
import nibabel as nib
from pathlib import Path
import shutil


# cfg=get_cfg_defaults()
# subj_id=0 #not anything fundamental, just based on how it shows up on listdir
# subj=os.listdir(cfg.PATHS.TRAINING_PATH)[subj_id]

# dti_90=diffusion.dti(cfg.PATHS.TRAINING_PATH + subj + '/diffusion/90/dtifit/dtifit',cfg.PATHS.TRAINING_PATH + os.listdir(cfg.PATHS.TRAINING_PATH)[0] + '/diffusion/90/diffusion/nodif_brain_mask.nii.gz')
# diff_6=diffusion.diffVolume(cfg.PATHS.TRAINING_PATH + os.listdir(cfg.PATHS.TRAINING_PATH)[0] + '/diffusion/6/diffusion/')
# test=dti_90.signalFromDti(diff_6.bvecs_sorted[1].T)

# #save in scratch
# out_path=Path(cfg.PATHS.SCRATCH + subj + '/signal_from_dti_test/' + subj)
# out_path.mkdir(parents=True,exist_ok=True)

# source_folder=Path(cfg.PATHS.TRAINING_PATH + os.listdir(cfg.PATHS.TRAINING_PATH)[0] + '/diffusion/6/diffusion/')
# files=os.listdir(source_folder)
# for file in files:
#     shutil.copy(source_folder.joinpath(file), out_path)

# nii=nib.Nifti1Image(test,diff_6.vol.affine)
# nib.save(nii,out_path / 'data.nii.gz')

cfg=get_cfg_defaults()
sub=2 #not anything fundamental, just based on how it shows up on listdir
subjects=os.listdir(cfg.PATHS.TRAINING_PATH)
subs_path=cfg.PATHS.TRAINING_PATH
b_dirs=cfg.MODEL.NDIRS
N_patch=500
H=cfg.INPUT.H
Nc=cfg.INPUT.NC

diffusion_with_bdirs_path = subs_path + '/' + subjects[sub] + '/diffusion/' + str(b_dirs)  + '/diffusion/'
dti_with_bdirs_path       = subs_path + '/' + subjects[sub] + '/diffusion/' + str(b_dirs)  + '/dtifit/dtifit'
mask_for_bdirs_file       = subs_path + '/' + subjects[sub] + '/diffusion/' + str(b_dirs)  + '/diffusion/nodif_brain_mask.nii.gz'
dti_with_90dirs_path      = subs_path + '/' + subjects[sub] + '/diffusion/' + '90/'        + '/dtifit/dtifit'
mask_for_training_file    = subs_path + '/' + subjects[sub] + '/masks/mask.nii.gz'  
t1t2_path                 = subs_path + '/' + subjects[sub] + '/structural/'
this_subject =  training_data(diffusion_with_bdirs_path,
                              dti_with_bdirs_path,
                              dti_with_90dirs_path,
                              mask_for_bdirs_file,
                              t1t2_path,
                              mask_for_training_file,
                              H, 
                              N_train=N_patch,
                              Nc=Nc)

