#idetify which file you need
#map it to the correct res need reference for that
#so called cut pad
#move to correct folder

#lets use the white matter parcellations file

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import diffusion
from configs.config import get_cfg_defaults
import subprocess
import os
import nibabel as nib
from nibabel import processing
from utils.cutnifti import cuts_and_pad

def make_atlas(source_path,target_path): #target path should already exist
    ref=nib.load(str(source_path) + '/T1w/Diffusion/nodif_brain_mask.nii.gz') #reference for resampling
    wm_parc=nib.load( str(source_path) + '/T1w' + '/wmparc.nii.gz')
    masknii = processing.resample_from_to(wm_parc,ref,order=0)
    masknii = cuts_and_pad(masknii)
    #we are doing this for the test subjects
    nib.save(masknii,target_path + '/wmparc.nii.gz')

cfg=get_cfg_defaults()

#get test subjects
subjs=os.listdir(cfg.PATHS.TESTING_PATH)

for subj in subjs:
    print(subj)
    source_path=cfg.PATHS.RAW_PATH + '/%s'%subj
    target_path=cfg.PATHS.TESTING_PATH + '/%s'%subj+'/masks/'
    make_atlas(source_path,target_path)


