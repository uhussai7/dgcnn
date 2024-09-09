from dataGrab import data_grab_save_preproc
from configs.config import get_cfg_defaults
import h5py
from pathlib import Path
import os
import sys



cfg=get_cfg_defaults()
#cfg.INPUT.N_SUBJECTS=1

test_or_train=sys.argv[1]

if test_or_train=='train':
    source_path=cfg.PATHS.TRAINING_PATH
    target_path=cfg.PATHS.TRAINING_PREPROC_PATH
else:
    source_path=cfg.PATHS.TESTING_PATH
    target_path=cfg.PATHS.TESTING_PREPROC_PATH

subjects=os.listdir(source_path)

for sub in subjects:
    data_grab_save_preproc(sub,source_path,target_path)

# out_path=cfg.PATHS.TRAINING_PREPROC_PATH
# f= h5py.File(str(out_path) + subjects[0] +'/data.h5','r')
# with h5py.File(str(out_path) + subjects[0] +'/data.h5','r') as f:
#     loaded_tensors = {name: torch.tensor(f[name][:]) for name in tensor_dict.keys()}