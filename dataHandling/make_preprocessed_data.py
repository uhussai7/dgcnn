from dataGrab import data_grab_save_preproc
from configs.config import get_cfg_defaults
import h5py
from pathlib import Path
import os
import sys



cfg=get_cfg_defaults()
#cfg.INPUT.N_SUBJECTS=1

test_or_train='test'#sys.argv[1]

if test_or_train=='train':
    source_path=cfg.PATHS.TRAINING_PATH
    target_path=cfg.PATHS.TRAINING_PREPROC_PATH
else:
    source_path=cfg.PATHS.TESTING_ROTATED_PATH
    target_path=cfg.PATHS.TESTING_ROTATED_PREPROC_PATH

# subjects=[102816,  104820,  108020,  130518,  150019,  194746,  211417,  304727,  395251,  436239,  693764,  723141 , 952863,
# 103111,  107321 , 127731,  144125,  172332,  202113,  211821,  310621,  406836,  561949,  695768,  902242]
# #os.listdir(source_path)
subjects=[  130518,    194746,   304727,  436239,  693764,  723141 , 952863,
103111,  107321 , 127731,  144125,  172332,  202113,  211821,  310621,  406836,  561949,  695768]

for sub in subjects:
    print(sub)
    data_grab_save_preproc(str(sub),source_path,target_path)

# out_path=cfg.PATHS.TRAINING_PREPROC_PATH
# f= h5py.File(str(out_path) + subjects[0] +'/data.h5','r')
# with h5py.File(str(out_path) + subjects[0] +'/data.h5','r') as f:
#     loaded_tensors = {name: torch.tensor(f[name][:]) for name in tensor_dict.keys()}