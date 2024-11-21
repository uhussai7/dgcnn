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
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

cfg=get_cfg_defaults()
cfg.merge_from_file(cfg.PATHS.CONFIG_PATH + sys.argv[1])
data_module=PreprocData(cfg)
trainer=DgcnnTrainer(cfg,devices=torch.cuda.device_count(),
                     strategy='ddp_find_unused_parameters_true',callbacks=[TimingCallback()],#,MemoryLoggerCallback()],
                     max_epochs=cfg.TRAIN.MAX_EPOCHS)
model=DNet(cfg)

# trainer.fit(model,datamodule=data_module)


#handle the versioning correctly
version=int(sys.argv[2]) if len(sys.argv) >2 else 0
root_dir=trainer.default_root_dir
checkpoint_path=root_dir+'/lightning_logs/version_%d/checkpoints/'%version
resume_checkpoint=None
if version is None:
    trainer.fit(model,datamodule=data_module)
else:
    if os.path.exists(checkpoint_path):
        checkpoints=os.listdir(checkpoint_path)
        if len(checkpoints)>0:
            print('These are the checkpoints in the version provided...',checkpoints)
            epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
            max_epoch_ind=np.argsort(epochs)[-1]
            max_epoch=epochs[max_epoch_ind]
            resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
        else:
            print('No checkpoints found, will put checkpoints from training in the version provided')
    else:
        print('Version is provided but version path does not exist. I will make it.')
        Path(checkpoint_path).mkdir(parents=True,exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path)    
    logger=CSVLogger(save_dir=root_dir,version=version)
    trainer=DgcnnTrainer(cfg,devices=torch.cuda.device_count(),
                     strategy='ddp_find_unused_parameters_true',callbacks=[TimingCallback(),checkpoint_callback],#,MemoryLoggerCallback()],
                     max_epochs=cfg.TRAIN.MAX_EPOCHS)
    trainer.fit(model,datamodule=data_module,ckpt_path=resume_checkpoint)














# import os
# import torch
# from preprocessing import training_data
# import numpy as np
# import icosahedron
# import dihedral12 as d12
# import trainingScalars as training
# from dataGrab import data_grab
# from torch.nn import functional as F
# from torch import nn
# import sys

# #grab training data
# N_subjects=int(sys.argv[1])
# X, Xflat, S0X, Y, S0Y, mask_train, interp_matrix, interp_matrix_ind=data_grab(N_subjects,'/home/u2hussai/project/u2hussai/niceData/training/')

# #initalize the network
# H=5 #size if 2d grid will be h=5+1 w=5*(5+1)
# Nc=16 # 16x16x16 patch size
# ico=icosahedron.icomesh(m=H-1) #get the icosahedorn
# I, J, T = d12.padding_basis(H=H)  # for padding

# Ndirs = 6 #number of diffusion directions
# Nscalars = 1 #this is to hold on to S0
# Nshells = 1 #we use 1 shell
# Cinp = 64 #this is the number of "effective" 3d filters
# Cin = Cinp*(Nscalars + Nshells*Ndirs) #the number of actual filters needed

# #3d convs
# filterlist3d=[9,Cin,Cin,Cin] #3d layers, 9 = 2 (T1,T2) + 7 (S0 + 6 diffusion directions)
# activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)] #3d layers activatiions

# #2d gconvs
# gfilterlist2d =[Cinp,Cinp,Cinp,Cinp,1] #gconv layers
# gactivationlist2d = [F.relu for i in range(0,len(gfilterlist2d)-1)] #gconv layers activations
# gactivationlist2d[-1]=None #turning of last layer activation

# #model configuration
# modelParams={'H':H,
#              'shells':Nshells,
#              'gfilterlist': gfilterlist2d,
#              'linfilterlist': None,
#              'gactivationlist': gactivationlist2d,
#              'lactivationlist': None,
#              'filterlist3d' : filterlist3d,
#              'activationlist3d':activationlist3d,
#              'loss': nn.MSELoss(),
#              'bvec_dirs': Ndirs,
#              'batch_size': 1,
#              'lr': 1e-4,
#              'factor': 0.5,
#              'Nepochs': 20,
#              'patience': 7,
#              'Ntrain': X.shape[0],
#              'Ntest': 1,
#              'Nvalid': 1,
#              'interp': 'inverse_distance',
#              'basepath': '/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/',
#              'type': 'ip-on',
#              'misc':'residual5dscalar'
#             }

# #training class
# trnr = training.trainer(modelParams,
#                         Xtrain=X, Ytrain=Y-Xflat, S0Ytrain=S0Y-S0X, interp_matrix_train=interp_matrix,
#                         interp_matrix_ind_train=interp_matrix_ind,mask=mask_train,
#                         Nscalars=Nscalars,Ndir=Ndirs,ico=ico,
#                         B=1,Nc=Nc,Ncore=100,core=ico.core_basis,
#                         core_inv=ico.core_basis_inv,
#                         zeros=ico.zeros,
#                         I=I,J=J)
# trnr.makeNetwork()
# trnr.net=trnr.net.cuda()
# trnr.save_modelParams() #save model parameters
# trnr.train() #this will save checkpoints