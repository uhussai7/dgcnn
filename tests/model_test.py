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
cfg=get_cfg_defaults()
cfg.merge_from_file(cfg.PATHS.CONFIG_PATH + '/SlimMixed.yaml')#2d_Nsubs-15.yaml')
# subj='211417'
data_module=PreprocData(cfg)#,subj=subj)
trainer=DgcnnTrainer(cfg,devices=torch.cuda.device_count())#,strategy='ddp_find_unused_parameters_true',callbacks=[TimingCallback()])
model=DNet(cfg)
trainer.fit(model,datamodule=data_module)
# checkpoint_path=os.path.join(trainer._create_dir_name(cfg),'lightning_logs','version_1','checkpoints')
# checkpoints=os.listdir(checkpoint_path)
# print('These are the checkpoints',checkpoints)
# epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
# max_epoch_ind=np.argsort(epochs)[-1]
# max_epoch=epochs[max_epoch_ind]
# resume_checkpoint=os.path.join(checkpoint_path,checkpoints[max_epoch_ind])
# print('Loading checkpoint:',resume_checkpoint)
# model=DNet.load_from_checkpoint(resume_checkpoint,cfg=cfg)
# prediction=trainer.predict(model,datamodule=data_module)
# predictions=torch.cat(prediction)
# #put the volume together
# with h5py.File(data_module.data_file, 'r') as f:
#     xp=f['xp'][:]
#     yp=f['yp'][:]
#     zp=f['zp'][:]
#     S0=f['S0'][:]
#     Xflatmean=f['Xflatstd'][()]
#     Xflatstd=f['Xflatstd'][()]
#     Xflatstd=f['Xflatstd'][()]
# #get output ready
# signal=predictions.squeeze(-3) + data_module.data_list['Xflat']
# signal=signal*Xflatstd + Xflatmean

# h,w=signal.shape[-2:]
# signal=signal.reshape(-1,h,w)

# Ncore=N=100
# ico=icosahedron.icomesh(m=int(cfg.INPUT.H)-1)
# core=ico.core_basis
# I=ico.I_internal
# J=ico.J_internal
# N_random=N-2
# rng=np.random.default_rng()
# inds=rng.choice(N-2,size=N_random,replace=False)+1
# inds[0]=0


# bvals=np.zeros(N_random)
# x_bvecs=np.zeros(N_random)
# y_bvecs=np.zeros(N_random)
# z_bvecs=np.zeros(N_random)

# x_bvecs[1:]=ico.X_in_grid[core==1].flatten()[inds[1:]]
# y_bvecs[1:]=ico.Y_in_grid[core==1].flatten()[inds[1:]]
# z_bvecs[1:]=ico.Z_in_grid[core==1].flatten()[inds[1:]]

# bvals[1:]=1000


# diff_out=np.zeros(S0.shape + (N_random,))
# diff_out[:,:,:,0]=S0

# signal = signal[:,core==1]
# diff_out[xp,yp,zp,1:] = signal[:,inds[1:]]

# test_path=Path(os.path.join(cfg.PATHS.SCRATCH,'dgcnn',subj))
# test_path.mkdir(parents=True,exist_ok=True)


# diff_out=nib.Nifti1Image(diff_out,np.eye(4))
# nib.save(diff_out,test_path.joinpath('data_network.nii.gz'))

# #write the bvecs and bvals
# fbval = open(test_path.joinpath('bvals_network'), "w")
# for bval in bvals:
#     fbval.write(str(bval)+" ")
# fbval.close()

# fbvecs = open(test_path.joinpath('bvecs_network'),"w")
# for x in x_bvecs:
#     fbvecs.write(str(x)+ ' ')
# fbvecs.write('\n')
# for y in y_bvecs:
#     fbvecs.write(str(y)+ ' ')
# fbvecs.write('\n')
# for z in z_bvecs:
#     fbvecs.write(str(z)+ ' ')
# fbvecs.write('\n')
# fbvecs.close()

# #the mask
# mask=np.zeros_like(S0)
# mask[xp,yp,zp]=data_module.data_list['mask_train'].flatten()
# mask=nib.Nifti1Image(mask,np.eye(4))
# nib.save(diff_out,test_path.joinpath('nodif_brain_mask.nii.gz'))

# import matplotlib.pyplot as plt
# fig,ax=plt.subplots(10,2,figsize=(35,60))
# for i in range(0,10):
#     ax[i,0].imshow(data_module.data_list['Xflat'][i,8,8,8,:,:])
#     ax[i,1].imshow(data_module.data_list['Xflat'][i,8,8,8,:,:]+predictions[i,8,8,8,0,:,:])
# plt.tight_layout()