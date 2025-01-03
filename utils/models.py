
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
import nibabel as nib
import shutil

def create_dir_name(cfg):
    out_path=Path(os.path.join(cfg.PATHS.MODELS_PATH,cfg.MODEL.NAME,
                                   'Nsubs-%d'%int(cfg.INPUT.NSUBJECTS)))
    return str(out_path)    

class predictor:
    def __init__(self,cfg,model,subj,save_path):
        self.cfg=cfg
        self.model=model
        self.subj=subj
        self.trainer=DgcnnTrainer(cfg,devices=torch.cuda.device_count(),logger=False)#,strategy='ddp_find_unused_parameters_true',callbacks=[TimingCallback()])
        self.save_path=str(save_path)
        self.data_module=PreprocData(self.cfg,subj=self.subj)
        
        #self.prepare_data()
        self.get_affine()

    def prepare_data(self):
        if self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D == 0:
            with h5py.File(self.data_module.data_file, 'r') as f:
                self.xp=f['xp'][:]
                self.yp=f['yp'][:]
                self.zp=f['zp'][:]
                self.S0=f['S0'][:]
                self.Xflatmean=f['Xflatstd'][()]
                self.Xflatstd=f['Xflatstd'][()]
        else:
            with h5py.File(self.data_module.data_file, 'r') as f:
                self.xp=f['xp'][:]
                self.yp=f['yp'][:]
                self.zp=f['zp'][:]
                self.S0=f['S0'][:]
                self.Xflatmean=f['Xflatstd'][()]
                self.Xflatstd=f['Xflatstd'][()]

    def get_affine(self):
        affine_path=Path(os.path.join(self.cfg.PATHS.TESTING_PATH,self.subj,'diffusion','90','diffusion'))
        self.affine=nib.load(affine_path.joinpath('nodif_brain_mask.nii.gz')).affine

    def brain_mask(self):
        mask_path=Path(os.path.join(self.cfg.PATHS.TESTING_PATH,self.subj,'masks'))
        return nib.load(mask_path.joinpath('mask.nii.gz')).get_fdata()

    def predict(self):
        predictions=self.trainer.predict(self.model,datamodule=self.data_module)
        predictions=torch.cat(predictions)

        self.prepare_data()

        if self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D == 0:
            
            print('predictions have shape',predictions.shape)
            print('X ahs shape',self.data_module.data_list['X'].moveaxis(1,-1)[3:].shape)

            signal=predictions+self.data_module.data_list['X'].moveaxis(1,-1)[:,:,:,:,3:]
            signal=signal*self.Xflatstd + self.Xflatmean
            signal=signal.reshape(-1,self.cfg.MODEL.NDIRS)

            print('signal has shape',signal.shape)

            diff_out=np.zeros(self.S0.shape + (signal.shape[-1]+1,))
            diff_out[:,:,:,0]=self.S0

            print('diff_out has shape',diff_out.shape)

            diff_out[self.xp,self.yp,self.zp,1:] = signal

            #need to just copy over bvals and bvecs

            #path to bvecs
            in_path=os.path.join(self.cfg.PATHS.TESTING_PATH,self.subj,'diffusion',
                                    str(self.cfg.MODEL.NDIRS),'diffusion')
            bvecs_path=os.path.join(in_path,'bvecs')
            bvals_path=os.path.join(in_path,'bvals')
            mask_path=os.path.join(self.cfg.PATHS.TESTING_PATH,self.subj,'masks','mask.nii.gz')
            nodif_brain_mask_path=os.path.join(in_path,'nodif_brain_mask.nii.gz')

            #test_path=Path(os.path.join(self.model_path,'predictions',self.subj)) #we want to have a model folder in each subject path
            test_path=Path(self.save_path)
            test_path.mkdir(parents=True,exist_ok=True)

            shutil.copy(bvecs_path,os.path.join(test_path,'bvecs_network'))
            shutil.copy(bvals_path,os.path.join(test_path,'bvals_network'))
            shutil.copy(mask_path,os.path.join(test_path,'mask.nii.gz'))
            shutil.copy(nodif_brain_mask_path,os.path.join(test_path,'nodif_brain_mask.nii.gz'))

            diff_out=nib.Nifti1Image(diff_out,self.affine)
            nib.save(diff_out,test_path.joinpath('data_network.nii.gz'))

        else:
            signal=predictions.squeeze(-3) + self.data_module.data_list['Xflat']
            signal=signal*self.Xflatstd + self.Xflatmean

            h,w=signal.shape[-2:]
            signal=signal.reshape(-1,h,w)

            Ncore=N=100
            ico=icosahedron.icomesh(m=int(self.cfg.INPUT.H)-1)
            core=ico.core_basis
            I=ico.I_internal
            J=ico.J_internal
            N_random=N-2
            rng=np.random.default_rng()
            inds=rng.choice(N-2,size=N_random,replace=False)+1
            inds[0]=0


            bvals=np.zeros(N_random)
            x_bvecs=np.zeros(N_random)
            y_bvecs=np.zeros(N_random)
            z_bvecs=np.zeros(N_random)

            x_bvecs[1:]=ico.X_in_grid[core==1].flatten()[inds[1:]]
            y_bvecs[1:]=ico.Y_in_grid[core==1].flatten()[inds[1:]]
            z_bvecs[1:]=ico.Z_in_grid[core==1].flatten()[inds[1:]]

            bvals[1:]=1000

            diff_out=np.zeros(self.S0.shape + (N_random,))
            diff_out[:,:,:,0]=self.S0

            signal = signal[:,core==1]
            diff_out[self.xp,self.yp,self.zp,1:] = signal[:,inds[1:]]

            #test_path=Path(os.path.join(self.model_path,'predictions',self.subj))
            test_path=Path(self.save_path)
            test_path.mkdir(parents=True,exist_ok=True)

            diff_out=nib.Nifti1Image(diff_out,self.affine)
            nib.save(diff_out,test_path.joinpath('data_network.nii.gz'))

            #write the bvecs and bvals
            fbval = open(test_path.joinpath('bvals_network'), "w")
            for bval in bvals:
                fbval.write(str(bval)+" ")
            fbval.close()

            fbvecs = open(test_path.joinpath('bvecs_network'),"w")
            for x in x_bvecs:
                fbvecs.write(str(x)+ ' ')
            fbvecs.write('\n')
            for y in y_bvecs:
                fbvecs.write(str(y)+ ' ')
            fbvecs.write('\n')
            for z in z_bvecs:
                fbvecs.write(str(z)+ ' ')
            fbvecs.write('\n')
            fbvecs.close()

        mask=np.zeros_like(self.S0)
        mask[self.xp,self.yp,self.zp]=self.data_module.data_list['mask_train'].flatten()
        mask=nib.Nifti1Image(mask,self.affine)
        nib.save(diff_out,test_path.joinpath('nodif_brain_mask.nii.gz'))

        brain_mask_nii=nib.Nifti1Image(self.brain_mask(),self.affine)
        nib.save(brain_mask_nii,test_path.joinpath('mask.nii.gz'))