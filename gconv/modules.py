import torch
from .gPyTorch import opool
from torch.nn.modules.module import Module
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d
from .gPyTorch import gNetFromList
import pickle
from torch.nn import InstanceNorm3d
from torch.nn import Conv3d
from torch.nn import ModuleList
from torch.nn import DataParallel
from torch.nn import Linear
from torch.nn import Sequential
from .icosahedron import sphere_to_flat_basis
import os
import lightning as L
from torch.nn.functional import mse_loss
from collections import OrderedDict
from torch.nn import functional as F
from . import icosahedron
from . import dihedral12 as d12
from pathlib import Path
import timeit

# Timing callback
class TimingCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []  # List to store epoch times

    def on_train_epoch_start(self, trainer, pl_module):
        # Start the timer at the beginning of each epoch
        self.epoch_start_time = timeit.default_timer()

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        # End the timer at the end of each epoch and calculate the elapsed time
        epoch_end_time = timeit.default_timer()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)  # Store the epoch time

        # Log the epoch time as a metric
        trainer.logger.log_metrics({'epoch_time': epoch_time}, step=trainer.current_epoch)

        # Optionally print the epoch time
        print(f"Epoch {trainer.current_epoch + 1} time: {epoch_time:.4f} seconds")

    def on_train_end(self, *args, **kwargs):
        # Optionally log the average epoch time at the end of training
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        print(f"Average epoch time: {avg_epoch_time:.4f} seconds")


class DgcnnTrainer(L.Trainer):
    """
    Class for training the encoder
    """
    def __init__(self,cfg,*args,**kwargs): #we do per subject
        self.cfg=cfg
        self.out_path=self._create_dir_name(cfg)
        super().__init__(*args,
                        default_root_dir=self.out_path,**kwargs)
        self.write_cfg_yaml()

    def _create_dir_name(self,cfg):
        out_path=Path(os.path.join(cfg.PATHS.MODELS_PATH,cfg.MODEL.NAME,
                                   'Nsubs-%d'%int(cfg.INPUT.NSUBJECTS)))
        out_path.mkdir(parents=True,exist_ok=True)
        return str(out_path)

    def write_cfg_yaml(self):
        with open(str(self.out_path)+'/config.yaml', 'w') as f:
            f.write(self.cfg.dump())

class DNet(L.LightningModule):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cfg=cfg
        self.model=self._get_model(*args,**kwargs)

    def _get_model(self,*args,**kwargs):
        if self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.DEPTH_3D == 0:
            print("Using 2d only model")
            return Lit2dOnly(self.cfg,*args,**kwargs)
        elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D == 0:
            print("Using 3d only model")
            return Lit3dOnly(self.cfg,*args,**kwargs)
        elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0:
            print("Using mixed model")
            return LitMixed(self.cfg,*args,**kwargs)

    def configure_optimizers(self,*args,**kwargs):
        return self.model.configure_optimizers(*args,**kwargs)

    def predict_step(self,*args,**kwargs):
        return self.model.predict_step(*args,**kwargs)

    def training_step(self,*args,**kwargs):
        # return self.model.training_step(*args,**kwargs)
        loss= self.model.training_step(*args,**kwargs)
        self.log("train_loss", loss)
        return loss

    def forward(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)
    

class LitMixed(L.LightningModule):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.cfg=cfg
        self.Ndirs=cfg.MODEL.NDIRS

        #define the 3d block
        C_3d=int(self.cfg.MODEL.WIDTH_3D) #has to be some multiple of Ndirs because we use Ndirs of the channels to project them down
        depth_3d=int(self.cfg.MODEL.DEPTH_3D)
        filterlist3d=[9] #3d layers, 9 = 2 (T1,T2) + 7 (S0 + 6 diffusion directions)
        [filterlist3d.append(C_3d) for i in range(0, depth_3d)]
        activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)] #3d layers activatiions

        #define the 2d block
        C_2d=int(self.cfg.MODEL.WIDTH_2D) #should be combatible be 3d block
        depth_2d=int(self.cfg.MODEL.DEPTH_2D)
        filterlist2d=[C_2d] 
        [filterlist2d.append(C_2d) for i in range(0, depth_2d)]
        filterlist2d[-1]=self.cfg.INPUT.NSHELLS
        activationlist2d = [F.relu for i in range(0,len(filterlist2d)-1)] #3d layers activatiions
        activationlist2d[-1]=None

        #get the icosahedron
        ico=icosahedron(m=int(cfg.INPUT.H)-1)
        I, J, T = d12.padding_basis(H=cfg.INPUT.H) 

        self.model = residualnetScalars(filterlist3d,
                                        activationlist3d,
                                        filterlist2d,
                                        activationlist2d,
                                        cfg.INPUT.H,
                                        cfg.INPUT.NSHELLS,
                                        cfg.INPUT.NC,
                                        I,
                                        J,
                                        cfg.INPUT.NSCALARS,
                                        cfg.INPUT.NDIRS,
                                        ico)

    def training_step(self,batch, batch_idx):
        x,y,w,mask=batch #we need mask too, so different data loaders are needed
        y_hat=self.model(x,w)
        loss=mse_loss(y[mask==1,:,:],y_hat[mask==1,:,:])
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.model.parameters(), lr=self.cfg.TRAIN.LR)  # , weight_decay=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.cfg.TRAIN.FACTOR,
                                      patience=self.cfg.TRAIN.PATIENCE,
                                      verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'train_loss'  # Replace 'val_loss' with the metric you want to monitor
                                }
                }

    def forward(self,x):
        return self.model(x)

class Lit2dOnly(L.LightningModule): #only the 2d conv
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.cfg=cfg
        self.Ndirs=cfg.MODEL.NDIRS

        #if self.cfg.MODEL.DEPTH_3D<1:
        C=int(self.cfg.MODEL.WIDTH_2D)
        depth_2d=int(self.cfg.MODEL.DEPTH_2D)
        filterlist2d=[1] 
        [filterlist2d.append(C) for i in range(0, depth_2d)]
        filterlist2d[-1]=self.cfg.INPUT.NSHELLS
        activationlist2d = [F.relu for i in range(0,len(filterlist2d)-1)] #3d layers activatiions
        activationlist2d[-1]=None

        self.model=Sequential(OrderedDict(
                    [
                        ('move2batch',in2d()),
                        ('gconv',gnet3d(self.cfg.INPUT.H,filterlist2d,self.cfg.INPUT.NSHELLS,activationlist2d)),
                        ('move2batch_inv',out2d(self.cfg.TRAIN.BATCH_SIZE,self.cfg.INPUT.NC,self.cfg.INPUT.NSHELLS))
                    ]
                   ))

    def training_step(self,batch, batch_idx):
        x,y,mask=batch#batch['Xflat'],batch['Y'],batch['mask_train'] #we need mask too, so different data loaders are needed
        x,y=x.unsqueeze(-3),y.unsqueeze(-3)
        y_hat=self.model(x)
        loss=mse_loss(y[mask==1,:,:],y_hat[mask==1,:,:])
        #self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x,y,mask=batch#batch['Xflat'],batch['Y'],batch['mask_train'] #we need mask too, so different data loaders are needed
        x,y=x.unsqueeze(-3),y.unsqueeze(-3)
        y_hat=self.model(x)
        return y_hat



    def configure_optimizers(self):
        optimizer = optim.Adamax(self.model.parameters(), lr=self.cfg.TRAIN.LR)  # , weight_decay=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.cfg.TRAIN.FACTOR,
                                      patience=self.cfg.TRAIN.PATIENCE,
                                      verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'train_loss'  # Replace 'val_loss' with the metric you want to monitor
                                }
                }

    def forward(self,x):
        return self.model(x)
         

class Lit3dOnly(L.LightningModule): #only takes 3d convs
    def __init__(self,cfg,*args,**kwargs):        
        super().__init__(*args,**kwargs)  
        
        self.cfg=cfg
        self.Ndirs=cfg.MODEL.NDIRS

        #if self.cfg.MODEL.DEPTH_2D<1:
        C=int(self.cfg.MODEL.WIDTH_3D)
        depth_3d=int(self.cfg.MODEL.DEPTH_3D)
        filterlist3d=[9] #3d layers, 9 = 2 (T1,T2) + 7 (S0 + 6 diffusion directions)
        [filterlist3d.append(C) for i in range(0, depth_3d)]
        filterlist3d[-1]=self.Ndirs
        activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)] #3d layers activatiions
        activationlist3d[-1]=None
        self.model =conv3dList(filterlist3d, activationlist3d)

    def training_step(self, batch, batch_idx):
        x,y,mask=batch #we need mask too, so different data loaders are needed
        y_hat=self.model(x)
        loss=mse_loss(y[mask==1],y_hat[mask==1])
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.model.parameters(), lr=self.cfg.TRAIN.LR)  # , weight_decay=0.001)
        optimizer.zero_grad()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.cfg.TRAIN.FACTOR,
                                      patience=self.cfg.TRAIN.PATIENCE,
                                      verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'train_loss'  # Replace 'val_loss' with the metric you want to monitor
                                }
                }

    def forward(self,x):
        return self.model(x)


class to_icosahedron(Module):
    """
    This module moves from N-directions to the 2d icosahedron
    """
    def __init__(self,ico_mesh,subject_interp_matrices):
        #subject_interp_matrices is the subject specific interpolation matrix [sub_id,...]
        super(to_icosahedron,self).__init__()
        self.icomesh=ico_mesh
        self.subject_interp_matrices=subject_interp_matrices

    def forward(self,x,sub_id):
        #x will have shape [batch,C,Nc,Nc,Nc]
        B=x.shape[0]
        C=x.shape[1]
        N=x.shape[2]
        x=x.moveaxis(1,-1)
        x=x.view([-1,C])
        
class in3d(Module):
    """
    Here we are just moving the 2d space into the batch dimension
    """
    def __init__(self,core):
        super(in3d, self).__init__()
        self.core = core

    def forward(self,x):
        #x will have shape [batch,Nc,Nc,Nc,C,h,w]
        B = x.shape[0]
        Nc = x.shape[1]
        C = x.shape[-3]
        x=x[:,:,:,:,:,self.core==1]
        Ncore = x.shape[-1]
        x = x.moveaxis((-1,-2),(1,2))
        x=x.contiguous()
        x = x.view([B*Ncore,C,Nc,Nc,Nc])
        return x

class out3d(Module):
    """
    Inverse of in3d
    """
    def __init__(self,B,Nc,Ncore,core_inv,I,J,zeros):
        super(out3d, self).__init__()
        self.B = B
        self.Nc = Nc
        self.Ncore = Ncore
        self.core_inv = core_inv
        self.I = I
        self.J = J
        self.zeros = zeros

    def forward(self,x):
        #x will have shape [B*Ncore, C, Nc, Nc, Nc]
        C = x.shape[1]
        x = x.view(self.B,self.Ncore,C,self.Nc,self.Nc,self.Nc)
        x = x[:, self.core_inv, :, :, :, :]  # shape is [B,h,w,C,Nc,Nc,Nc])
        x = x[:, self.I, self.J, :, :, :, :] #padding
        x = x.moveaxis((1, 2, 3), (-2, -1, -3))
        x[:, :, :, :, :, self.zeros == 1] = 0 #zeros

        return x

class in2d(Module):
    """
    Moving 3d dimensions to batch dimension
    """
    def __init__(self):
        super(in2d, self).__init__()

    def forward(self,x):
        #x has shape [batch, Nc, Nc, Nc, C, h, w]
        B = x.shape[0]
        Nc = x.shape[1]
        C = x.shape[-3]
        h = x.shape[-2]
        w = x.shape[-1]
        x = x.view((B*Nc*Nc*Nc,C,h,w)) #patch voxels go in as a batch
        return x

class out2d(Module):
    """
    inverse of in2d
    """
    def __init__(self,B,Nc,Cout):
        super(out2d, self).__init__()
        self.B = B
        self.Nc = Nc
        self.Cout= Cout

    def forward(self,x):
        h=x.shape[-2]
        w=x.shape[-1]
        x = x.view([self.B,self.Nc,self.Nc,self.Nc,self.Cout,h,w])
        return x

class gnet3d(Module): #this module (layer) takes in a 3d patch but only convolves in internal space
    def __init__(self,H,filterlist,shells=None,activationlist=None): #shells should be same as filterlist[0]
        super(gnet3d,self).__init__()
        self.filterlist = filterlist
        self.gconvs=gNetFromList(H,filterlist,shells,activationlist= activationlist)
        self.pool = opool(filterlist[-1])

    def forward(self,x):
        x = self.gconvs(x) #usual g conv
        x = self.pool(x) # output shape is [batch*Nc^3,filterlist[-1],h,w]
        return x

#this module takes in scalar and 3d input with shapes [B Nc^3,Cinp,Nscalar] and [B Nc^3,Cinp Nshell,h,w]
class gnet3dScalar(Module):
    def __init__(self,H,LCinp,LCout,filterlist,shells=None,activationlist=None): #shells should be same as filterlist[0]
        super(gnet3dScalar,self).__init__()

        self.LCinp = LCinp
        self.LCout = LCout
        self.filterlist = filterlist
        self.gconvs=gNetFromList(H,filterlist,self.LCinp*shells,activationlist= activationlist)
        self.pool = opool(filterlist[-1])

        self.lin = Linear(LCinp,LCout)


    def forward(self,x):
        #linear part
        x0 = x[0] #assuming x0 has shape [BNc^3,Cinp,Nscalar]
        x0 = x0.moveaxis(1,-1) #shape is [BNc^3,Nscalar, Cin]
        x0 = self.lin(x0)

        #gConv2d part
        x = x[1]
        x = self.gconvs(x) #usual g conv
        x = self.pool(x)
        return x0,x

#this is a class to make lists out of
class conv3d(Module):
    """
    This class combines conv3d and batch norm layers and applies a provided activation
    """
    def __init__(self,Cin,Cout,activation=None,norm=True):
        super(conv3d,self).__init__()
        self.activation= activation
        self.conv = Conv3d(Cin,Cout,3,padding=1)
        self.norm = InstanceNorm3d(Cout)
        self.batch_norm = norm

    def forward(self,x):
        if self.activation!=None:
            x=self.activation(self.conv(x))
        else:
            x=self.conv(x)

        if self.batch_norm:
            x = self.norm(x)

        return x

#this makes the list for 3d convs
class conv3dList(Module):
    def __init__(self,filterlist,activationlist=None):
        super(conv3dList,self).__init__()
        self.conv3ds=[]
        self.filterlist = filterlist
        self.activationlist = activationlist
        if activationlist is None:
            self.activationlist = [None for i in range(0,len(filterlist)-1)]
        for i in range(0,len(filterlist)-1):
            if i==0:
                self.conv3ds=[conv3d(filterlist[i],filterlist[i+1],self.activationlist[i])]
            else:
                norm = True
                if i == len(filterlist) - 2:
                    norm = False
                self.conv3ds.append(conv3d(filterlist[i],filterlist[i+1],self.activationlist[i],norm=norm))
        self.conv3ds = ModuleList(self.conv3ds)

    def forward(self,x):
        for i,conv in enumerate(self.conv3ds):
            x = conv(x)
        return x

class threeTotwo(Module):
    def __init__(self,Nshells,Nscalar,Ndir,ico,I,J):
        super(threeTotwo,self).__init__()
        self.ico = ico
        self.Nshells = Nshells
        self.Nscalar = Nscalar
        self.Ndir = Ndir
        self.I = I
        self.J = J


    def forward(self,x,w):
        #x will have have shape [B,Cin,Nc,Nc,Nc]
        B=x.shape[0]
        Cin = x.shape[1]
        Nc = x.shape[-1]
        Nscalar = self.Nscalar
        Ndir = self.Ndir
        Nshells = self.Nshells

        #we assume that x is flattened as [B,Cin=Cinp(Nscalar + NshellNdir),Nc,Nc,Nc]
        x = x.moveaxis(1,-1)#shape is now [B,Nc,Nc,Nc,Cin]
        x = x.view([-1,Cin])

        Cinp = int(Cin / (Nscalar + Nshells * Ndir))
        x = x.view([x.shape[0],Cinp,Nscalar + Nshells*Ndir])
        x_0 = x[:,:,0:Nscalar] #this is the scalar part
        x = x[:,:,Nscalar:] #this is the part to project to 2d has shape [x.shape[0],Cinp,Nshells*Ndir]
        x = x.view([x.shape[0],Cinp,Nshells,Ndir])
        #per shell matrix multiplication w should have shape [Nshells,N2d,2Ndir]
        N2d = w.shape[1]
        x_out = torch.empty([x.shape[0],Cinp,Nshells,N2d]).to(x.device.type).float()
        for m in range(0,Nshells):
            x_out[:,:,m,:] = torch.matmul(torch.cat([x[:,:,m,:],
                                                     x[:,:,m,:]],-1).reshape(x.shape[0],1,Cinp,2*Ndir),
                                                     w[m,:,:].to(x.device.type).T).view([x.shape[0],Cinp,N2d]).float()

            x_out[:,:,m,:] = 0.5*(x_out[:,:,m,:]+x_out[:,:,m,self.ico.antipodals])

        #move to icosahedron
        del x
        basis = sphere_to_flat_basis(self.ico)
        h = basis.shape[-2]
        w = basis.shape[-1]
        out = torch.empty([x_out.shape[0],Cinp,Nshells,h,w])

        i_nan, j_nan = np.where(np.isnan(basis))
        basis[i_nan, j_nan] = 0
        basis = basis.astype(int)

        out = x_out[:,:,:,basis]
        del x_out
        out = out[:,:,:,self.I[0,:,:],self.J[0,:,:]]
        out =out.view([out.shape[0],Cinp*Nshells,h,w])

        return x_0,out

class twoToThree(Module):
    def __init__(self,Nc,Nshell,Nscalar):
        super(twoToThree,self).__init__()
        self.Nc= Nc
        self.Nshell =Nshell
        self.Nscalar = Nscalar

    def forward(self,x):
        x0 = x[0]
        x = x[1]


        if (self.Nshell ==1 and self.Nscalar==1):
            B=int(x0.shape[0]/self.Nc**3)
            x0 = x0.view([B,self.Nc,self.Nc,self.Nc]) # only works with Nshell =1 and Nscalar=1
        else:
            x0 = x0.view([self.B, self.Nc, self.Nc, self.Nc,self.Nshell,self.Nscalar])  # only works with Nshell =1 and
            # Nscalar=1

        B = int(x.shape[0] / self.Nc ** 3)
        Cout = x.shape[1]
        h=x.shape[-2]
        w = x.shape[-1]

        if(Cout==1):
            x = x.view([B, self.Nc, self.Nc, self.Nc, h, w])
        else:
            x = x.view([B,self.Nc,self.Nc,self.Nc,Cout,h,w])

        return x0,x
    

class residualnetScalars(Module):
    def __init__(self, filterlist3d, activationlist3d, filterlist2d, activationlist2d, H, Nshells, Nc,I, J,Nscalar,Ndir,ico):
        super(residualnetScalars, self).__init__()
        # params
        self.Nshells = Nshells
        self.Nscalar = Nscalar
        self.Ndir = Ndir
        self.Cin = filterlist3d[-1]
        self.Cinp = int(self.Cin / (Nscalar + Nshells * Ndir))
        print('Cin is',self.Cin)
        print('Nscalar is',Nscalar)
        print('Nshells',Nshells)
        print('Ndir',Ndir)
        print('cINP IS', self.Cinp)
        self.ico = ico

        self.flist3d = filterlist3d
        self.alist3d = activationlist3d
        self.flist2d = filterlist2d
        self.alist2d = activationlist2d

        self.three2t = threeTotwo(Nshells,Nscalar,Ndir,ico,I,J)
        self.two2t = twoToThree(Nc,Nshells,Nscalar)

        self.conv3ds = conv3dList(filterlist3d, activationlist3d)
        self.gconvs = gnet3dScalar(H,self.Cinp,Nscalar, filterlist2d, Nshells, activationlist2d)

        self.conv3ds = DataParallel(self.conv3ds)
        self.gconvs = DataParallel(self.gconvs)

    def forward(self,x,w):
        x= self.conv3ds(x)
        x= self.three2t(x,w)
        x= self.gconvs(x)
        x= self.two2t(x)
        return x
    



#here we should build the models but we have to keep things simple,

#- we have two modules 3d and 2d.
#-- we have to do just 3d and just 2d.
#-- we have to do varying 3d and 2d depth, width


#we will take instructions from the config file and build the model

#we will have 3d layers, 2d layers and mapping between them.
# the only 3d and only 2d are just seperate to begin with

#The one that starts from 2d is different since we have the projected data.
#-- we can keep it general and project from 3d with the interp matrix (have to c confirm these both were created the same way: Xflat is with interpolation matrices, Y is with ico signal from dti)
#-- okay so we can just go from the 3d all the time and use the interp matrices to project
#-- this will help us keep inputs the same for extreme 2d and extreme 3d
#-- only issue is with pure 3d output, here the the data the will be sent is different (how is it? where are the base models files? Will have to recreate this )
# -- -- Since we have to change the outputs for the base model, better to just change inputs for the 3d only model also
# -- so have 3, 2d, 3d, mixed
# -- mixed: 3d first 2d after, vice versa
# -- also have to decide what is the width
# -- lets keep total depth fixed since we know this is what is needed for super res
# -- so params are: 1) width, 2) 3d/2d depth, and 3) swap (the swap is tricky)
# -- I think we should stick to 3d first and argue that the 2d is what is upsampling (this is the best option)


#if 3d is on then we project otherwise we don't?
