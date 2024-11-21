import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import training_data
import os
import torch
import numpy as np
import h5py
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict


class PreprocDataset(Dataset):
    def __init__(self, data_list, interp_matrices=None):
        self.data_list = data_list
        self.interp_matrices = interp_matrices

        if interp_matrices is not None:
            self._getitem = self._getitem_with_matrices
        else:
            self._getitem = self._getitem_without_matrices

    def __len__(self):
        return len(self.data_list[list(self.data_list.keys())[0]])

    def __getitem__(self, idx):
        return self._getitem(idx)

    # def _getitem_with_matrices(self, idx):
    #     sample = {key: value[idx] for key, value in self.data_list.items()}
    #     interp_index = sample['interp_inds']
    #     print(sample['interp_inds'])
    #     sample['interp_matrix'] = self.interp_matrices[int(interp_index.item())]
    #     return sample

    def _getitem_with_matrices(self, idx):
        #out=[self.data_list[key][idx] for key in self.data_list.keys()]
        interp_index = self.data_list['interp_inds'][idx]
        #print(sample['interp_inds'])
        interp_matrix = self.interp_matrices[int(interp_index.item())]
        return [self.data_list['X'][idx], self.data_list['Y'][idx], 
                self.data_list['mask_train'][idx], self.data_list['Xflat'][idx],
                interp_matrix]


    def _getitem_without_matrices(self, idx):
        #return {key: value[idx] for key, value in self.data_list.items()} #should this really be returning a dictionary?
        return [self.data_list[key][idx] for key in self.data_list.keys()] #should this really be returning a dictionary?
    
class PreprocData(LightningDataModule):
    def __init__(self, cfg, batch_size=1, num_workers=4, subj=None): #subj needs to be provided for testing
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.N_subjects = cfg.INPUT.NSUBJECTS
        self.data_list = None
        self.interp_matrices = None
        self.subj=subj

    def prepare_data(self):
        # Implement any pre-download or preprocessing steps if needed
        pass

    def setup(self, stage):
        if stage == 'fit':
            # Load training data
            in_path = self.cfg.PATHS.TRAINING_PREPROC_PATH
            self.data_list, self.interp_matrices = self._load_data(in_path, testing=False)
            #print(self.data_list)

        elif stage == 'predict':
            # Load testing data
            in_path = self.cfg.PATHS.TESTING_PREPROC_PATH
            self.data_list, self.interp_matrices = self._load_data(in_path, testing=True)

    def _load_data(self, in_path, testing):
        subjects = os.listdir(in_path)
        data_list = []
        interp_matrices = []
        if testing==False:
            print('Loading subjects from path: ', str(in_path))
            for s in range(self.N_subjects):
                subj = subjects[s]
                print(f'Loading subject {subj}')
                with h5py.File(os.path.join(in_path, subj, 'data.h5'), 'r') as f:
                    if self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.DEPTH_3D == 0:
                        data_list.append(self.load_2d_only(f, testing))
                    elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D == 0:
                        data_list.append(self.load_3d_only(f, testing))
                    elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.REVERSE == True:
                        data_list.append(self.load_mixed_r(f, s, testing))
                        interp_matrices.append(f['interp_matrix'][:])
                    elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.REVERSE == False:
                        data_list.append(self.load_mixed(f,testing))
                    else:
                        raise Exception('Invalid config')
        else:
            print('Loading subjects from path: ', str(in_path))
            subj=self.subj
            print(f'Loading subject {subj}')
            self.data_file=os.path.join(in_path, subj, 'data.h5')
            with h5py.File(self.data_file, 'r') as f:
                if self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.DEPTH_3D == 0:
                    data_list.append(self.load_2d_only(f, testing))
                elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D == 0:
                    data_list.append(self.load_3d_only(f, testing))
                elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.REVERSE == True:
                    data_list.append(self.load_mixed_r(f, 0, testing))
                    interp_matrices.append(f['interp_matrix'][:])
                elif self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.REVERSE == False:
                    data_list.append(self.load_mixed(f,testing))
                else:
                    raise Exception('Invalid config')

        return self.dic_cat(data_list), interp_matrices

    #this is done in the following manner so we only load the data that is needed
    def load_mixed(self,f,testing):
        if testing is False:
            keys=['Xflat','Y', 'mask_train']
            out_dic = OrderedDict((name, torch.tensor(f[name][:])) for name in keys)
            out_dic['t1t2']=torch.tensor(f['X'][:])[:,:2]
            return out_dic
        else:
            keys=['Xflat','Y', 'mask_train']
            out_dic= OrderedDict((name, torch.tensor(f[name][:])) for name in keys)
            out_dic['t1t2']=torch.tensor(f['X'][:])[:,:2]
            return out_dic

    def load_2d_only(self, f, testing):
        if testing is False:
            keys=['Xflat', 'Y', 'mask_train']
            return OrderedDict((name, torch.tensor(f[name][:])) for name in keys)
        else:
            keys = ['Xflat','Y', 'mask_train']#['S0X', 'Xflat', 'S0Y', 'Y', 'mask_train',]
            return OrderedDict((name, torch.tensor(f[name][:])) for name in keys)

    def load_3d_only(self, f, testing):
        #keys = ['S0X', 'X', 'S0Y', 'Ybase', 'mask_train'] if testing else ['S0X', 'X', 'S0Y', 'Ybase', 'mask_train']
        keys = ['X', 'Ybase', 'mask_train'] if testing else [ 'X', 'Ybase', 'mask_train']
        return OrderedDict((name, torch.tensor(f[name][:])) for name in keys)

    def load_mixed_r(self, f, s, testing):
        keys = ['S0X', 'X', 'Xflat', 'interp_matrix', 'S0Y', 'Y', 'mask_train'] if testing else ['X', 'Y', 'mask_train', 'Xflat']
        this_data = OrderedDict((name, torch.tensor(f[name][:])) for name in keys)
        interp_inds = torch.zeros(this_data['X'].shape[0])
        interp_inds[:] = s
        this_data['interp_inds'] = interp_inds
        return this_data

    def dic_cat(self, dict_list):
        keys = dict_list[0].keys()
        out_d = {key: [] for key in keys}
        for d in dict_list:
            for key in keys:
                out_d[key].append(d[key])
        for key in keys:
            out_d[key] = torch.cat(out_d[key], dim=0)
        return out_d

    def train_dataloader(self):
        dataset = PreprocDataset(self.data_list, self.interp_matrices if self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.REVERSE == True else None)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # Define the validation DataLoader if applicable
        pass

    def predict_dataloader(self):
        # The setup method ensures data is loaded correctly for testing
        dataset = PreprocDataset(self.data_list, self.interp_matrices if self.cfg.MODEL.DEPTH_3D > 0 and self.cfg.MODEL.DEPTH_2D > 0 and self.cfg.MODEL.REVERSE == True else None)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage=None):
        # Clean up any resources after training or testing
        pass

# class preprocData:
#     def __init__(self,cfg,testing=False): 
#         self.cfg=cfg
#         self.N_subjects=self.cfg.INPUT.NSUBJECTS
#         self.testing=testing

#     def load_2d_only(self,f): #f is the file, s is the subject ind
#         if self.testing:
#             this_data={name: torch.tensor(f[name][:]) for name in ['S0X','Xflat', 'S0Y' ,'Y', 'mask_train']} #change these keys as needed
#         else:
#             this_data={name: torch.tensor(f[name][:]) for name in ['Xflat','Y', 'mask_train']} #change these keys as needed
#         return this_data
    
#     def load_3d_only(self,f):
#         if self.testing:
#             this_data={name: torch.tensor(f[name][:]) for name in ['S0X','X', 'S0Y' ,'Ybase', 'mask_train']} #change these keys as needed
#         else:
#             this_data={name: torch.tensor(f[name][:]) for name in ['S0X','X', 'S0Y' ,'Ybase', 'mask_train']} #change these keys as needed
#         return this_data

#     def load_mixed(self,f,s):
#         if self.testing:
#             this_data={name: torch.tensor(f[name][:]) for name in ['S0X','X','Xflat', 'S0Y' ,'Y', 'mask_train']} #change these keys as needed
#         else:
#             this_data={name: torch.tensor(f[name][:]) for name in ['X','Xflat','Y', 'mask_train']} #change these keys as needed
#         interp_inds=torch.zeros(this_data['Xflat'].shape[0])
#         interp_inds[:]=s
#         this_data['interp_inds']=interp_inds
#         return this_data
    
#     def dic_cat(self,dict_list):
#         keys=dict_list[0].keys()
#         out_d={key:[] for key in keys}
#         for d in dict_list:
#             for key in keys:  
#                 out_d[key].append(d[key])
        
#         for key in keys:
#             out_d[key]=torch.cat(out_d[key],dim=0)
        
#         return out_d

#     def load_preproc(self): #this is based on the config
#         #we need some logic here to load the data based on the model we are using
#         if self.testing==True: #this is for testing/prediction mode
#             in_path=self.cfg.PATHS.TESTING_PREPROC_PATH
#         else:
#             in_path=self.cfg.PATHS.TRAINING_PREPROC_PATH

#         subjects=os.listdir(in_path)

#         data_list=[]
#         interp_matrices=[]
#         for s in range(0,self.N_subjects):
#             subj=subjects[s]
#             print('Loading subject %s'%subj)
#             if self.cfg.MODEL.DEPTH_2D>0 and  self.cfg.MODEL.DEPTH_3D==0: #2d only
#                 with h5py.File(str(in_path) + subj +'/data.h5','r') as f:
#                     data_list.append(self.load_2d_only(f))

#             elif self.cfg.MODEL.DEPTH_3D>0 and  self.cfg.MODEL.DEPTH_2D==0: #3d only
#                 with h5py.File(str(in_path) + subj +'/data.h5','r') as f:
#                     data_list.append(self.load_3d_only(f))
            
#             elif self.cfg.MODEL.DEPTH_3D>0 and  self.cfg.MODEL.DEPTH_2D>0: #mixed
#                 with h5py.File(str(in_path) + subj +'/data.h5','r') as f:
#                     data_list.append(self.load_mixed(f,s))
#                     interp_matrices.append(f['interp_matrix'][:])
        
#         return self.dic_cat(data_list), interp_matrices
        

        

def data_grab(N_subjects,subs_path,b_dirs=6,H=5,Nc=16,N_patch=500):
    """
    This function will grab all the relevant data
    :param N_subjects: Number of subjects
    :param subs_path: Path for subjects
    """

    subjects = os.listdir(subs_path) #get subjects
    if N_subjects > len(subjects): #check if enough subjects available
        raise ValueError('Number of subjects requested is greater than those available')

    #initialize data lists, these will be contactenated as tensors later
    X=[]
    Y=[]
    S0Y=[]
    Xflat =[]
    S0X=[]
    mask_train= []
    interp_matrix = []
    interp_matrix_ind = []

    #loop through all the subjects
    for sub in range(0,N_subjects):
        print('Loading data from subject: ',subjects[sub])
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

        X.append(this_subject.X) #this is T1,T2, and bdirs diffusion data
        S0Y.append(this_subject.Y[0]) #dtifit S0 for labels
        Y.append(this_subject.Y[1]) #icosahedron signal from dtifit for labels
        mask_train.append(this_subject.mask_train) #training mask
        interp_matrix.append(torch.from_numpy(np.asarray(this_subject.diff_input.interpolation_matrices))) #interpolation matrices
        Xflat.append(this_subject.Xflat) #diffusion data projected on icosahedron for inputs
        S0X.append(this_subject.S0X) #Raw S0 from input diffusion data

        #this is to track with subjects interpolation matrices to use
        this_interp_matrix_inds = torch.ones([this_subject.X.shape[0]])
        this_interp_matrix_inds[this_interp_matrix_inds==1]=sub
        interp_matrix_ind.append(this_interp_matrix_inds)

    X = torch.cat(X)
    Y=torch.cat(Y)
    S0Y = torch.cat(S0Y)
    mask_train = torch.cat(mask_train)
    interp_matrix = torch.cat(interp_matrix)
    interp_matrix_ind=torch.cat(interp_matrix_ind).int()
    Xflat = torch.cat(Xflat)
    S0X = torch.cat(S0X)

    return X, Xflat, S0X, Y, S0Y, mask_train, interp_matrix, interp_matrix_ind


def data_grab_save_preproc(sub,subs_path,preproc_path,b_dirs=6,H=5,Nc=16,N_patch=1000):
    #save preprocessed data as h5 files

    print('Loading data from subject: ',sub)

    diffusion_with_bdirs_path = subs_path + '/' + sub + '/diffusion/' + str(b_dirs)  + '/diffusion/'
    dti_with_bdirs_path       = subs_path + '/' + sub + '/diffusion/' + str(b_dirs)  + '/dtifit/dtifit'
    mask_for_bdirs_file       = subs_path + '/' + sub + '/diffusion/' + str(b_dirs)  + '/diffusion/nodif_brain_mask.nii.gz'
    dti_with_90dirs_path      = subs_path + '/' + sub + '/diffusion/' + '90/'        + '/dtifit/dtifit'
    mask_for_training_file    = subs_path + '/' + sub + '/masks/mask.nii.gz'  
    t1t2_path                 = subs_path + '/' + sub + '/structural/'
    this_subject =  training_data(diffusion_with_bdirs_path,
                                    dti_with_bdirs_path,
                                    dti_with_90dirs_path,
                                    mask_for_bdirs_file,
                                    t1t2_path,
                                    mask_for_training_file,
                                    H, 
                                    N_train=N_patch,
                                    Nc=Nc)
    



    # X.append(this_subject.X) #this is T1,T2, and bdirs diffusion data
    # S0Y.append(this_subject.Y[0]) #dtifit S0 for labels
    # Y.append(this_subject.Y[1]) #icosahedron signal from dtifit for labels
    # mask_train.append(this_subject.mask_train) #training mask
    # interp_matrix.append(torch.from_numpy(np.asarray(this_subject.diff_input.interpolation_matrices))) #interpolation matrices
    # Xflat.append(this_subject.Xflat) #diffusion data projected on icosahedron for inputs
    # S0X.append(this_subject.S0X) #Raw S0 from input diffusion data

    out_path=Path(preproc_path + '/%s/'%sub)
    out_path.mkdir(parents=True,exist_ok=True)

    print('saving in ' + str(out_path) + '/data.h5')

    with h5py.File(str(out_path) + '/data.h5','w') as f:
        f.create_dataset('X',                   data=this_subject.X.numpy())
        f.create_dataset('Xflat',               data=this_subject.Xflat.numpy())
        f.create_dataset('S0X',                 data=this_subject.S0X.numpy())
        f.create_dataset('Y',                   data=this_subject.Y[1].numpy())
        f.create_dataset('S0Y',                 data=this_subject.Y[0].numpy())
        f.create_dataset('S0_Ybase',            data=this_subject.Y_base[:,0].numpy())
        f.create_dataset('Ybase',               data=this_subject.Y_base[:,1:].numpy())
        f.create_dataset('mask_train',          data=this_subject.mask_train.numpy())
        f.create_dataset('interp_matrix',       data=torch.from_numpy(np.asarray(this_subject.diff_input.interpolation_matrices)))
        f.create_dataset('Xmean',               data=this_subject.Xmean)
        f.create_dataset('Xstd',                data=this_subject.Xstd)
        f.create_dataset('Xflatmean',           data=this_subject.Xflatmean)
        f.create_dataset('Xflatstd',            data=this_subject.Xflatstd)
        f.create_dataset('S0Xmean',             data=this_subject.S0Xmean)
        f.create_dataset('S0Xstd',              data=this_subject.S0Xstd)
        f.create_dataset('xp',                  data=this_subject.xp)
        f.create_dataset('yp',                  data=this_subject.yp)
        f.create_dataset('zp',                  data=this_subject.zp)
        f.create_dataset('S0',                  data=this_subject.diff_input.vol.get_fdata()[:,:,:,this_subject.diff_input.inds[0]].mean(-1))


    



