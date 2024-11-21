#the aim here is to to test the 2d model on the rotated data
#we can generate new testdata with rotated bvecs

#we need to make a copy of all the subjects and then rotate the bvecs and rewrite them
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config import get_cfg_defaults
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

cfg=get_cfg_defaults()

def read_bvecs(file_path):
    f = open(file_path, "r")
    bvecs=[]
    for line_ in f.readlines():#this is read lines
        line_=line_.split(' ')[:-1]
        #print(line_)
        line=[float(l.strip()) for l in line_]
        #[print(l) for l in line_]
        #print(line)
        bvecs.append((np.asarray(line)))
    f.close()
    return np.asarray(bvecs)

def rotate_bvecs(bvecs, angles, seq='zyx'):
    return R.from_euler(seq,angles,degrees=True).as_matrix()@bvecs
    
def write_bvecs(bvecs,file_path):
    #print(bvecs)
    #print(file_path)
    fbvecs = open(file_path, "w")
    for x in bvecs[0]:
        fbvecs.write(str(x)+ ' ')
    fbvecs.write('\n')
    for y in bvecs[1]:
        fbvecs.write(str(y)+ ' ')
    fbvecs.write('\n')
    for z in bvecs[2]:
        fbvecs.write(str(z)+ ' ')
    fbvecs.write('\n')
    fbvecs.close()


#source path
diff_path_source=cfg.PATHS.TESTING_PATH
diff_path_target=cfg.PATHS.TESTING_ROTATED_PATH

print(diff_path_source)
#set dubjects list
subjects=os.listdir(diff_path_source)
dirs=[6,90]
for sub in subjects:
    a,b,c=random.randint(0,90),random.randint(0,90),random.randint(0,90)
    for dir in dirs:
        print('------------------------------------------------------------')
        print(sub)
        print('Ndirs',dir)

        
        #get the orginal
        subj_path_source=os.path.join(diff_path_source,sub,'diffusion',str(dir),'diffusion','bvecs')
        bvecs_orig=read_bvecs(subj_path_source)

        print('Source',subj_path_source)
        
        #rotate
        bvecs_rotated=rotate_bvecs(bvecs_orig,[a,b,c])
        
        #set target path
        subj_path_target=os.path.join(diff_path_target,sub,'diffusion',str(dir),'diffusion','bvecs')
        write_bvecs(bvecs_rotated,subj_path_target)

        print('Target',subj_path_target)


        #save original
        subj_path_target=os.path.join(diff_path_target,sub,'diffusion',str(dir),'diffusion','bvecs_orig')
        write_bvecs(bvecs_orig,subj_path_target)

        print('------------------------------------------------------------')

#read the six direction

