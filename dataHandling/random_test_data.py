
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import diffusion
from configs.config import get_cfg_defaults


def make_diffusion(in_path,sub,sub_dir):
    print('Preparing diffusion data for subject:', str(sub))
    #try:
    diff=diffusion.diffVolume(in_path + sub + '/T1w/Diffusion/')
    diff.shells()
    diff.randomBvecs(sub_dir+'/diffusion/') #has cutpad flat default to true
    # except:
    #     print('Error encountered while loading diffusion data, likely missing directions!')
    print('Done with diffusion data for subject:', str(sub))


cfg=get_cfg_defaults()
diff_path_source=cfg.PATHS.TESTING_PATH
diff_path_target=cfg.PATHS.TESTING_RANDOM_PATH
in_path=cfg.PATHS.RAW_PATH

subjects=[102816]#os.listdir(diff_path_source)


print('source dir is ', in_path)
print('will save in ',diff_path_target)
response = input("Proceed? (yes/no): ").strip().lower()

if response != "yes":
    print("Exiting program.")
    exit()

for sub in subjects:
    make_diffusion(in_path,str(sub),diff_path_target+ '/' + str(sub))
