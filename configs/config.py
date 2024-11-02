from yacs.config import CfgNode as CN
import os

#Declare some nodes
_C = CN()
_C.SYSTEM = CN()
_C.PATHS = CN()
_C.MODEL= CN()
_C.TRAIN=CN()
_C.INPUT=CN()

#Paths
_C.PATHS.TRAINING_PATH=os.environ.get('DGCNN_TRAIN_PATH')
_C.PATHS.TESTING_PATH=os.environ.get('DGCNN_TEST_PATH')
_C.PATHS.TESTING_ROTATED_PATH=os.environ.get('DGCNN_TEST_ROTATED_PATH')
_C.PATHS.TRAINING_PREPROC_PATH=os.environ.get('DGCNN_TRAIN_PREPROC_PATH')
_C.PATHS.TESTING_PREPROC_PATH=os.environ.get('DGCNN_TEST_PREPROC_PATH')
_C.PATHS.TESTING_ROTATED_PREPROC_PATH=os.environ.get('DGCNN_TEST_ROTATED_PREPROC_PATH')
_C.PATHS.MODELS_PATH=os.environ.get('DGCNN_MODELS_PATH')
_C.PATHS.CONFIG_PATH=os.environ.get('DGCNN_CONFIG_PATH')
_C.PATHS.SCRATCH=os.environ.get('DGCNN_SCRATCH_PATH')

#Model
_C.MODEL.NAME='default'
_C.MODEL.NDIRS=6
_C.MODEL.WIDTH_2D=64
_C.MODEL.WIDTH_3D=64
_C.MODEL.DEPTH_2D=3
_C.MODEL.DEPTH_3D=3
_C.MODEL.REVERSE=False #setting this to true will put 3d convolutions first

#Training
_C.TRAIN.LR=1e-4
_C.TRAIN.FACTOR=0.5
_C.TRAIN.PATIENCE=7
_C.TRAIN.MAX_EPOCHS=20
_C.TRAIN.BATCH_SIZE=1

#Input
_C.INPUT.NDIRS=6
_C.INPUT.NSCALARS=1
_C.INPUT.NSHELLS=1
_C.INPUT.H=5
_C.INPUT.NC=16
_C.INPUT.NSUBJECTS=5
_C.INPUT.NCORE=100

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()