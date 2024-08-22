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
_C.PATHS.MODELS_PATH=os.environ.get('DGCNN_MODELS_PATH')

#Model
_C.MODEL.THREE_FILTERS=[9,384,384,384]
_C.MODEL.THREE_ACTIVATIONS=[64,64,64,64,1]

#Train parameters
_C.TRAIN.MAX_EPOCHS=10
_C.TRAIN.BATCH_SIZE=20

#Input
_C.INPUT.NDIRS=6
_C.INPUT.NSCALARS=1
_C.INPUT.NSHELLS=1
_C.INPUT.H=5
_C.INPUT.N_C=16
_C.INPUT.N_SUBJECTS=15

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()