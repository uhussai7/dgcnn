from dataGrab import data_grab
from configs.config import get_cfg_defaults

cfg=get_cfg_defaults()
cfg.INPUT.N_SUBJECTS=1
X, Xflat, S0X, Y, S0Y, mask_train, interp_matrix, interp_matrix_ind=data_grab(cfg.INPUT.N_SUBJECTS,
                                                                              cfg.PATHS.TRAINING_PATH)