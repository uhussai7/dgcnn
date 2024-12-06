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
                     strategy='ddp_find_unused_parameters_true',callbacks=[TimingCallback(),MemoryLoggerCallback()],
                     max_epochs=cfg.TRAIN.MAX_EPOCHS)
model=DNet(cfg)
trainer.fit(model,datamodule=data_module)

