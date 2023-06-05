
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

import argparse
import os
import pandas as pd
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from our_models import TrainingModel


import torch 
from torchvision.datasets.utils import download_url

class Args(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

args = json.load(open('args.json','r'))
config=json.load(open('config.json','r'))
args = Args(args) # dict2object

    
from pytorch_lightning.callbacks import ModelCheckpoint

print("Creating captioning dataset")
train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)  


samplers = [None, None, None]

train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                      batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                      is_trains=[True, False, False], collate_fns=[None,None,None])         


if __name__=='__main__':
    # %%time
    args.result_dir = os.path.join(args.output_dir, 'result')
    #     print(config)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path(config['ann_root']).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
#     if config['load_checkpoint']:
# #         train_model = TrainingModel.load_from_checkpoint(config['checkpoint'],config=config,hparams_file=config['hparams_path'])
#         train_model = TrainingModel.load_from_checkpoint(config['checkpoint'],config=config)
# #         train_model = TrainingModel(config)
#     else:
#         train_model = TrainingModel(config)
# #     train_model = TrainingModel.load_from_checkpoint('checkpoints/model-epoch=04-val_loss=0.03.ckpt',config=config)

    train_model = TrainingModel(config)

    # training

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='max'
    )
    logger=CSVLogger(save_dir="logs/")
    

    trainer = pl.Trainer(
        default_root_dir="checkpoints/",
    #     resume_from_checkpoint='checkpoints/model-epoch=00-val_loss=1.65.ckpt',

        accelerator='auto', 
        devices='auto', 
#         precision=16,
        callbacks=[checkpoint_callback,TQDMProgressBar(refresh_rate=10)],
        max_epochs=config['max_epoch'], 
        logger=logger,
        strategy='auto'
    )
    print("Start training")
#     trainer.fit(train_model, train_loader,val_loader,ckpt_path='checkpoints/model-epoch=00-val_loss=0.00.ckpt')
    if args.evaluate:
        trainer.validate(train_model, val_loader)
        # trainer.validate(train_model, val_loader,ckpt_path=config['checkpoint'])
    else:
        trainer.fit(train_model, train_loader,val_loader)
        # trainer.save_checkpoint(filepath="checkpoints/current_checkpoint.ckpt")
        torch.save({'model':train_model.model.state_dict()},'checkpoints/captioner.pth')
