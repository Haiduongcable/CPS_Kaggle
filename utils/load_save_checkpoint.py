import torch 
import numpy as np 
import os 
import time 
from config import config
from collections import OrderedDict
from utils.pyt_utils import load_model, load_model_cpu

def save_checkpoint(model, optimizer_l, optimizer_r, epoch ):
    '''
    params: model: pytorch model (branch 1 branch 2)
    params: optimizer_l: optimizer for left branch
    params: optimizer_r: optimizer for right branch 
    params: epoch: number of epoch to continue
    
    (Create directory) and save state dict checkpoint
    '''
    path_dir = config.path_save_checkpoint
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    path_save = path_dir + "/" + "checkpoint_epoch_" + str(epoch) + ".pth"
    
    print("Saving checkpoint to file {}".format(path_save))
    
    t_start = time.time()

    state_dict = {}

    
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        key = k
        if k.split('.')[0] == 'module':
            key = k[7:]
        new_state_dict[key] = v
    state_dict['model'] = new_state_dict
    
    if optimizer_l is not None:
        state_dict['optimizer_l'] = optimizer_l.state_dict()
    if optimizer_r is not None:
        state_dict['optimizer_r'] = optimizer_r.state_dict()
    state_dict['epoch'] = epoch

    torch.save(state_dict, path_save)
    del state_dict
    del new_state_dict
    print("Saved checkpoint")
    
def save_bestcheckpoint(model, optimizer_l, optimizer_r ):
    '''
    params: model: pytorch model (branch 1 branch 2)
    params: optimizer_l: optimizer for left branch
    params: optimizer_r: optimizer for right branch 
    params: epoch: number of epoch to continue
    
    (Create directory) and save state dict checkpoint
    '''
    path_dir = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/CPS_Kaggle/medical_weight"
    path_save = path_dir + "/" + "bestcheckpoint.pth"
    
    print("Saving checkpoint to file {}".format(path_save))
    
    t_start = time.time()

    state_dict = {}

    
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        key = k
        if k.split('.')[0] == 'module':
            key = k[7:]
        new_state_dict[key] = v
    state_dict['model'] = new_state_dict
    
    if optimizer_l is not None:
        state_dict['optimizer_l'] = optimizer_l.state_dict()
    if optimizer_r is not None:
        state_dict['optimizer_r'] = optimizer_r.state_dict()

    torch.save(state_dict, path_save)
    del state_dict
    del new_state_dict
    print("Saved checkpoint")

def load_checkpoint(path_checkpoint, network, optimizer_l, optimizer_r, epoch):
    '''
    params: path_checkpoint: path to load checkpoint
    params: model: pytorch model (branch 1 branch 2)
    params: optimizer_l: optimizer for left branch
    params: optimizer_r: optimizer for right branch 
    params: epoch: number of epoch to continue
    
    Restore checkpoint from pth file 
    return: model, optimizer_l, optimizer_r
    '''
    state_dict = torch.load(path_checkpoint)

    model = load_model(network, state_dict['model'], False)
    if 'optimizer_l' in state_dict:
       optimizer_l.load_state_dict(state_dict['optimizer_l'])
    if 'optimizer_r' in state_dict:
        optimizer_r.load_state_dict(state_dict['optimizer_r'])
    epoch = state_dict['epoch'] + 1
    del state_dict
    print("Load checkpoint from file {}".format(path_checkpoint))
    return model , optimizer_l, optimizer_r, epoch


def load_checkpoint_model(path_checkpoint, network):
    '''
    params: path_checkpoint: path to load checkpoint
    params: model: pytorch model (branch 1 branch 2)
    params: optimizer_l: optimizer for left branch
    params: optimizer_r: optimizer for right branch 
    params: epoch: number of epoch to continue
    
    Restore checkpoint from pth file 
    return: model, optimizer_l, optimizer_r
    '''
    state_dict = torch.load(path_checkpoint, map_location="cpu")

    model = load_model_cpu(network, state_dict['model'], False)
    del state_dict
    print("Load checkpoint from file {}".format(path_checkpoint))
    return model