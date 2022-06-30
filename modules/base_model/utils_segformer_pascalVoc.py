import os 
import time 
import torch 
import torch.nn as nn 
import torch.nn.functional as F





def load_checkpoint_mit(model, path_checkpoint):
    '''
    Args: model
    '''
    state_dict_checkpoint = torch.load(path_checkpoint)
    
    model.load_state_dict(state_dict_checkpoint, strict=False)
    ckpt_keys = set(state_dict_checkpoint.keys())
    own_keys = set(model.state_dict().keys())
    print(own_keys)
    # print(own_keys)
    #print("own keys",len(own_keys))
    #print("checkpoint keys", len(ckpt_keys))
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    
    print("Missing key: ", len(missing_keys))
    print("Missing key: ", missing_keys)
    
    print("Unexpected key: ", len(unexpected_keys))
    print("Unexpected keys: ",unexpected_keys )

    del state_dict_checkpoint

    return model