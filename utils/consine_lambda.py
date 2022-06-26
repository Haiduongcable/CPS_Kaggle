from black import main
import numpy as np 
import time 
import os 
from tqdm import tqdm 
import math 


def consine_lambda(epoch):
    '''
    Step: 0.1 -> 0.5: 5
          0.5 -> 1.0: 5
          1.0 -> 1.5: 10
          
    Args: epoch: current epoch 
    Return: lambda
    '''
    lambda_loss = [0.1 + 0.08*i for i in range(5)] + \
                [0.5 + 0.1*i for i in range(5)] + \
                [1.0 + 0.05*i for i in range(10)] + \
                [1.5] * 30
    return lambda_loss[epoch]
if __name__ == '__main__':
    consine_lambda(0)
    