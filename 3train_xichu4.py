# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:48:29 2023

@author: Owner
"""


import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import os
import time
import json
import logging

from torchmeta.utils.data import BatchMetaDataLoader

from torch.utils.data import TensorDataset, DataLoader

from xichu import xichu
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
#from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
#,Normalize


from maml.model import ModelConvOmniglot

#from maml.model2 import resnet18
from maml.metalearners.maml_SF import ModelAgnosticMetaLearning
#from maml.metalearners.proto import ProtoMetaLearning
import ignite.distributed as idist

from shutil import copyfile
from utils import momentum_update
import random

from scipy import signal

import numpy as np
#import torch
#import cv2
#import random
from scipy.signal import resample
from PIL import Image
import scipy




def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    print('device:{}'.format(device))
        # folder = os.path.join(args.output_folder,
        #                       time.strftime('%Y-%m-%d_%H%M%S'))
        

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shots,
                                      num_test_per_class=args.num_shots_test)
    class_augmentations = args.Data_Augmentation
    class_augmentations1 = None
    transform =  Compose([args.Normalize])

    meta_train_dataset = xichu(args.folder, transform=transform,
        target_transform=Categorical(args.num_ways),
        num_classes_per_task=args.num_ways, meta_train=True,
        class_augmentations=class_augmentations,
        dataset_transform=dataset_transform, download=True)
    meta_val_dataset = xichu(args.folder,
        target_transform=Categorical(args.num_ways),
        num_classes_per_task=args.num_ways, meta_val=True,
        class_augmentations=class_augmentations1,
        dataset_transform=dataset_transform)

    model = ModelConvOmniglot(args.num_ways, hidden_size=args.hidden_size,feature_size=512)
    loss_function = F.cross_entropy

    #meta_train_dataloader = idist.auto_dataloader(meta_train_dataset,
          # batch_size=60, shuffle=True, num_workers=4,
          # pin_memory=False)
    
    #meta_val_dataloader = idist.auto_dataloader(meta_val_dataset,
           # batch_size=60, shuffle=True, num_workers=4,
          # pin_memory=False)



    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
        batch_size=args.batch_size,drop_last=True, shuffle=True, num_workers=args.num_workers,
        pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
        batch_size=args.batch_size, drop_last=True,shuffle=True, num_workers=args.num_workers,
        pin_memory=True)  #num_workers=8 有时管用有时不管用
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(model, meta_optimizer, dataloader=meta_train_dataloader,
        first_order=False, num_adaptation_steps=args.num_steps,
        step_size=args.step_size, loss_function=loss_function, device=device)

    best_value = None

    # Training loop
    fail_count = 0
    #out_test_L=[]
    #out_test_T=[]
    #Task_R=[]
    #Task_T=[]
    
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        if epoch<args.n_warmup:
            pem_eta = 1
        else:
            pem_eta = 0.9
        #print('progress evaluation eta={}'.format(pem_eta))
        metalearner.train(meta_train_dataloader, max_batches=args.num_batches, pem_eta=pem_eta,
                          verbose=args.verbose, desc='Training', leave=False)
        #a1=metalearner.train(meta_train_dataloader, max_batches=args.num_batches, pem_eta=pem_eta,
                          #verbose=args.verbose, desc='Training', leave=False)
        momentum_update(metalearner.model, metalearner.eval_model,
                        replace=True)
        results,test_logits,test_targets,b1 = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        #out_test_L.append(test_logits.detach().numpy())
        #out_test_T.append(test_targets.detach().numpy())
        '''
        if epoch<18 :
            Task_R.append(a1)
            Task_T.append(b1)
            output_Task_R = np.concatenate(Task_R, axis=0)
            output_Task_T = np.concatenate(Task_T, axis=0)
            np.save(os.path.join('D:\伪监督故障诊断\数据分析', 'output_Task_R.npy'),  output_Task_R, allow_pickle=False)
            np.save(os.path.join('D:\伪监督故障诊断\数据分析', 'output_Task_T.npy'),  output_Task_T, allow_pickle=False)
            
            '''
            

        # Save best model
        if (best_value is None) or (('accuracies_after' in results)
                and (best_value < results['accuracies_after'])):
            best_value = results['accuracies_after']
            save_model = True
        # elif (best_value is None) or (best_value > results['mean_outer_loss']):
        #     best_value = results['mean_outer_loss']
        #     save_model = True
        else:
            save_model = False


        if save_model:
            torch.save(model.state_dict(), 'E:\\研究数据\\西储大学\\xichu76lei_380\\best_model.pth')
            #output_array_L = np.concatenate(out_test_L, axis=0)
            #output_array_T = np.concatenate(out_test_T, axis=0)
            #np.save(os.path.join('D:\伪监督故障诊断\数据分析', 'out_test_L.npy'),  output_array_L, allow_pickle=False)
            #np.save(os.path.join('D:\伪监督故障诊断\数据分析', 'out_test_T.npy'),  output_array_T, allow_pickle=False)
            
            
            
        # if (epoch+1)%10==0:
        #     copyfile(args.model_path,'{}_e_{}.th'.format(args.model_path.split('.th')[0],epoch+1))
        if save_model:
            fail_count = 0
        else:
            fail_count+=1
            # print('fail counts = {}'.format(fail_count))
            # if fail_count >= 2:
            #     # init_fr *= fr_momentum
            #     fail_count = 0
        print('epoch = {}, best value = {}, fail counts = {}'.format(epoch, best_value, fail_count))


if __name__ == '__main__':
    import argparse
    

    class RandomAddGaussian(object):
        def __init__(self, sigma=0.01):
            self.sigma = sigma
    
        def __call__(self, seq):
            if np.random.randint(2):
                return seq
            else:
                seq=seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)
                return seq
    
    
    class RandomScale(object):
        def __init__(self, sigma=0.01):
            self.sigma = sigma
    
        def __call__(self, seq):
            if np.random.randint(2):
                return seq
            else:
                scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
                return seq*scale_factor
    
    
    class RATranslation(object):
        def __init__(self, p=0.5):
            self.p =p
        def __call__(self, seq):
            if np.random.randint(2):
                return seq
            else:
                a=len(seq)
                return np.concatenate((seq[int(a*self.p):],seq[0:int(a*self.p)]),axis=0)


    
    class Normalize(object):
        def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
            self.type = type
    
        def __call__(self, seq):
            if  self.type == "0-1":
                seq = (seq-seq.min())/(seq.max()-seq.min())
            elif  self.type == "1-1":
                seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
            elif self.type == "mean-std" :
                seq = (seq-seq.mean())/seq.std()
            else:
                raise NameError('This normalization is not included!')
    
            return seq

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('--folder', default='E:\\研究数据\西储大学\\xichu76lei_380\\')
    #E:\\研究数据\西储大学\\xichu76lei_380\\
    #D:\\伪监督故障诊断\数据分析\\250分类_448batch_numwork_20_loss0.92
    parser.add_argument('--num-ways', type=int, default=10,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=32,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=0,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', default=True,action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    # Progress Evaluation
    # parser.add_argument('--eval-beta', type=float, default=0.9,
    #                     help='beta is a parameter to control the momentum updating for the eval model'
    #                          '(default: 0.9).')
    parser.add_argument('--n-warmup', type=int, default=10,
                        help='the number of warm up')
    parser.add_argument('--Data-Augmentation',
         default=None
        )
    #[[RandomAddGaussian(0.02),RandomAddGaussian(0.08),RandomScale(0.05),RATranslation(0.8),RATranslation(0.5)]]
    parser.add_argument('--Normalize',
         default=Normalize('0-1'),
        )
    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
