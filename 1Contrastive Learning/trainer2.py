import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import TensorDataset, DataLoader

#from module import MimgnetEncoder, CelebrEncoder,NetCNN


from DRSN_CW import BasicBlock,RSNet

from data_NEW import CWRUDataset

#from data import MimgNetDataset
from utils import NTXentLoss


import Cluster_balance
#import visdom
#vis = visdom.Visdom(env='SIMCLR111')


class Trainer(object):
    def __init__(self, args):
        self.args = args
    
        # dataset = MimgNetDataset(os.path.join(self.args.data_dir), mode='train', simclr=True)
        if self.args.dataset == "CWRU":
            dataset = CWRUDataset(os.path.join(self.args.data_dir), mode='train', simclr=True)
           # h_size = 64
            #layers = 6
            #sample_len = 1024
            #feat_size = (sample_len//2**layers)*h_size
            #feat_size = 256 WDCNN时用
    
            #self.encoder = NetCNN(output_size=1024, hidden_size=h_size, layers=layers,
                                 #channels=1, embedding_size=feat_size).to(args.device)
            self.encoder = RSNet(BasicBlock, [2, 2, 2, 2]).to(args.device)
            
       

        self.trloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=5,
            pin_memory=True
        )

        last_hidden_size = 2*2*args.hidden_size
        self.l1 = nn.Linear(last_hidden_size, last_hidden_size).to(args.device)
        self.l2 = nn.Linear(last_hidden_size, int(0.25*last_hidden_size)).to(args.device)
        self.l3 = nn.Linear(last_hidden_size, last_hidden_size).to(args.device)
        self.l4 = nn.Linear(last_hidden_size, 256).to(args.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+list(self.l1.parameters())+list(self.l2.parameters()),
            lr=args.lr
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=len(self.trloader), 
            eta_min=0,
            last_epoch=-1
        )

        self.criterion = NTXentLoss(
            device=args.device, 
            batch_size=args.batch_size, 
            temperature=0.1, 
            use_cosine_similarity=True
        )
        self.writer = SummaryWriter('log')
        self.softmax=nn.Softmax(dim=1)
    def train(self):
        global_step = 0
        best_loss = 1000000.0

        for global_epoch in range(self.args.train_epochs):
            avg_loss = []
            with tqdm(total=len(self.trloader)) as pbar:
                for xis, xjs,_ in self.trloader:
                    self.encoder.train()
                    self.l1.train()
                    self.l2.train()
                    self.l3.train()
                    self.l4.train()

                    self.encoder.zero_grad()
                    self.l1.zero_grad()
                    self.l2.zero_grad()
                    self.l3.zero_grad()
                    self.l4.zero_grad()

                    xis = xis.to(self.args.device)
                    xjs = xjs.to(self.args.device)

                    zis = self.l2(F.relu(self.l1(self.encoder(xis))))
                    zjs = self.l2(F.relu(self.l1(self.encoder(xjs))))
                    yis = self.softmax(self.l4(F.relu(self.l3(self.encoder(xis)))))
                    yjs = self.softmax(self.l4(F.relu(self.l3(self.encoder(xjs)))))


                    zis = F.normalize(zis, dim=1)
                    zjs = F.normalize(zjs, dim=1)
                    
                    
                    criterion_cluster=Cluster_balance.ClusterLoss(256, 1, self.args.device).to(self.args.device)
                    loss_cluster=criterion_cluster(yis,yjs)
                    

                    loss_in = self.criterion(zis, zjs)  
                    loss=loss_in+loss_cluster
                    loss.backward()          
                    self.optimizer.step()

                    postfix = OrderedDict(
                        {'loss': '{0:.4f}'.format(loss)}
                    )
                    pbar.set_postfix(**postfix)

                    pbar.update(1)
                    global_step += 1
                    avg_loss.append(loss.item())
                    

                    if self.args.debug:
                        break

            if global_epoch >= 10:
                self.scheduler.step()
            
            avg_loss = np.mean(avg_loss)
            self.writer.add_scalars(
                        'train', 
                        {'loss': avg_loss}, 
                        global_epoch)
            state = {
                'encoder_state_dict': self.encoder.state_dict(),
                'l1_state_dict': self.l1.state_dict(),
                'l2_state_dict': self.l2.state_dict(),
                'l3_state_dict': self.l3.state_dict(),
                'l4_state_dict': self.l4.state_dict()
            }
            torch.save(state, os.path.join(self.args.save_dir, 'best1.pth'))

            print("{0}-th EPOCH Loss: {1:.4f}".format(global_epoch, avg_loss))
            #vis.line(Y=[avg_loss], X=[global_epoch],update=None if global_epoch == 0 else 'append', win='simCLR')

            if self.args.debug:
                break
        
        self.save_feature()

    def save_feature(self):
        self.encoder.eval()
        os.makedirs(self.args.feature_save_dir, exist_ok=True)
        
        for mode in ['train', 'val']:
            dataset = CWRUDataset(os.path.join(self.args.data_dir), mode=mode)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                shuffle=False
            )            
            features = []
            xs=[]
            ys=[]
            print("START ENCODING {} SET".format(mode))
            with tqdm(total=len(loader)) as pbar:
                for x,y in loader:
                    x = x.to(self.args.device)
                    f = self.encoder(x)
                    features.append(f.detach().cpu().numpy())
                    xs.append(x.reshape(-1,1024).detach().cpu().numpy())
                    ys.append(y.detach().cpu().numpy())
                    pbar.update(1)
            features = np.concatenate(features, axis=0)
            xs=np.concatenate(xs, axis=0)
            ys=np.concatenate(ys, axis=0)
            
            
            print("SAVE ({0}, {1}) shape array".format(features.shape[0], features.shape[1]))
            np.save(os.path.join(self.args.feature_save_dir, "{}_features.npy".format(mode)), features)
            print("SAVE ({0}, {1}) shape array".format(xs.shape[0], xs.shape[1]))
            np.save(os.path.join(self.args.feature_save_dir, "{}_xs.npy".format(mode)), xs)
            print("SAVE ({0}, {1}) shape array".format(xs.shape[0], xs.shape[1]))
            np.save(os.path.join(self.args.feature_save_dir, "{}_ys.npy".format(mode)), ys)
            print("SAVE ({0}) shape array".format(ys.shape[0]))
            
