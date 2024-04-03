import os
import argparse
import torch
from trainer2 import Trainer
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimCLR pretrainig')

    # Directory Argument
    parser.add_argument('--data_dir', type=str, default="E:\\研究数据\\西储大学\\xichu76lei_380")
    parser.add_argument('--save_dir', type=str, default="D:\\无监督因果推导下的故障诊断方法\\simclr")
    parser.add_argument('--feature_save_dir', type=str, default="D:\\无监督因果推导下的故障诊断方法\\simclr")
    parser.add_argument('--dataset', type=str, default="CWRU")

    # Model Argument
    parser.add_argument('--hidden-size', type=int, default=128)

    # Training Argument
    parser.add_argument('--batch-size', type=int, default=448)
    parser.add_argument('--train-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-3)

    # System Argument
    parser.add_argument('--debug', action='store_true')
    #parser.add_argument('--gpu-id', type=int, default=0)

    args = parser.parse_args()

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    args.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter('log')

    trainer = Trainer(args)
    trainer.train()
