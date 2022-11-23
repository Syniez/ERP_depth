import torch.optim as optim
import numpy as np
import argparse
import datetime
import torch
import wandb
import os

from torch.utils.data import DataLoader, Dataset
from Dataloaders.dataset_loader_360d import Dataset
from DPT.dpt.models import DPTDepthModel
from losses import MSELoss, AngleLoss
#from utils import validation
from tqdm import tqdm

wandb_available = True


def main(args):
    input_dir = args.rootpath
    train_file = args.trainfile
    val_file = args.valfile
    batch_size = args.bs
    epochs = args.epoch
    validate_every = args.validate

    train_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = True, flip = True, root_path=input_dir, path_to_img_list=train_file),
                                                batch_size=batch_size, shuffle=True, num_workers=8, drop_last = True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = True, flip = True, root_path=input_dir, path_to_img_list=val_file),
                                                batch_size=batch_size, shuffle=False, num_workers=8, drop_last = False, pin_memory=True)


    Net = DPTDepthModel(path = None, backbone='vitb_rn50_384', non_negative=True, enable_attention_hooks=False).cuda()
    optimizer = optim.AdamW(list(Net.parameters()), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)
    
    criterion_depth = MSELoss()
    criterion_ssim = AngleLoss()


    for epoch in range(epochs):
        total_depth_loss = 0
        Net.train()

        for batch, (rgb, depth, mask) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            
            rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
            pred = Net(rgb)
            
            l_depth = criterion_depth(pred, depth, mask)

            loss = l_depth
            total_depth_loss += loss.item()

            #if batch % validate_every == 0 and batch > 0:
            #    val_score = validate(Net, val_loader)

            if batch % 50 == 0 and batch > 0:
                print('[Epoch %d--Iter %d]depth loss %.4f' % (epoch, batch, total_depth_loss/(batch+1)))
                if wandb_available:
                    wandb.log({"Train/{MES_loss}": loss})

            loss.backward()
            optimizer.step()
            scheduler.step()

        torch.save(Net.state_dict(), './checkpoints/DPT_base_' + str(epoch) + '.pth')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='structure3d')
    parser.add_argument('--rootpath', type=str, default='/data/rvi/dataset/')
    parser.add_argument('--trainfile', type=str, default='./filenames/train_structure3d.txt')
    parser.add_argument('--valfile', type=str, default='./filenames/test_structure3d.txt')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--validate', type=int, default=10000)
    args = parser.parse_args()

    if wandb_available:
        wandb.init(project="ERP_Depth", entity="syniez", reinit=True)
        wandb.run.name = datetime.datetime.now().strftime("%m-%d-%H:%M")
        wandb.config.update(args)
        wandb.run.save()

    main(args)