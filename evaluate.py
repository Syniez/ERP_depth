import torch.optim as optim
import numpy as np
import argparse
import datetime
import torch
import cv2

from torch.utils.data import DataLoader, Dataset
from Dataloaders.dataset_loader_360d import Dataset
from DPT.dpt.models import DPTDepthModel
from losses import MSELoss, AngleLoss
from tqdm import tqdm



def main(args):
    input_dir = args.rootpath
    val_file = args.testfile
    batch_size = 1

    test_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = True, flip = True, root_path=input_dir, path_to_img_list=val_file),
                                                batch_size=batch_size, shuffle=False, num_workers=8, drop_last = False, pin_memory=True)


    Net = DPTDepthModel(path = None, backbone='vitb_rn50_384', non_negative=True, enable_attention_hooks=False).cuda()
    optimizer = optim.AdamW(list(Net.parameters()), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)

    idx = 0

    for batch, (rgb, depth, mask) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            Net.eval()
            
            rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
            pred = Net(rgb)

            print(idx)
            cv2.imwrite("./results/" + str(idx).zfill(5) + ".png", pred.squeeze().cpu().numpy())
            idx += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='structure3d')
    parser.add_argument('--rootpath', type=str, default='/data/rvi/dataset/')
    parser.add_argument('--testfile', type=str, default='./filenames/test_structure3d.txt')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/DPT_base_0.pth')
    args = parser.parse_args()

    main(args)