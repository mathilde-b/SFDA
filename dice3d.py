#!/usr/env/bin python3.6
import numpy
import io
import re
import random
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union
from binary import hd, ravd, hd95, hd_var, assd, asd
import torch
import numpy as np
from torch import Tensor
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from utils import id_, map_, class2one_hot, resize_im, soft_size
from utils import simplex, sset, one_hot, dice_batch
from argparse import Namespace
import os
import pandas as pd
import imageio

def dice3d(all_grp,all_inter_card,all_card_gt,all_card_pred,all_pred,all_gt,all_pnames,metric_axis,pprint=False,do_hd=0,do_asd=0,best_epoch_val=0):
    _,C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    list(filter(lambda a: a != 0.0, unique_patients))
    unique_patients = [u.item() for u in unique_patients]
    batch_dice = torch.zeros((len(unique_patients), C))
    batch_hd = torch.zeros((len(unique_patients), C))
    batch_asd = torch.zeros((len(unique_patients), C))
    if do_hd>0 or do_asd>0:
        all_pred = all_pred.cpu().numpy()
        all_gt = all_gt.cpu().numpy()
    # do DICE
    for i, p in enumerate(unique_patients):
        inter_card_p = torch.einsum("bc->c", [torch.masked_select(all_inter_card, all_grp == p).reshape((-1, C))])
        card_gt_p= torch.einsum("bc->c", [torch.masked_select(all_card_gt, all_grp == p).reshape((-1, C))])
        card_pred_p= torch.einsum("bc->c", [torch.masked_select(all_card_pred, all_grp == p).reshape((-1, C))])
        dice_3d = (2 * inter_card_p + 1e-8) / ((card_pred_p + card_gt_p)+ 1e-8)
        batch_dice[i,...] = dice_3d
        if pprint:
            print(p,dice_3d.cpu())
    indices = torch.tensor(metric_axis)
    dice_3d = torch.index_select(batch_dice, 1, indices)
    dice_3d_mean = dice_3d.mean(dim=0)
    dice_3d_mean = torch.round(dice_3d_mean * 10**4) / (10**4)
    dice_3d_sd = dice_3d.std(dim=0)
    dice_3d_sd = torch.round(dice_3d_sd * 10**4) / (10**4)

    # do HD and / or ASD
    #if dice_3d_mean.mean()>best_epoch_val:
    if dice_3d_mean.mean()>0:
        for i, p in enumerate(unique_patients):
            root_name = [re.split('(\d+)', x.item())[0] for x in all_pnames][0]
            bool_p = [int(re.split('_',re.split(root_name,x.item())[1])[0])==p for x in all_pnames]
            slices_p = all_pnames[bool_p]
            #if do_hd >0 or dice_3d_mean.mean()>best_epoch_val:
            if do_hd> 0 or do_asd >0 :
                all_gt_p = all_gt[bool_p,:]
                all_pred_p = all_pred[bool_p,:]
                sn_p = [int(re.split('_',x)[1]) for x in slices_p]
                ord_p = np.argsort(sn_p)
                label_gt = all_gt_p[ord_p,...]
                label_pred = all_pred_p[ord_p,...]
                asd_3d_var_vec = [None] * C
                hd_3d_var_vec= [None] * C
                for j in range(0,C):
                    label_pred_c = numpy.copy(label_pred)
                    label_pred_c[label_pred_c!=j]=0
                    label_pred_c[label_pred_c==j]=1
                    label_gt_c = numpy.copy(label_gt)
                    label_gt_c[label_gt!=j]=0
                    label_gt_c[label_gt==j]=1
                    if len(np.unique(label_pred_c))>1: # len(np.unique(label_gt_c))>1 should always be true...
                        if do_hd>0:
                            if root_name=="Subj_":
                                hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[1.25,1.25,2]).item() #in mm
                            else:
                                hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c).item() #in voxel
                        if do_asd > 0:
                            asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item() #ASD IN VOXEL TO COMPARE TO PNP PAPER
                    else:
                        hd_3d_var_vec[j]=np.NaN
                        asd_3d_var_vec[j] = np.NaN
            if do_asd>0:
                asd_3d_var = torch.from_numpy(np.asarray(asd_3d_var_vec))  # np.nanmean(hd_3d_var_vec)
                batch_asd[i,...] = asd_3d_var

            if do_hd>0:
                hd_3d_var = torch.from_numpy(np.asarray(hd_3d_var_vec))  # np.nanmean(hd_3d_var_vec)
                batch_hd[i,...] = hd_3d_var

    [hd_3d, hd_3d_sd] = get_mean_sd(batch_hd,indices)
    [asd_3d, asd_3d_sd] = get_mean_sd(batch_asd,indices)
    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean.cpu().numpy(), dice_3d_sd.cpu().numpy()])
    if pprint:
        print('asd_3d_mean',asd_3d, "asd_3d_sd", asd_3d_sd, "hd_3d_mean", hd_3d, "hd_3d_sd", hd_3d_sd,"dice 3d",dice_3d.item())
    [return_asd,return_asd_sd] = [asd_3d.item(),asd_3d_sd.item()] if do_asd >0 else [0,0]
    [return_hd,return_hd_sd] = [hd_3d.item(),hd_3d_sd.item()] if do_hd >0 else [0,0]
    return dice_3d.item(), dice_3d_sd.item(), return_asd, return_asd_sd,return_hd,return_hd_sd


def get_mean_sd(x,indices):
    x_ind = torch.index_select(x, 1, indices)
    x_mean = x_ind.mean(dim=0)
    x_mean = torch.round(x_mean * 10**4) / (10**4)
    x_std = x_ind.std(dim=0)
    x_std = torch.round(x_std * 10**4) / (10**4)
    x_mean, x_std= map_(lambda t: t.mean(), [x_mean,x_std])
    return x_mean,x_std
