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
from bounds import ConstantBounds,TagBounds, PreciseBounds
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


def dice3d(all_grp,all_inter_card,all_card_gt,all_card_pred,all_pred,all_gt,all_pnames,metric_axis,pprint=False,do_hd=0,best_epoch_val=0):
    _,C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    unique_patients = unique_patients[unique_patients != torch.ones_like(unique_patients)*666]
    unique_patients = [u.item() for u in unique_patients] 
    batch_dice = torch.zeros((len(unique_patients), C))
    batch_hd = torch.zeros((len(unique_patients), C))
    batch_asd = torch.zeros((len(unique_patients), C))
    if do_hd>0:
        all_pred = all_pred.cpu().numpy()
        all_gt = all_gt.cpu().numpy()
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

    # do HD
    #if dice_3d_mean.mean()>best_epoch_val:
    if dice_3d_mean.mean()>0:
        for i, p in enumerate(unique_patients):
            try:
                bool_p = [int(re.split('_',re.split('Case',x.item())[1])[0])==p for x in all_pnames]
                #bool_p = [int(re.split('_',re.split('slice',x.item())[1])[0])==p for x in all_pnames]
                data = "saml"
            except:
                #bool_p = [int(re.split('_',re.split('Subj_',x.item())[1])[0])==p for x in all_pnames]
                data = "ivd"
            slices_p = all_pnames[bool_p]
            if do_hd >0 :
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
                    if len(np.unique(label_pred_c))>1 and len(np.unique(label_gt_c))>1:
                        if data=="saml":
                            asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item() 
                            hd_3d_var_vec[j] = asd_3d_var_vec[j]
                        elif data=="ivd":
                            hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[1.25,1.25,2]).item()
                            asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c,[1.25,1.25,2]).item() 
                    else:
                        hd_3d_var_vec[j]=np.NaN
                        asd_3d_var_vec[j] = np.NaN
                hd_3d_var=torch.from_numpy(np.asarray(hd_3d_var_vec))# np.nanmean(hd_3d_var_vec)
                asd_3d_var=torch.from_numpy(np.asarray(asd_3d_var_vec))# np.nanmean(hd_3d_var_vec)
            if pprint and do_hd:
                print(p,asd_3d_var)
            if do_hd>0:
                batch_hd[i,...] = hd_3d_var
                batch_asd[i,...] = asd_3d_var

    hd_3d = torch.index_select(batch_hd, 1, indices)
    asd_3d = torch.index_select(batch_asd, 1, indices)
    hd_3d_mean = hd_3d.mean(dim=0)
    #asd_3d_mean = asd_3d.mean(dim=0)
    asd_3d_mean = torch.from_numpy(np.nanmean(asd_3d, axis=0))
    hd_3d_mean = torch.round(hd_3d_mean * 10**4) / (10**4)
    asd_3d_mean = torch.round(asd_3d_mean * 10**4) / (10**4)
    hd_3d_sd = hd_3d.std(dim=0)
    asd_3d_sd = asd_3d.std(dim=0)
    hd_3d_sd = torch.round(hd_3d_sd * 10**4) / (10**4)
    asd_3d_sd = torch.round(asd_3d_sd * 10**4) / (10**4)
    asd_3d_mean = asd_3d_mean.cpu().numpy()

    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean, dice_3d_sd])
    [hd_3d, hd_3d_sd] = map_(lambda t: t.mean(), [hd_3d_mean, hd_3d_sd])
    [asd_3d, asd_3d_sd] = map_(lambda t: t.mean(), [asd_3d_mean, asd_3d_sd])
    if pprint:
        print('asd_3d_mean',asd_3d_mean.mean(), "asd_3d_sd", asd_3d_sd, "hd_3d_mean", hd_3d_mean, "hd_3d_sd", hd_3d_sd)
        print("hd_3d_mean", hd_3d_mean, "hd_3d_sd", hd_3d_sd)
    return dice_3d.item(), dice_3d_sd.item(), asd_3d.item(), asd_3d_sd.item()

def run_dices(args: Namespace) -> None:

    for folder in args.folders:
        subfolders = args.subfolders
        all_dices=[0] * len(subfolders)
        for i, subfolder in enumerate(subfolders):
            print(subfolder)
            epc = int(subfolder.split('r')[1])
            dice_i = dice3d(args.base_folder, folder, subfolder, args.grp_regex, args.gt_folder)
            all_dices[epc] = dice_i

        df = pd.DataFrame({"3d_dice": all_dices})
        df.to_csv(Path(args.save_folder, 'dice_3d.csv'), float_format="%.4f", index_label="epoch")

def metrics_calc(all_grp,all_inter_card,all_card_gt,all_card_pred, metric_axis,pprint=False):
    _, C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    batch_dice = torch.zeros((len(unique_patients), C))
    batch_avd = torch.zeros((len(unique_patients), C))
    for i, p in enumerate(unique_patients):
        inter_card_p = torch.einsum("bc->c", [torch.masked_select(all_inter_card, all_grp == p).reshape((-1, C))])
        card_gt_p= torch.einsum("bc->c", [torch.masked_select(all_card_gt, all_grp == p).reshape((-1, C))])
        card_pred_p= torch.einsum("bc->c", [torch.masked_select(all_card_pred, all_grp == p).reshape((-1, C))])
        dice_3d = (2 * inter_card_p + 1e-8) / ((card_pred_p + card_gt_p)+ 1e-8)
        avd = (card_pred_p + card_gt_p - 2 * inter_card_p + 1e-8) / (card_gt_p + 1e-8)
        if pprint:
            dice_3d = torch.round(dice_3d * 10**2) / (10**2)
            print(p,dice_3d)
        batch_dice[i,...] = dice_3d
        batch_avd[i,...] = avd

    indices = torch.tensor(metric_axis)
    dice_3d = torch.index_select(batch_dice, 1, indices)
    avd = torch.index_select(batch_avd, 1, indices)
    dice_3d_mean = dice_3d.mean(dim=0)
    avd_mean = avd.mean(dim=0)
    print('metric_axis dice',dice_3d_mean)
    dice_3d_sd = dice_3d.std(dim=0)
    avd_sd = avd.std(dim=0)
    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean, dice_3d_sd])
    [avd, avd_sd] = map_(lambda t: t.mean(), [avd_mean, avd_sd])

    return dice_3d.item(), dice_3d_sd.item(), avd.item(), avd_sd.item()




