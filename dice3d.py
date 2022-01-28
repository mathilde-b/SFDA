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
from MySampler import Sampler
import os
from utils import id_, map_, class2one_hot, resize_im, soft_size
from utils import simplex, sset, one_hot, dice_batch
from argparse import Namespace
import os
import pandas as pd

import imageio

def hd3dn(all_grp,all_card_gt,all_pred,all_gt,all_pnames,metric_axis,pprint=False,do_hd=0):
    list(filter(lambda a: a != "0.0", all_pnames))
    list(filter(lambda a: a != 0.0, all_pnames))
    _,C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    list(filter(lambda a: a != 0.0, unique_patients))
    unique_patients = unique_patients[unique_patients != torch.ones_like(unique_patients)*666]
    unique_patients = [u.item() for u in unique_patients] 
    batch_hd = torch.zeros((len(unique_patients), C))
    for i, p in enumerate(unique_patients):
        #print(p)
        try:
            bool_p = [int(re.split('_',re.split('slice',x.item())[1])[0])==p for x in all_pnames]
            data = "whs"
        except:
            bool_p = [int(re.split('_',re.split('Subj_',x.item())[1])[0])==p for x in all_pnames]
            data = "ivd"
        slices_p = all_pnames[bool_p]
        if do_hd >0:
            #print(bool_p,"bool_p")
            all_gt_p = all_gt[bool_p,:]
            all_pred_p = all_pred[bool_p,:] 
            sn_p = [int(re.split('_',x)[1]) for x in slices_p]
            ord_p = np.argsort(sn_p)
            label_gt = all_gt_p[ord_p,...]
            label_pred = all_pred_p[ord_p,...]
            hd_3d_var_vec= [None] * C
            for j in range(0,C):
                label_pred_c = numpy.copy(label_pred)
                label_pred_c[label_pred_c!=j]=0
                label_pred_c[label_pred_c==j]=1
                label_gt_c = numpy.copy(label_gt)
                label_gt_c[label_gt!=j]=0
                label_gt_c[label_gt==j]=1
                if len(np.unique(label_pred_c))>1 and len(np.unique(label_gt_c))>1:
                    if data=="whs":
                        hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[0.6,0.44,0.44]).item()
                    elif data=="ivd":
                        hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[1.25,1.25,2]).item()
                else:
                    hd_3d_var_vec[j]=np.NaN
            hd_3d_var=torch.from_numpy(np.asarray(hd_3d_var_vec))# np.nanmean(hd_3d_var_vec)
        dice_3d = (2 * inter_card_p + 1e-8) / ((card_pred_p + card_gt_p)+ 1e-8)
        if pprint and do_hd:
            print(p,hd_3d_var) 
        if do_hd>0:
            batch_hd[i,...] = hd_3d_var

    indices = torch.tensor(metric_axis)
    hd_3d = torch.index_select(batch_hd, 1, indices)

    hd_3d_mean = hd_3d.mean(dim=0)
    hd_3d_mean = torch.round(hd_3d_mean * 10**4) / (10**4)

    hd_3d_sd = hd_3d.std(dim=0)
    hd_3d_sd = torch.round(hd_3d_sd * 10**4) / (10**4)

    print('metric_axis hd 3d',np.round(hd_3d_mean,2), 'mean ',np.round(hd_3d_mean.mean(),2),'std ',np.round(hd_3d_sd,2), 'std mean', np.round(hd_3d_sd.mean(),2))
    [hd_3d, hd_3d_sd] = map_(lambda t: t.mean(), [hd_3d_mean, hd_3d_sd])
    return hd_3d.item(), hd_3d_sd.item()


def dice3dn2(all_grp,all_inter_card,all_card_gt,all_card_pred,all_pred,all_gt,all_pnames,metric_axis,pprint=False,do_hd=0,best_epoch_val=0):
    list(filter(lambda a: a != "0.0", all_pnames))
    list(filter(lambda a: a != 0.0, all_pnames))
    _,C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    list(filter(lambda a: a != 0.0, unique_patients))
    unique_patients = unique_patients[unique_patients != torch.ones_like(unique_patients)*666]
    unique_patients = [u.item() for u in unique_patients] 
    batch_dice = torch.zeros((len(unique_patients), C))
    batch_hd = torch.zeros((len(unique_patients), C))
    batch_asd = torch.zeros((len(unique_patients), C))
    if do_hd>0:
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

    # do HD
    #if dice_3d_mean.mean()>best_epoch_val:
    if dice_3d_mean.mean()>0:
        for i, p in enumerate(unique_patients):
            try:
                #bool_p = [int(re.split('_',re.split('Case',x.item())[1])[0])==p for x in all_pnames]
                bool_p = [int(re.split('_',re.split('slice',x.item())[1])[0])==p for x in all_pnames]
                #bool_p = [int(re.split('_',re.split('Subj_',x.item())[1])[0])==p for x in all_pnames]
                data = "saml"
            except:
                bool_p = [int(re.split('_',re.split('Case',x.item())[1])[0])==p for x in all_pnames]
                #bool_p = [int(re.split('_',re.split('Subj_',x.item())[1])[0])==p for x in all_pnames]
                data = "ivd"
            slices_p = all_pnames[bool_p]
            #if do_hd >0 or dice_3d_mean.mean()>best_epoch_val:
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
                            #print(np.unique(label_pred_c),np.unique(label_gt_c))
                            #hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[0.6,0.44,0.44]).item()
                            #print(assd(label_pred_c, label_gt_c).item())

                            asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item() #ASD IN VOXEL TO COMPARE
                            hd_3d_var_vec[j] = asd_3d_var_vec[j]
                        elif data=="ivd":
                            hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[1.25,1.25,2]).item()
                            asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item() #ASD IN VOXEL TO COMPARE TO PNP PAPER
                    else:
                        #print(len(np.unique(label_pred_c)),len(np.unique(label_gt_c)))
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

    #if dice_3d_mean.mean()<best_epoch_val:
    #if dice_3d_mean.mean()<0:
    #    print('dice 3d',np.round(dice_3d_mean*100,2), 'mean ',np.round(100*dice_3d_mean.mean(),2),'std ',np.round(100*dice_3d_sd,2), 'std mean', np.round(100*dice_3d_sd.mean(),2))
    #else:
    print('dice 3d',np.round(dice_3d_mean*100,2), 'mean ',np.round(100*dice_3d_mean.mean(),2),'std ',np.round(100*dice_3d_sd,2), 'std mean', np.round(100*dice_3d_sd.mean(),2),'hd_3d',hd_3d_mean, 'hd std',hd_3d_sd,'asd',asd_3d_mean,'mean ',asd_3d_mean.mean())

    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean, dice_3d_sd])
    [hd_3d, hd_3d_sd] = map_(lambda t: t.mean(), [hd_3d_mean, hd_3d_sd])
    [asd_3d, asd_3d_sd] = map_(lambda t: t.mean(), [asd_3d_mean, asd_3d_sd])
    if pprint:
        print('asd_3d_mean',asd_3d_mean.mean(), "asd_3d_sd", asd_3d_sd, "hd_3d_mean", hd_3d_mean, "hd_3d_sd", hd_3d_sd)
        print("hd_3d_mean", hd_3d_mean, "hd_3d_sd", hd_3d_sd)
        #print("in AA,LA,LV,Myo order:",asd_3d_sd.tolist()[3,1,2,0])
    return dice_3d.item(), dice_3d_sd.item(), asd_3d.item(), asd_3d_sd.item()


def dice3dn(all_grp,all_inter_card,all_card_gt,all_card_pred,all_pred,all_gt,all_pnames,metric_axis,pprint=False,do_hd=0):
    #print(all_card_gt.shape)
    list(filter(lambda a: a != "0.0", all_pnames))
    list(filter(lambda a: a != 0.0, all_pnames))
    #print(all_pnames,"all_pnames")
    _,C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    list(filter(lambda a: a != 0.0, unique_patients))
    #unique_patients = unique_patients[unique_patients!="0.0"]
    #unique_patients = unique_patients[unique_patients!=0.0]
    #print(sum(unique_patients == 0))
    #print(all_pred.sum())
    #print(all_gt.sum())
    #print(np.unique(all_pred),np.unique(all_gt),"np.unique(all_pred),np.unique(all_gt)")

    unique_patients = unique_patients[unique_patients != torch.ones_like(unique_patients)*666]
    unique_patients = [u.item() for u in unique_patients] 
    #unique_patients = unique_patients[unique_patients != 666]
    #print(unique_patients)
    batch_dice = torch.zeros((len(unique_patients), C))
    batch_hd = torch.zeros((len(unique_patients), C))
    batch_asd = torch.zeros((len(unique_patients), C))
    #hd_3d = torch.zeros((len(unique_patients)))
    #hd95_3d = torch.zeros((len(unique_patients)))
    #hd95_3d_bis = torch.zeros((len(unique_patients)))
    #hd_3d_var = torch.zeros((len(unique_patients)))
    #asd_3d = torch.zeros((len(unique_patients)))
    #ravd_3d = torch.zeros((len(unique_patients)))
    #print(len(all_pred))
    if do_hd>0:
        all_pred = all_pred.cpu().numpy()
        all_gt = all_gt.cpu().numpy()
    for i, p in enumerate(unique_patients):
        #print(p)
        try:
            bool_p = [int(re.split('_',re.split('slice',x.item())[1])[0])==p for x in all_pnames]
            data = "whs"
        except:
            bool_p = [int(re.split('_',re.split('Subj_',x.item())[1])[0])==p for x in all_pnames]
            data = "ivd"
        #print(bool_p,"bool_p")
        #print(np.sum(bool_p))
        #bool_p = bool_p.detach() 
        #print(all_pnames[1:10])
        #print(len(np.unique(all_pnames)))
        #print(bool_p[1:10])
        #print(all_grp[:,1])
        slices_p = all_pnames[bool_p]
        #print(all_pred.shape,all_grp.shape, "pred versus all_grp")
        #print((all_grp[:,1] == p).shape,'all_grp[:,1] == p shape')
        #print(all_grp,'all_grp')
        #print(sum(all_grp[:,1] == p),'all_grp[:,1] == p sum')
        #all_gt_p = all_gt[all_grp[:,1] == p,:,:]
        #all_gt_p = all_gt[bool_p,:]
        #all_pred_p = all_pred[all_grp[:,1] == p,:,:].cpu() 
        #all_pred_p = all_pred[bool_p,:] 
        #print(all_gt,all_pred)
        #print(all_gt_p.shape,len(slices_p)) # should be the same
        #sn_p = [int(re.split('_',x)[2].split('.')[0]) for x in slices_p]
        #ord_p = np.argsort(sn_p)
        #label_gt = all_gt_p[ord_p,...]
        #print(all_gt_p, all_pred_p)
        #label_pred = all_pred_p[ord_p,...]
        #print(label_gt.sum())
        #print(all_gt_p, all_pred_p)
        inter_card_p = torch.einsum("bc->c", [torch.masked_select(all_inter_card, all_grp == p).reshape((-1, C))])
        card_gt_p= torch.einsum("bc->c", [torch.masked_select(all_card_gt, all_grp == p).reshape((-1, C))])
        card_pred_p= torch.einsum("bc->c", [torch.masked_select(all_card_pred, all_grp == p).reshape((-1, C))])
        #all_pred = all_pred.cpu().numpy()
        #all_gt = all_gt.cpu().numpy()
        #hd_3d[i]= hd(label_pred, label_gt,[1.25,1.25,2] ).item()
        if do_hd >0:
            #print(slices_p)
            #all_pred = all_pred.cpu().numpy()
            #print(all_pred.shape)
            #all_gt = all_gt.cpu().numpy()
            all_gt_p = all_gt[bool_p,:]
            all_pred_p = all_pred[bool_p,:] 
            #sn_p = [int(re.split('_',x)[2].split('.')[0]) for x in slices_p]
            #print(slices_p)
            #sn_p = [int(re.split('_',re.split('slice',x)[1])[0]) for x in slices_p]
            sn_p = [int(re.split('_',x)[1]) for x in slices_p]
            #print(sn_p)
            ord_p = np.argsort(sn_p)
            #print(ord_p)
            label_gt = all_gt_p[ord_p,...]
            label_pred = all_pred_p[ord_p,...]
            #print(np.unique(label_pred))
            hd_3d_var_vec= [None] * C
            hd_3d_var_vec2= [None] * C
            hd_3d_var_vec3= [None] * C
            asd_3d_var_vec = [None] * C
            #hd_3d_var_vec[:] = np.NaN
            #asd_3d_vec[:] = np.NaN
            #print(np.unique(label_pred), np.unique(label_gt))
            for j in range(0,C):
                #print(j)
                #print(np.unique(label_pred),"j",j, np.unique(label_pred))
                label_pred_c = numpy.copy(label_pred)
                label_pred_c[label_pred_c!=j]=0
                label_pred_c[label_pred_c==j]=1
                label_gt_c = numpy.copy(label_gt)
                label_gt_c[label_gt!=j]=0
                label_gt_c[label_gt==j]=1
                #print(np.unique(label_pred_c),np.unique(label_gt_c),"np.unique(label_pred_c),np.unique(label_gt_c)")
                #hd_3d_var_vec[j-1] = hd_var(label_pred_c, label_gt_c,[1.25,1.25,2],1,do_hd).item()
                if len(np.unique(label_pred_c))>1 and len(np.unique(label_gt_c))>1:
                    if data=="whs":
                        #print("data:whs")
                        hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[0.6,0.44,0.44]).item()
                        #hd_3d_var_vec2[j] = hd(label_pred_c, label_gt_c,[0.6,0.44,0.44]).item()
                        #hd_3d_var_vec3[j] = hd(label_pred_c, label_gt_c,[1,1,1]).item()
                        asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item() #ASD IN VOXEL TO COMPARE
                        #print("hd100:",round(hd(label_pred_c, label_gt_c,[0.6,0.44,0.44]).item()), ", hd100 in vox",round(hd(label_pred_c, label_gt_c,[1,1,1]).item()), ", hd95:",round(hd_3d_var_vec[j]))
                    elif data=="ivd":
                        #print("data:ivd")
                        hd_3d_var_vec[j] = hd95(label_pred_c, label_gt_c,[1.25,1.25,2]).item()
                        asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item() #ASD IN VOXEL TO COMPARE TO PNP PAPER
                    #print(hd_var(label_pred_c, label_gt_c,[1.25,1.25,2],1,do_hd).item())
                    #asd_3d_var_vec[j] = assd(label_pred_c, label_gt_c).item()
                else:
                    hd_3d_var_vec[j]=np.NaN
                    asd_3d_var_vec[j] = np.NaN
            #print("hd_3d_var_vec", hd_3d_var_vec)
            hd_3d_var=torch.from_numpy(np.asarray(hd_3d_var_vec))# np.nanmean(hd_3d_var_vec)
            #hd_3d_var2=torch.from_numpy(np.asarray(hd_3d_var_vec2))# np.nanmean(hd_3d_var_vec)
            #hd_3d_var3=torch.from_numpy(np.asarray(hd_3d_var_vec3))# np.nanmean(hd_3d_var_vec)
            asd_3d_var=torch.from_numpy(np.asarray(asd_3d_var_vec))# np.nanmean(hd_3d_var_vec)
            #asd_3d[i]=np.nanmean(asd_3d_vec)
        #hd95_3d_bis[i]= hd95(label_pred, label_gt,[1.25,2,1.25]).item()
        #ravd_3d[i]= abs(ravd(label_pred, label_gt))
        #print(hd_3d[i])
        #if p == 0:
        #    print("inter_card_p:",inter_card_p.detach())
        #    print("card_gt_p:", card_gt_p.detach())
        #    print("card_pred_p:",card_pred_p.detach())
        #print(card_gt_p.shape)
        dice_3d = (2 * inter_card_p + 1e-8) / ((card_pred_p + card_gt_p)+ 1e-8)
        if pprint and do_hd:
            #dice_3d = torch.round(dice_3d * 10**4) / (10**4)
            print(p,dice_3d.cpu(),hd_3d_var, asd_3d_var) 
        if pprint and not(do_hd):
            print(p,dice_3d.cpu())
        batch_dice[i,...] = dice_3d
        if do_hd>0:
            batch_hd[i,...] = hd_3d_var
            batch_asd[i,...] = asd_3d_var

    indices = torch.tensor(metric_axis)
    dice_3d = torch.index_select(batch_dice, 1, indices)
    hd_3d = torch.index_select(batch_hd, 1, indices)
    asd_3d = torch.index_select(batch_asd, 1, indices)

    dice_3d_mean = dice_3d.mean(dim=0)
    hd_3d_mean = hd_3d.mean(dim=0)
    asd_3d_mean = asd_3d.mean(dim=0)
    dice_3d_mean = torch.round(dice_3d_mean * 10**4) / (10**4)
    hd_3d_mean = torch.round(hd_3d_mean * 10**4) / (10**4)
    asd_3d_mean = torch.round(asd_3d_mean * 10**4) / (10**4)

    dice_3d_sd = dice_3d.std(dim=0)
    hd_3d_sd = hd_3d.std(dim=0)
    asd_3d_sd = asd_3d.std(dim=0)
    dice_3d_sd = torch.round(dice_3d_sd * 10**4) / (10**4)
    hd_3d_sd = torch.round(hd_3d_sd * 10**4) / (10**4)
    asd_3d_sd = torch.round(asd_3d_sd * 10**4) / (10**4)

    #hd95_3d_sd = hd95_3d.std()
    #hd95_3d_mean = hd95_3d.mean()
    #hdvar_3d_mean = hd_3d_var.mean()
    #hdvar_3d_sd = hd_3d_var.std()
    #ravd_3d_sd = ravd_3d.std()
    #ravd_3d_mean = ravd_3d.mean()
    print('metric_axis dice 3d',np.round(dice_3d_mean*100,2), 'mean ',np.round(100*dice_3d_mean.mean(),2),'std ',np.round(100*dice_3d_sd,2), 'std mean', np.round(100*dice_3d_sd.mean(),2))
    if pprint:
        print('metric_axis asd 3d',np.round(asd_3d_mean,2), 'mean ',np.round(asd_3d_mean.mean(),2),'std ',np.round(asd_3d_sd,2), 'std mean', np.round(asd_3d_sd.mean(),2))
        print('metric_axis hd 3d',np.round(hd_3d_mean,2), 'mean ',np.round(hd_3d_mean.mean(),2),'std ',np.round(hd_3d_sd,2), 'std mean', np.round(hd_3d_sd.mean(),2))
         
        print("DICE",np.round(dice_3d_mean[0].item()*100,1), " $\pm$ ", np.round(dice_3d_sd[0].item()*100,1),
                "&",np.round(dice_3d_mean[1].item()*100,1), " $\pm$ ", np.round(dice_3d_sd[1].item()*100,1),
                "&",np.round(dice_3d_mean[2].item()*100,1), " $\pm$ ", np.round(dice_3d_sd[2].item()*100,1), 
                "&",np.round(dice_3d_mean[3].item()*100,1), " $\pm$ ", np.round(dice_3d_sd[3].item()*100,1), 
                "&",np.round(dice_3d_mean.mean().item()*100,1), " $\pm$ ", np.round(dice_3d_sd.mean().item()*100,1))
        print("HD", np.round(hd_3d_mean[0].item(),1), " $\pm$ ", np.round(hd_3d_sd[0].item(),1),
                "&",np.round(hd_3d_mean[1].item(),1), " $\pm$ ", np.round(hd_3d_sd[1].item(),1),
                "&",np.round(hd_3d_mean[2].item(),1), " $\pm$ ", np.round(hd_3d_sd[2].item(),1),
                "&",np.round(hd_3d_mean[3].item(),1), " $\pm$ ", np.round(hd_3d_sd[3].item(),1),
                "&",np.round(hd_3d_mean.mean().item(),1), " $\pm$ ", np.round(hd_3d_sd.mean().item(),1))
        print("ASD", np.round(asd_3d_mean[0].item(),1), " $\pm$ ", np.round(asd_3d_sd[0].item(),1),
                "&",np.round(asd_3d_mean[1].item(),1), " $\pm$ ", np.round(asd_3d_sd[1].item(),1),
                "&",np.round(asd_3d_mean[2].item(),1), " $\pm$ ", np.round(asd_3d_sd[2].item(),1),
                "&",np.round(asd_3d_mean[3].item(),1), " $\pm$ ", np.round(asd_3d_sd[3].item(),1),
                "&",np.round(asd_3d_mean.mean().item(),1), " $\pm$ ", np.round(asd_3d_sd.mean().item(),1))
        

    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean, dice_3d_sd])
    [hd_3d, hd_3d_sd] = map_(lambda t: t.mean(), [hd_3d_mean, hd_3d_sd])
    [asd_3d, asd_3d_sd] = map_(lambda t: t.mean(), [asd_3d_mean, asd_3d_sd])
    if pprint:
        print('asd_3d_mean',asd_3d_mean, "asd_3d_sd", asd_3d_sd, "hd_3d_mean", hd_3d_mean, "hd_3d_sd", hd_3d_sd)
    #return dice_3d.item(), dice_3d_sd.item(), hd_3d_mean.item(), hd_3d_sd.item(), asd_3d_mean.item(),asd_3d_sd.item()
    return dice_3d.item(), dice_3d_sd.item(), hd_3d.item(), hd_3d_sd.item()


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


def dice3d(base_folder, folder, subfoldername, grp_regex, gt_folder, C):
    if base_folder == '':
        work_folder = Path(folder, subfoldername)
    else:
        work_folder = Path(base_folder,folder, subfoldername)
    #print(work_folder)
    filenames = map_(lambda p: str(p.name), work_folder.glob("*.png"))
    grouping_regex: Pattern = re.compile(grp_regex)

    stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)
    patients: List[str] = [match.group(0) for match in matches]

    unique_patients: List[str] = list(set(patients))
    #print(unique_patients)
    batch_dice = torch.zeros((len(unique_patients), C))
    for i, patient in enumerate(unique_patients):
        patient_slices = [f for f in stems if f.startswith(patient)]
        w,h = [256,256]
        n = len(patient_slices)
        t_seg = np.ndarray(shape=(w, h, n))
        t_gt = np.ndarray(shape=(w, h, n))
        for slice in patient_slices:
            slice_nb = int(re.split(grp_regex, slice)[1])
            seg = imageio.imread(str(work_folder)+'/'+slice+'.png')
            gt = imageio.imread(str(gt_folder )+'/'+ slice+'.png')
            if seg.shape != (w, h):
                seg = resize_im(seg, 36)
            if gt.shape != (w, h):
                gt = resize_im(gt, 36)
            seg[seg == 255] = 1
            t_seg[:, :, slice_nb] = seg
            t_gt[:, :, slice_nb] = gt
        t_seg = torch.from_numpy(t_seg)
        t_gt = torch.from_numpy(t_gt)
        batch_dice[i,...] = dice_batch(class2one_hot(t_seg,3), class2one_hot(t_gt,3))[0] # do not save the interclasses etcetc
        #df = pd.DataFrame({"val_batch_dice": batch_dice})
        #df.to_csv(Path(savefolder, 'dice_3d.csv'), float_format="%.4f", index_label="epoch")
    return batch_dice.mean(dim=0), batch_dice.std(dim=0)





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

if __name__ == "__main__":


    #args = Namespace(base_folder='', folders='fs_Wat_on_Inn_n', subfolders='iter000', gt_folder='data/all_transverse/train/GT/', save_folder='results/Inn/combined_adv_lambda_e1', grp_regex="Subj_\\d+_")
    #run_dices(args)

    '''
    data = 'Inn_from_server'
    subfolder = 'val'
    #
    # #DA 3d dice
    # result_fold = 'data/ivd2/'+subfolder
    # args = Namespace(base_folder='data/ivd2/'+subfolder,  folders=['fs_Wat_on_'+data], subfolders=['iter000'], gt_folder='data/ivd2/'+subfolder+'/GT/',save_folder=result_fold+'/fs_Wat_on_'+data+'_check', grp_regex="Subj_\\d+_")
    # run_dices(args)
    #
    #Constrained loss 3d dice on val
    result_fold = 'results/'+data+'/'
    methodname = next(os.walk(result_fold))[1]
    #foldernames = [result_fold + m for m in methodname]
    foldernames = [result_fold + 'presize_heaviest']

    for i, f in enumerate(foldernames):
        print(f)
        subfolders_all = next(os.walk(f))[1]
        grouping_regex = re.compile('iter')
        matches = map_(grouping_regex.search, subfolders_all)
        subfolders = [match.string+'/val/' for match in matches if type(match) != type(None)]
        args = Namespace(base_folder='', folders=[f], subfolders=subfolders, gt_folder='data/ivd2/'+subfolder+'/GT/', save_folder=f, grp_regex="Subj_\\d+_")
        run_dices(args)


    #Constrained loss 3d dice on test
    # result_fold = 'results_test/'+data+'/'
    # methodname = next(os.walk(result_fold))[1]
    #
    # for i, m in enumerate(methodname):
    #     print(m)
    #     args = Namespace(base_folder=result_fold, folders=[m], subfolders=['iter000'], gt_folder='data/ivd2/test/GT/', save_folder=result_fold+m, grp_regex="Subj_\\d+_")
    #     run_dices(args)
    
    '''




